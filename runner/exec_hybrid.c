// SPDX-License-Identifier: MIT
// Hybrid resident GPU server (exec_hybrid, 2026-11-12 experiment): exec_server2 plus hand-written QPU matmul jobs.
// Plan op "qpu_mm" (not a shader) is submitted as a raw V3D CSD job on the shared arena / weights (dma-buf), ordered with the Vulkan
// segments around it by explicit sync files (Vulkan SYNC_FD semaphore -> DRM syncobj -> CSD job -> sync file -> Vulkan semaphore).
// usage: exec_hybrid weights.bin mm16.qpu [mm16rs.qpu [mm16relu.qpu [mm16rm.qpu]]] < requests      (same request protocol as exec_server2)
// qpu_mm push words: 0 K, 1 ntiles, 2 A_off, 3 A_pitch, 4 O_off, 5 O_pitch (floats), 6 W_off (floats in the weights buffer), 7 tile_words, 8 M
// push 10 = 1: the weight stream is in the arena (qpu_pack output) instead of the weights buffer
// qpu_mmrs: same, plus 9 = float bits of the scale: out += scale * (acc + bias) (second code image at +32 KB); qpu_mmrelu: relu epilogue, no bias-free change (third image at +64 KB); qpu_mmrm: output row-major [m][n] with row pitch push[5] floats (fourth image at +96 KB)
// Resident GPU server (exec_server2) for the Pi 400 V3D Vulkan path. Copyright (c) 2026 gpu-port-kit contributors. See LICENSE.
// Run a plan of compute dispatches in one command buffer with a barrier after each.
// usage: exec plan.txt arena_in.bin weights_in.bin arena_out.bin out_off out_n
// plan line: <shader_name> <local> <gx> <gy> <p0> ... <p31>   (shader spv loaded from gpu/<name>.spv)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include <drm.h>
#include <drm/v3d_drm.h>
#include <pthread.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
/* Fast read from write-combined GPU memory (A.map) using 64-byte NEON burst loads (like heart-nano round 3). */
static void copy_from_wc(void *dst, const void *src, size_t bytes) {
#if defined(__aarch64__)
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    size_t i = 0;
    for (; i + 64 <= bytes; i += 64) {
        uint8x16_t v0 = vld1q_u8(s + i);
        uint8x16_t v1 = vld1q_u8(s + i + 16);
        uint8x16_t v2 = vld1q_u8(s + i + 32);
        uint8x16_t v3 = vld1q_u8(s + i + 48);
        vst1q_u8(d + i,      v0);
        vst1q_u8(d + i + 16, v1);
        vst1q_u8(d + i + 32, v2);
        vst1q_u8(d + i + 48, v3);
    }
    if (i < bytes) memcpy(d + i, s + i, bytes - i);
#else
    memcpy(dst, src, bytes);
#endif
}
static int argmax_f32(const float *row, int n) {
    int best = 0; float bv = row[0];
#if defined(__aarch64__)
    float32x4_t vmax = vdupq_n_f32(-1e38f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t m0 = vmaxq_f32(vld1q_f32(row + i), vld1q_f32(row + i + 4));
        float32x4_t m1 = vmaxq_f32(vld1q_f32(row + i + 8), vld1q_f32(row + i + 12));
        vmax = vmaxq_f32(vmax, vmaxq_f32(m0, m1));
    }
    float hi = vmaxvq_f32(vmax);
    for (int j = i; j < n; j++) if (row[j] > hi) hi = row[j];
    for (int j = 0; j < n; j++) if (row[j] == hi) return j;
#endif
    for (int j = 1; j < n; j++) if (row[j] > bv) { bv = row[j]; best = j; }
    return best;
}
/* 2026-11-16: keep one CPU core busy while a request runs. The ondemand governor otherwise leaves the (idle) CPU at its lowest clock, and
   every GPU job launch (interrupt -> scheduler -> next job, ~400 per clip) runs at that clock. GPUPORT_SPIN=0 = off. Capped at 30 s. */
static volatile int spin_on = 0; static int spin_enabled = 1; static unsigned spin_gen = 0; static double spin_cap = 30.0;
static pthread_mutex_t spin_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t spin_cv = PTHREAD_COND_INITIALIZER;
static void *spinner(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&spin_mu); while (!spin_on) pthread_cond_wait(&spin_cv, &spin_mu); unsigned g = spin_gen; double cap = spin_cap; pthread_mutex_unlock(&spin_mu);
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        while (spin_on && spin_gen == g) {
            for (volatile int i = 0; i < 20000; i++) ;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9 > cap) { pthread_mutex_lock(&spin_mu); if (spin_gen == g) spin_on = 0; pthread_mutex_unlock(&spin_mu); break; }
        }
    }
    return NULL;
}
/* on = 1 with a cap in seconds (request: 30, "wake" line: 0.25); every call starts a new generation so an expiring wake cannot stop a request's spin */
static void spin_set(int on, double cap) { if (!spin_enabled) return; pthread_mutex_lock(&spin_mu); spin_gen++; spin_cap = cap; spin_on = on; pthread_cond_signal(&spin_cv); pthread_mutex_unlock(&spin_mu); }
static double nowsec(void){struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed %d line %d\n", #x, r_, __LINE__); exit(1); } } while (0)
#define ARENA_FLOATS (24u << 20)
#define WEIGHT_FLOATS (48u << 20)
static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev; static VkQueue queue;
static uint32_t qfam; static VkCommandPool pool; static VkPhysicalDeviceMemoryProperties mp;
typedef struct { VkBuffer buf; VkDeviceMemory mem; float *map; size_t bytes; int fd; } Buf;
static int hybrid = 0;
static Buf mkbuf(size_t bytes) {
    Buf b = {0}; b.bytes = bytes; b.fd = -1;
    VkExternalMemoryBufferCreateInfo ebi = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = hybrid ? &ebi : NULL, .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    CK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(dev, b.buf, &rq);
    uint32_t ti = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((rq.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { ti = i; break; }
    VkExportMemoryAllocateInfo exi = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = hybrid ? &exi : NULL, .allocationSize = rq.size, .memoryTypeIndex = ti };
    CK(vkAllocateMemory(dev, &ai, NULL, &b.mem));
    CK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    CK(vkMapMemory(dev, b.mem, 0, bytes, 0, (void **)&b.map));
    if (hybrid) {
        PFN_vkGetMemoryFdKHR gfd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
        VkMemoryGetFdInfoKHR gi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = b.mem, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
        CK(gfd(dev, &gi, &b.fd));
    }
    return b;
}
typedef struct { char name[32]; uint32_t local; uint32_t gx, gy; uint32_t push[32]; } Op;
typedef struct { char name[32]; uint32_t local; VkPipeline pipe; } Pipe;
static Pipe pipes[32]; static int npipes = 0;
static VkPipelineLayout pl; static VkDescriptorSetLayout dsl; static VkDescriptorSet ds;
static VkPipeline get_pipe(const char *name, uint32_t local) {
    for (int i = 0; i < npipes; i++) if (!strcmp(pipes[i].name, name) && pipes[i].local == local) return pipes[i].pipe;
    char path[256]; snprintf(path, sizeof path, "%s.spv", name);
    FILE *fs = fopen(path, "rb"); if (!fs) { fprintf(stderr, "no shader %s\n", path); exit(1); }
    fseek(fs, 0, SEEK_END); long len = ftell(fs); rewind(fs);
    uint32_t *code = malloc(len); fread(code, 1, len, fs); fclose(fs);
    VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = len, .pCode = code };
    VkShaderModule mod; CK(vkCreateShaderModule(dev, &smi, NULL, &mod));
    VkSpecializationMapEntry sme = { .constantID = 0, .offset = 0, .size = 4 };
    uint32_t sd = 0xffffffffu;   /* fix 2026-11-05: spec constant 0 must not be the local size (gemm12 reads M from it) */
    VkSpecializationInfo sinfo = { .mapEntryCount = 1, .pMapEntries = &sme, .dataSize = 4, .pData = &sd };
    VkPipelineShaderStageCreateInfo ss = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main", .pSpecializationInfo = &sinfo };
    VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = ss, .layout = pl };
    VkPipeline p; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &p));
    strncpy(pipes[npipes].name, name, 31); pipes[npipes].local = local; pipes[npipes].pipe = p; npipes++;
    return p;
}

/* ---- raw V3D DRM side ---- */
static int dfd = -1;
static uint32_t hA, hW, hC; static uint32_t vaA, vaW, vaC; static uint32_t *ubo; static const size_t CODE_BO = 1 << 20;
#define DCK(x) do { int r_ = (x); if (r_ < 0) { fprintf(stderr, "%s failed errno %d (%s) line %d\n", #x, errno, strerror(errno), __LINE__); exit(1); } } while (0)
static uint32_t import_bo(int fd, uint32_t *va) {
    struct drm_prime_handle ph = { .fd = fd }; DCK(ioctl(dfd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &ph));
    struct drm_v3d_get_bo_offset go = { .handle = ph.handle }; DCK(ioctl(dfd, DRM_IOCTL_V3D_GET_BO_OFFSET, &go)); *va = go.offset; return ph.handle;
}
static uint32_t syncobj_new(void) { struct drm_syncobj_create c = {0}; DCK(ioctl(dfd, DRM_IOCTL_SYNCOBJ_CREATE, &c)); return c.handle; }
static void syncobj_free(uint32_t h) { struct drm_syncobj_destroy d = { .handle = h }; ioctl(dfd, DRM_IOCTL_SYNCOBJ_DESTROY, &d); }
static void syncobj_from_syncfile(uint32_t h, int fd) { struct drm_syncobj_handle sh = { .handle = h, .flags = DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE, .fd = fd }; DCK(ioctl(dfd, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &sh)); }
static int syncobj_to_syncfile(uint32_t h) { struct drm_syncobj_handle sh = { .handle = h, .flags = DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE, .fd = -1 }; DCK(ioctl(dfd, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &sh)); return sh.fd; }
static void csd_submit(uint32_t code_va, uint32_t uniforms_va, uint32_t in_sync, uint32_t out_sync) {
    uint32_t bos[3] = { hA, hW, hC };
    struct drm_v3d_submit_csd c; memset(&c, 0, sizeof c);
    c.cfg[0] = 1u << 16; c.cfg[1] = 1u << 16; c.cfg[2] = 1u << 16;
    c.cfg[3] = ((16 * 16 + 15) / 16 - 1) << 12 | (16 << 8) | 16;     /* 16 invocations per workgroup, 16 workgroups per supergroup */
    c.cfg[4] = 8 - 1;                                                 /* 8 threads */
    c.cfg[5] = code_va; c.cfg[6] = uniforms_va;
    c.bo_handles = (uintptr_t)bos; c.bo_handle_count = 3; c.in_sync = in_sync; c.out_sync = out_sync;
    DCK(ioctl(dfd, DRM_IOCTL_V3D_SUBMIT_CSD, &c));
}
static void drm_setup(Buf *A, Buf *W, int ncode, char **codepaths) {
    dfd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC); DCK(dfd);
    hA = import_bo(A->fd, &vaA); hW = import_bo(W->fd, &vaW);
    struct drm_v3d_create_bo cb = { .size = CODE_BO }; DCK(ioctl(dfd, DRM_IOCTL_V3D_CREATE_BO, &cb)); hC = cb.handle; vaC = cb.offset;
    struct drm_v3d_mmap_bo mb = { .handle = hC }; DCK(ioctl(dfd, DRM_IOCTL_V3D_MMAP_BO, &mb));
    uint8_t *m = mmap(NULL, CODE_BO, PROT_READ | PROT_WRITE, MAP_SHARED, dfd, mb.offset); if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    size_t n = 0;
    for (int ci = 0; ci < ncode && ci < 14; ci++) {
        FILE *f = fopen(codepaths[ci], "rb"); if (!f) { fprintf(stderr, "no qpu code %s\n", codepaths[ci]); exit(1); }
        n += ci < 6 ? fread(m + ci * 32768, 1, 32768, f) : fread(m + 196608 + (ci - 6) * 8192, 1, 8192, f); fclose(f);   /* ci >= 6: 4 x 8 tiling kernels (qpu_mm4.py) */
    }
    ubo = (uint32_t *)(m + 262144);
    fprintf(stderr, "qpu code %zu bytes at va 0x%x; arena va 0x%x weights va 0x%x\n", n, vaC, vaA, vaW);
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: exec_hybrid weights.bin mm16.qpu < requests\n"); return 1; }
    hybrid = 1;
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ic = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    CK(vkCreateInstance(&ic, NULL, &inst));
    uint32_t np = 0; vkEnumeratePhysicalDevices(inst, &np, NULL); VkPhysicalDevice pd[4]; vkEnumeratePhysicalDevices(inst, &np, pd); phys = pd[0];
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL); VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio };
    const char *exts[] = { "VK_KHR_external_memory", "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf", "VK_KHR_external_semaphore", "VK_KHR_external_semaphore_fd" };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .enabledExtensionCount = 5, .ppEnabledExtensionNames = exts };
    CK(vkCreateDevice(phys, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfam, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
    CK(vkCreateCommandPool(dev, &pci, NULL, &pool));
    Buf A = mkbuf((size_t)ARENA_FLOATS * 4), W = mkbuf((size_t)WEIGHT_FLOATS * 4);
    memset(A.map, 0, A.bytes);
    FILE *f = fopen(argv[1], "rb"); fread(W.map, 4, WEIGHT_FLOATS, f); fclose(f);   /* weights: once */
    VkDescriptorSetLayoutBinding bd[2] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT } };
    VkDescriptorSetLayoutCreateInfo dsli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bd };
    CK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));
    VkPushConstantRange pr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 128 };
    VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkDescriptorPoolSize dps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 };
    VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps };
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dsa = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    CK(vkAllocateDescriptorSets(dev, &dsa, &ds));
    VkDescriptorBufferInfo bi0 = { .buffer = A.buf, .offset = 0, .range = VK_WHOLE_SIZE }, bi1 = { .buffer = W.buf, .offset = 0, .range = VK_WHOLE_SIZE };
    VkWriteDescriptorSet wr[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi0 },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi1 } };
    vkUpdateDescriptorSets(dev, 2, wr, 0, NULL);
    VkCommandBufferAllocateInfo cba = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cba, &cb));
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; CK(vkCreateFence(dev, &fci, NULL, &fence));
    drm_setup(&A, &W, argc - 2, &argv[2]);
    VkExportSemaphoreCreateInfo esi = { .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO, .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
    VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &esi };
    VkSemaphore semS, semW; CK(vkCreateSemaphore(dev, &sci, NULL, &semS)); CK(vkCreateSemaphore(dev, &sci, NULL, &semW));
    PFN_vkGetSemaphoreFdKHR gsf = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(dev, "vkGetSemaphoreFdKHR");
    PFN_vkImportSemaphoreFdKHR isf = (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(dev, "vkImportSemaphoreFdKHR");
    VkCommandBuffer cbs[256];
    {
        VkCommandBufferAllocateInfo cba_all = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 256 };
        CK(vkAllocateCommandBuffers(dev, &cba_all, cbs));
    }
    uint32_t so_pool[2] = { syncobj_new(), syncobj_new() };
    int bp_off = getenv("GPUPORT_BLOCKPASS_OFF") != NULL;
    int sk_full = getenv("GPUPORT_SK_FULL") != NULL;
    float *rd_buf = NULL; size_t rd_cap = 0;
    Op *ops = malloc(sizeof(Op) * 4096);
    { const char *sp = getenv("GPUPORT_SPIN"); if (sp && sp[0] == '0') spin_enabled = 0; }
    if (spin_enabled) { pthread_t th; if (pthread_create(&th, NULL, spinner, NULL)) spin_enabled = 0; else pthread_detach(th); }
    fprintf(stderr, "ready\n"); fflush(stderr);
    char req[4096], planpath[1024], arenapath[1024], outpath[1024];
    unsigned long out_off, out_n;
    while (fgets(req, sizeof req, stdin)) {
        /* request: <plan> <arena_file> <arena_offset_floats> <out_file> <out_off> <out_n>
           or greedy argmax: argmax <plan> <arena_file> <arena_offset_floats> <out_off> <T> <V1> */
        unsigned long arena_off = 0, arg_T = 0, arg_V1 = 0;
        int greedy_mode = 0;
        if (!strncmp(req, "wake", 4)) { spin_set(1, 0.25); continue; }   /* no reply: the request follows */
        if (!strncmp(req, "argmax ", 7)) {
            if (sscanf(req + 7, "%1023s %1023s %lu %lu %lu %lu", planpath, arenapath, &arena_off, &out_off, &arg_T, &arg_V1) != 6) {
                printf("err bad request\n"); fflush(stdout); continue;
            }
            greedy_mode = 1;
            out_n = arg_T * arg_V1;
        } else if (sscanf(req, "%1023s %1023s %lu %1023s %lu %lu", planpath, arenapath, &arena_off, outpath, &out_off, &out_n) != 6) {
            printf("err bad request\n"); fflush(stdout); continue;
        }
        double t0 = nowsec();
        spin_set(1, 30.0);
        f = fopen(arenapath, "rb"); size_t na = fread(A.map + arena_off, 4, ARENA_FLOATS - arena_off, f); fclose(f); (void)na;
        /* plan */
        int nops = 0; char line[2048];
        /* 2026-11-16 binary plan (*.bplan, gpuport.plan.Plan.write_bin): "GPB1", op count, then the Op structs as they are (172 bytes each) */
        size_t pl_ = strlen(planpath); int binplan = pl_ > 6 && !strcmp(planpath + pl_ - 6, ".bplan");
        f = fopen(planpath, binplan ? "rb" : "r");
        if (f && binplan) {
            uint32_t hd[2] = {0, 0};
            if (fread(hd, 4, 2, f) != 2 || hd[0] != 0x31425047u || hd[1] > 4096 || fread(ops, sizeof(Op), hd[1], f) != hd[1]) { fclose(f); spin_set(0, 0.0); printf("err bad binary plan\n"); fflush(stdout); continue; }
            nops = (int)hd[1]; for (int i = 0; i < nops; i++) ops[i].name[31] = 0;
        }
        while (f && !binplan && fgets(line, sizeof line, f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            Op *o = &ops[nops]; char *tok = strtok(line, " \n");
            strncpy(o->name, tok, 31); o->local = atoi(strtok(NULL, " \n")); o->gx = atoi(strtok(NULL, " \n")); o->gy = atoi(strtok(NULL, " \n"));
            for (int i = 0; i < 32; i++) o->push[i] = (uint32_t)strtoul(strtok(NULL, " \n"), NULL, 10);
            nops++;
        }
        if (f) fclose(f);
        CK(vkResetCommandPool(dev, pool, 0));
        int ncb = 0, i0 = 0, need_wait = 0, nasm = 0, so_cur = 0; uint32_t h_prev = 0;
        CK(vkResetFences(dev, 1, &fence));
        for (int i = 0; i <= nops; i++) {
            int is_asm = (i < nops) && (!strcmp(ops[i].name, "qpu_mm") || !strcmp(ops[i].name, "qpu_mmrs") || !strcmp(ops[i].name, "qpu_mmrelu") || !strcmp(ops[i].name, "qpu_mmrm") || !strcmp(ops[i].name, "qpu_mmsk") || !strcmp(ops[i].name, "qpu_mmsilu") || !strcmp(ops[i].name, "qpu_mmh") || !strcmp(ops[i].name, "qpu_mmskh") || !strncmp(ops[i].name, "qpu4_", 5) || !strncmp(ops[i].name, "qpu1_", 5));
            if (i < nops && !is_asm) continue;
            /* ---- Vulkan segment [i0, i) ---- */
            uint32_t h_in = 0;
            if (i > i0) {
                if (h_prev) {                                  /* previous asm job's completion becomes this segment's wait */
                    int fd2 = syncobj_to_syncfile(h_prev); h_prev = 0;
                    VkImportSemaphoreFdInfoKHR ii = { .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR, .semaphore = semW, .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, .fd = fd2 };
                    CK(isf(dev, &ii)); need_wait = 1;
                }
                if (ncb >= 256) { fprintf(stderr, "too many Vulkan segments\n"); exit(1); }
                VkCommandBuffer cb = cbs[ncb++];
                VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
                CK(vkBeginCommandBuffer(cb, &cbi));
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
                for (int j = i0; j < i; j++) {
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, get_pipe(ops[j].name, ops[j].local));
                    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, ops[j].push);
                    vkCmdDispatch(cb, ops[j].gx, ops[j].gy, 1);
                    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
                    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
                }
                CK(vkEndCommandBuffer(cb));
                VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = need_wait, .pWaitSemaphores = &semW, .pWaitDstStageMask = &ws, .commandBufferCount = 1, .pCommandBuffers = &cb,
                                    .signalSemaphoreCount = is_asm, .pSignalSemaphores = &semS };
                CK(vkQueueSubmit(queue, 1, &si, is_asm ? VK_NULL_HANDLE : fence)); need_wait = 0;
                if (is_asm) {
                    VkSemaphoreGetFdInfoKHR gi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, .semaphore = semS, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
                    int fd1 = -1; CK(gsf(dev, &gi, &fd1));
                    h_in = so_pool[so_cur]; so_cur ^= 1; syncobj_from_syncfile(h_in, fd1); close(fd1);
                }
            } else if (is_asm && h_prev) { h_in = h_prev; h_prev = 0; }   /* two asm jobs in a row: chain on the previous out syncobj */
            if (!is_asm) { if (i == nops && i == i0) { fprintf(stderr, "plan must end with a Vulkan op\n"); exit(1); } break; }
            /* ---- asm job ---- */
            /* qpu1_* (2026-11-16, qpu_mm1.py): 1 vector x 8 columns, row tails of <= 16 rows; same weight stream, uniforms and thread table as qpu4 with one block */
            int q1 = !strncmp(ops[i].name, "qpu1_", 5);
            if (q1 && ops[i].push[8] > 16) { fprintf(stderr, "qpu1: M %u > 16\n", ops[i].push[8]); exit(1); }
            if (q1 || !strncmp(ops[i].name, "qpu4_", 5)) {   /* 4 vectors x 8 columns kernel (qpu_mm4.py): k-major out, static weights in the weights buffer */
                uint32_t *pu4 = &ubo[nasm * 128]; uint32_t *q4 = ops[i].push;
                uint32_t K4 = q4[0], NTL = q4[1], M4 = q4[8], tw = q4[7];
                uint32_t nblk = (M4 + 63) / 64; if (nblk < 1) nblk = 1; if (nblk > 8) { fprintf(stderr, "qpu4: M %u > 512\n", M4); exit(1); }
                uint32_t G = 8 / nblk;
                pu4[0] = q4[3] * 4; pu4[1] = q4[5] * 4; pu4[2] = tw * 4 * G; pu4[3] = 8 * G * q4[5] * 4; pu4[4] = K4 / 2; pu4[5] = vaC + 262144 + nasm * 512 + 64; pu4[6] = q4[9];
                uint32_t *tb4 = pu4 + 16;
                for (uint32_t t = 0; t < 8; t++) {
                    uint32_t mb = t % nblk, g = t / nblk; int act = (t < G * nblk) && (g < NTL);
                    if (!act) { mb = 0; g = 0; }
                    uint32_t ntl = act ? (NTL - g + G - 1) / G : 1, v = act ? M4 - 64 * mb : 0;
                    tb4[t * 8 + 0] = vaA + 4 * (q4[2] + 64 * mb); tb4[t * 8 + 1] = vaW + 4 * (q4[6] + g * tw);
                    tb4[t * 8 + 2] = vaA + 4 * (q4[4] + g * 8 * q4[5] + 64 * mb); tb4[t * 8 + 3] = ntl; tb4[t * 8 + 4] = v;
                    tb4[t * 8 + 5] = tb4[t * 8 + 6] = tb4[t * 8 + 7] = 0;
                }
                int img4 = !strcmp(ops[i].name + 5, "mmrelu") ? 1 : !strcmp(ops[i].name + 5, "mmrs") ? 2 : 0;
                if (q1) img4 += 5;                             /* code images 11-13 (slots 5-7 of the 8 KB images) */
                uint32_t h_out4 = so_pool[so_cur]; so_cur ^= 1;
                csd_submit(vaC + 196608 + img4 * 8192, vaC + 262144 + nasm * 512, h_in, h_out4);
                h_prev = h_out4; nasm++; i0 = i + 1;
                continue;
            }
            uint32_t *pu = &ubo[nasm * 128]; uint32_t *q = ops[i].push;      /* 512-byte block: 16 parameter words, then the 8-thread work table at +64 B */
            uint32_t PI = q[3] * 4, OP = q[5] * 4;
            pu[0] = vaA + q[2] * 4; pu[1] = PI; pu[2] = PI + 64; pu[3] = 2 * PI; pu[4] = 64; pu[5] = vaA + q[4] * 4; pu[6] = OP; pu[7] = OP - 64; pu[8] = 16 * OP;
            pu[9] = (q[10] ? vaA : vaW) + q[6] * 4;     /* push 10: weight stream lives in the arena (packed activations) */ pu[10] = q[7] * 4; pu[11] = q[0] / 2; pu[14] = q[9]; pu[13] = 0;
            {   /* per-thread table: [m block base, first tile group, tile groups, tiles, valid m, passes, 0, 0]; blocks of 32 rows.
                   nblk <= 8: 8 // nblk tile groups share the n tiles; nblk > 8: every thread owns block t and repeats for passes of 256 rows */
                uint32_t M = q[8], NTl = q[1], nblk = (M + 31) / 32; if (nblk < 1) nblk = 1; uint32_t *tb = pu + 16;
                int headsjob = !strcmp(ops[i].name, "qpu_mmh") || !strcmp(ops[i].name, "qpu_mmskh");
                uint32_t bestp = 0, bestc = 0xffffffffu, pmin = (nblk + 7) / 8;
                for (uint32_t pp = pmin; pp <= nblk; pp++) {
                    uint32_t ng = (nblk + pp - 1) / pp, G = 8 / ng, c = pp * ((NTl + G - 1) / G);
                    if (c < bestc) { bestc = c; bestp = pp; }
                    if (headsjob || bp_off) break;
                }
                if (nblk > 256 / 32 * 4) { fprintf(stderr, "M too large\n"); exit(1); }
                {
                    uint32_t ng = (nblk + bestp - 1) / bestp, G = 8 / ng;
                    if (headsjob && q[17]) while (G > 1 && (NTl - 1) % G) G--;
                    for (uint32_t t = 0; t < 8; t++) {
                        uint32_t bgrp = t % ng, g = t / ng;
                        int ntl = (t < G * ng && g < NTl) ? (int)((NTl - g + G - 1) / G) : 0;
                        uint32_t pss = nblk - bgrp * bestp; if (pss > bestp) pss = bestp;
                        if (ntl <= 0) { uint32_t idle[8] = {0, 0, 1, 1, 0, 1, 0, 0}; memcpy(tb + t * 8, idle, 32); }
                        else { uint32_t row[8] = { bgrp * bestp * 32, g, G, (uint32_t)ntl, M, pss, 0, 0 }; memcpy(tb + t * 8, row, 32); }
                    }
                }
                pu[12] = vaC + 262144 + nasm * 512 + 64;
            }
            int img = !strcmp(ops[i].name, "qpu_mmrs") ? 1 : !strcmp(ops[i].name, "qpu_mmrelu") ? 2 : !strcmp(ops[i].name, "qpu_mmrm") ? 3 : !strcmp(ops[i].name, "qpu_mmsk") ? 4 : !strcmp(ops[i].name, "qpu_mmsilu") ? 5 : 0;
            int hd = !strcmp(ops[i].name, "qpu_mmh") ? 1 : !strcmp(ops[i].name, "qpu_mmskh") ? 2 : 0;
            if (hd == 2) img = 4;
            if (hd && q[8] > 256) { fprintf(stderr, "heads job: M %u > 256\n", q[8]); exit(1); }
            if (img == 4) {
                uint32_t TPf = q[5], LS = (TPf + 1) * 4, NS = TPf * 4, R0 = q[12];
                pu[5] = vaA + (uint32_t)((int32_t)q[4] - (int32_t)(R0 * TPf)) * 4; pu[6] = LS; pu[7] = NS - 16 * LS; pu[8] = 16 * NS; pu[14] = 16 * LS;
                for (uint32_t t = 0; t < 8; t++) { pu[16 + t * 8 + 6] = q[11]; pu[16 + t * 8 + 7] = R0; }
                uint32_t Ms = q[8], NTs = q[1], nbs = (Ms + 31) / 32; if (nbs < 1) nbs = 1;
                if (nbs <= 8 && !sk_full) {
                    uint32_t Gs = 8 / nbs; int Tk = (int)q[11], R0i = (int)R0;
                    for (uint32_t t = 0; t < 8; t++) {
                        uint32_t b = t % nbs, g = t / nbs; uint32_t *row = pu + 16 + t * 8;
                        int mlo = (int)b * 32, mhi = mlo + 31; if (mhi > (int)Ms - 1) mhi = (int)Ms - 1;
                        int rlo = R0i - mhi, rhi = R0i + Tk - 1 - mlo; if (rlo < 0) rlo = 0;
                        int tlo = rlo / 16, thi = rhi / 16; if (thi > (int)NTs - 1) thi = (int)NTs - 1;
                        int cnt = thi - tlo + 1, ntl = (t < Gs * nbs && cnt > (int)g) ? (cnt - (int)g + (int)Gs - 1) / (int)Gs : 0;
                        if (ntl <= 0) { row[0] = 0; row[1] = 0; row[2] = 1; row[3] = 1; row[4] = 0; row[5] = 1; }
                        else { row[0] = b * 32; row[1] = (uint32_t)tlo + g; row[2] = Gs; row[3] = (uint32_t)ntl; row[4] = Ms; row[5] = 1; }
                    }
                }
            }
            if (img == 3) { pu[6] = OP; pu[7] = 4 - 16 * OP; pu[8] = 64; pu[14] = 16 * OP; }
            if (hd) {
                uint32_t se = pu[14];
                pu[4] = q[14] * 4;
                if (hd == 1) { pu[13] = q[15] * 4; pu[14] = q[16] * 4; }
                else { pu[13] = se; pu[14] = q[15] * 4; pu[15] = q[16] * 4; }
                for (uint32_t t = 0; t < 8; t++) if (pu[16 + t * 8 + 4]) pu[16 + t * 8 + 5] = q[13];
            }
            uint32_t h_out = so_pool[so_cur]; so_cur ^= 1;
            csd_submit(hd ? vaC + 196608 + (2 + hd) * 8192 : vaC + img * 32768, vaC + 262144 + nasm * 512, h_in, h_out);
            h_prev = h_out; nasm++; i0 = i + 1;
        }
        CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 120ull * 1000000000ull));
        spin_set(0, 0.0);
        if (greedy_mode) {
            if (arg_V1 > rd_cap) { free(rd_buf); rd_cap = arg_V1; rd_buf = malloc(rd_cap * 4); }
            int prev_tok = -1, blank = (int)arg_V1 - 1, first_tok = 1;
            printf("done %.1f", (nowsec() - t0) * 1e3);
            for (unsigned long t = 0; t < arg_T; t++) {
                copy_from_wc(rd_buf, A.map + out_off + t * arg_V1, arg_V1 * 4);
                int a = argmax_f32(rd_buf, (int)arg_V1);
                if (a != prev_tok && a != blank) { printf("%c%d", first_tok ? ' ' : ',', a); first_tok = 0; }
                prev_tok = a;
            }
            printf("\n"); fflush(stdout);
        } else {
            if (out_n > rd_cap) { free(rd_buf); rd_cap = out_n; rd_buf = malloc(rd_cap * 4); }
            copy_from_wc(rd_buf, A.map + out_off, out_n * 4);
            f = fopen(outpath, "wb"); fwrite(rd_buf, 4, out_n, f); fclose(f);
            printf("done %.1f\n", (nowsec() - t0) * 1e3); fflush(stdout);
        }
    }
    free(ops); free(rd_buf);
    return 0;
}
