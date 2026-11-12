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
    for (int ci = 0; ci < ncode && ci < 4; ci++) {
        FILE *f = fopen(codepaths[ci], "rb"); if (!f) { fprintf(stderr, "no qpu code %s\n", codepaths[ci]); exit(1); }
        n += fread(m + ci * 32768, 1, 32768, f); fclose(f);
    }
    ubo = (uint32_t *)(m + 131072);
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
    fprintf(stderr, "ready\n"); fflush(stderr);
    char req[4096], planpath[1024], arenapath[1024], outpath[1024];
    unsigned long out_off, out_n;
    while (fgets(req, sizeof req, stdin)) {
        /* request: <plan> <arena_file> <arena_offset_floats> <out_file> <out_off> <out_n>  (offset form, persistent arena) */
        unsigned long arena_off = 0;
        if (sscanf(req, "%1023s %1023s %lu %1023s %lu %lu", planpath, arenapath, &arena_off, outpath, &out_off, &out_n) != 6) { printf("err bad request\n"); fflush(stdout); continue; }
        double t0 = nowsec();
        f = fopen(arenapath, "rb"); size_t na = fread(A.map + arena_off, 4, ARENA_FLOATS - arena_off, f); fclose(f); (void)na;
        /* plan */
        Op *ops = malloc(sizeof(Op) * 4096); int nops = 0; char line[2048];
        f = fopen(planpath, "r");
        while (f && fgets(line, sizeof line, f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            Op *o = &ops[nops]; char *tok = strtok(line, " \n");
            strncpy(o->name, tok, 31); o->local = atoi(strtok(NULL, " \n")); o->gx = atoi(strtok(NULL, " \n")); o->gy = atoi(strtok(NULL, " \n"));
            for (int i = 0; i < 32; i++) o->push[i] = (uint32_t)strtoul(strtok(NULL, " \n"), NULL, 10);
            nops++;
        }
        if (f) fclose(f);
        CK(vkResetCommandPool(dev, pool, 0));
        int ncb = 0, i0 = 0, need_wait = 0, nasm = 0; uint32_t h_prev = 0;
        CK(vkResetFences(dev, 1, &fence));
        for (int i = 0; i <= nops; i++) {
            int is_asm = (i < nops) && (!strcmp(ops[i].name, "qpu_mm") || !strcmp(ops[i].name, "qpu_mmrs") || !strcmp(ops[i].name, "qpu_mmrelu") || !strcmp(ops[i].name, "qpu_mmrm"));
            if (i < nops && !is_asm) continue;
            /* ---- Vulkan segment [i0, i) ---- */
            uint32_t h_in = 0; int made_in = 0;
            if (i > i0) {
                if (h_prev) {                                  /* previous asm job's completion becomes this segment's wait */
                    int fd2 = syncobj_to_syncfile(h_prev); syncobj_free(h_prev); h_prev = 0;
                    VkImportSemaphoreFdInfoKHR ii = { .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR, .semaphore = semW, .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, .fd = fd2 };
                    CK(isf(dev, &ii)); need_wait = 1;
                }
                VkCommandBufferAllocateInfo cba = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
                VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cba, &cb)); cbs[ncb++] = cb;
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
                    h_in = syncobj_new(); made_in = 1; syncobj_from_syncfile(h_in, fd1); close(fd1);
                }
            } else if (is_asm && h_prev) { h_in = h_prev; h_prev = 0; made_in = 1; }   /* two asm jobs in a row: chain on the previous out syncobj */
            if (!is_asm) { if (i == nops && i == i0) { fprintf(stderr, "plan must end with a Vulkan op\n"); exit(1); } break; }
            /* ---- asm job ---- */
            uint32_t *pu = &ubo[nasm * 128]; uint32_t *q = ops[i].push;      /* 512-byte block: 16 parameter words, then the 8-thread work table at +64 B */
            uint32_t PI = q[3] * 4, OP = q[5] * 4;
            pu[0] = vaA + q[2] * 4; pu[1] = PI; pu[2] = PI + 64; pu[3] = 2 * PI; pu[4] = 64; pu[5] = vaA + q[4] * 4; pu[6] = OP; pu[7] = OP - 64; pu[8] = 16 * OP;
            pu[9] = (q[10] ? vaA : vaW) + q[6] * 4;     /* push 10: weight stream lives in the arena (packed activations) */ pu[10] = q[7] * 4; pu[11] = q[0] / 2; pu[14] = q[9]; pu[13] = 0;
            {   /* per-thread table: [m block base, first tile group, tile groups, tiles, valid m, passes, 0, 0]; blocks of 32 rows.
                   nblk <= 8: 8 // nblk tile groups share the n tiles; nblk > 8: every thread owns block t and repeats for passes of 256 rows */
                uint32_t M = q[8], NTl = q[1], nblk = (M + 31) / 32; if (nblk < 1) nblk = 1; uint32_t *tb = pu + 16;
                if (nblk > 8) {
                    uint32_t P = (nblk + 7) / 8; if (P > 4) { fprintf(stderr, "M too large\n"); exit(1); }
                    for (uint32_t t = 0; t < 8; t++) { uint32_t row[8] = { t * 32, 0, 1, NTl, M, P, 0, 0 }; memcpy(tb + t * 8, row, 32); }
                } else {
                    uint32_t G = 8 / nblk;
                    for (uint32_t t = 0; t < 8; t++) {
                        int ntl = (t < G * nblk) ? (int)((NTl - t / nblk + G - 1) / G) : 0; if (t / nblk >= NTl) ntl = 0;
                        if (ntl <= 0) { uint32_t idle[8] = {0, 0, 1, 1, 0, 1, 0, 0}; memcpy(tb + t * 8, idle, 32); }
                        else { uint32_t row[8] = { (t % nblk) * 32, t / nblk, G, (uint32_t)ntl, M, 1, 0, 0 }; memcpy(tb + t * 8, row, 32); }
                    }
                }
                pu[12] = vaC + 131072 + nasm * 512 + 64;
            }
            int img = !strcmp(ops[i].name, "qpu_mmrs") ? 1 : !strcmp(ops[i].name, "qpu_mmrelu") ? 2 : !strcmp(ops[i].name, "qpu_mmrm") ? 3 : 0;
            if (img == 3) { pu[6] = OP; pu[7] = 4 - 16 * OP; pu[8] = 64; pu[14] = 16 * OP; }   /* row-major output: row bytes, step after odd acc, tile step, 16 rows */
            uint32_t h_out = syncobj_new();
            csd_submit(vaC + img * 32768, vaC + 131072 + nasm * 512, h_in, h_out);
            if (made_in) syncobj_free(h_in);
            h_prev = h_out; nasm++; i0 = i + 1;
        }
        CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 120ull * 1000000000ull));
        f = fopen(outpath, "wb"); fwrite(A.map + out_off, 4, out_n, f); fclose(f);
        free(ops);
        printf("done %.1f\n", (nowsec() - t0) * 1e3); fflush(stdout);
    }
    return 0;
}
