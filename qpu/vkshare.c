// Check 1 (2026-11-12): can the Vulkan arena be shared with a hand-written QPU job (py-videocore6) and stay correctly ordered?
// Vulkan side: exports the arena as a dma-buf, runs a long dispatch chain that writes A (async, no CPU wait), later a second chain that
// post-processes O. Between them the python side submits an assembly matmul CSD job on the imported BO. Driven over a unix socket.
// usage: vkshare <sockfd> <out_file> <K> <N> <nadd> <rounds>      (cwd = directory with add.spv)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <vulkan/vulkan.h>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed %d line %d\n", #x, r_, __LINE__); exit(1); } } while (0)
static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev; static VkQueue queue; static uint32_t qfam;
static VkPhysicalDeviceMemoryProperties mp;
typedef struct { VkBuffer buf; VkDeviceMemory mem; float *map; size_t bytes; int fd; } Buf;
static Buf mkbuf(size_t bytes, int exportable) {
    Buf b = {0}; b.bytes = bytes; b.fd = -1;
    VkExternalMemoryBufferCreateInfo ebi = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = exportable ? &ebi : NULL, .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    CK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(dev, b.buf, &rq);
    uint32_t ti = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((rq.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { ti = i; break; }
    VkExportMemoryAllocateInfo exi = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = exportable ? &exi : NULL, .allocationSize = rq.size, .memoryTypeIndex = ti };
    CK(vkAllocateMemory(dev, &ai, NULL, &b.mem));
    CK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    CK(vkMapMemory(dev, b.mem, 0, bytes, 0, (void **)&b.map));
    if (exportable) {
        PFN_vkGetMemoryFdKHR gfd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
        if (!gfd) { fprintf(stderr, "no vkGetMemoryFdKHR\n"); exit(2); }
        VkMemoryGetFdInfoKHR gi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = b.mem, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
        CK(gfd(dev, &gi, &b.fd));
    }
    return b;
}
static VkPipelineLayout pl; static VkDescriptorSetLayout dsl; static VkDescriptorSet ds;
static VkPipeline mkpipe(const char *name) {
    char path[256]; snprintf(path, sizeof path, "%s.spv", name);
    FILE *fs = fopen(path, "rb"); if (!fs) { fprintf(stderr, "no shader %s\n", path); exit(1); }
    fseek(fs, 0, SEEK_END); long len = ftell(fs); rewind(fs);
    uint32_t *code = malloc(len); if (fread(code, 1, len, fs) != (size_t)len) exit(1); fclose(fs);
    VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = len, .pCode = code };
    VkShaderModule mod; CK(vkCreateShaderModule(dev, &smi, NULL, &mod));
    VkSpecializationMapEntry sme = { .constantID = 0, .offset = 0, .size = 4 }; uint32_t sd = 0xffffffffu;
    VkSpecializationInfo sinfo = { .mapEntryCount = 1, .pMapEntries = &sme, .dataSize = 4, .pData = &sd };
    VkPipelineShaderStageCreateInfo ss = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main", .pSpecializationInfo = &sinfo };
    VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = ss, .layout = pl };
    VkPipeline p; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &p)); return p;
}
static void send_fd(int sock, int fd, const char *msg) {
    struct msghdr m = {0}; struct iovec iov = { (void *)msg, strlen(msg) }; char cbuf[CMSG_SPACE(sizeof(int))] = {0};
    m.msg_iov = &iov; m.msg_iovlen = 1; m.msg_control = cbuf; m.msg_controllen = sizeof cbuf;
    struct cmsghdr *c = CMSG_FIRSTHDR(&m); c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int)); memcpy(CMSG_DATA(c), &fd, sizeof(int));
    if (sendmsg(sock, &m, 0) < 0) { perror("sendmsg"); exit(1); }
}
static int recv_fd(int sock) {
    char c; struct iovec iov = { &c, 1 }; char cbuf[CMSG_SPACE(sizeof(int))] = {0};
    struct msghdr m = {0}; m.msg_iov = &iov; m.msg_iovlen = 1; m.msg_control = cbuf; m.msg_controllen = sizeof cbuf;
    if (recvmsg(sock, &m, 0) <= 0) exit(3);
    struct cmsghdr *cm = CMSG_FIRSTHDR(&m); int fd = -1; if (cm && cm->cmsg_type == SCM_RIGHTS) memcpy(&fd, CMSG_DATA(cm), sizeof(int)); return fd;
}
static void say(int sock, const char *s) { if (write(sock, s, strlen(s)) < 0) exit(1); }
static char hear(int sock) { char c = 0; if (read(sock, &c, 1) != 1) exit(3); return c; }
static VkCommandBuffer record(VkCommandPool pool, VkPipeline pipe, uint32_t ndisp, uint32_t n, uint32_t dst, uint32_t src) {
    VkCommandBufferAllocateInfo cba = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cba, &cb));
    VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CK(vkBeginCommandBuffer(cb, &cbi));
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    float one = 1.0f; uint32_t push[32] = { n, dst, src, 0 }; memcpy(&push[3], &one, 4);
    for (uint32_t i = 0; i < ndisp; i++) {
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, push);
        vkCmdDispatch(cb, (n + 127) / 128, 1, 1);
        VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    CK(vkEndCommandBuffer(cb)); return cb;
}
int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage\n"); return 1; }
    int sock = atoi(argv[1]); const char *outp = argv[2]; uint32_t K = atoi(argv[3]), N = atoi(argv[4]), nadd = atoi(argv[5]); int rounds = atoi(argv[6]); int wait1 = argc > 7 ? atoi(argv[7]) : 0; int explicit_sync = argc > 8 ? atoi(argv[8]) : 0;
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ic = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    CK(vkCreateInstance(&ic, NULL, &inst));
    uint32_t np = 0; vkEnumeratePhysicalDevices(inst, &np, NULL); VkPhysicalDevice pd[4]; vkEnumeratePhysicalDevices(inst, &np, pd); phys = pd[0];
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(phys, NULL, &ne, NULL); VkExtensionProperties *ep = malloc(ne * sizeof *ep); vkEnumerateDeviceExtensionProperties(phys, NULL, &ne, ep);
    int have_fd = 0, have_dma = 0; for (uint32_t i = 0; i < ne; i++) { if (!strcmp(ep[i].extensionName, "VK_KHR_external_memory_fd")) have_fd = 1; if (!strcmp(ep[i].extensionName, "VK_EXT_external_memory_dma_buf")) have_dma = 1; }
    fprintf(stderr, "device extensions: external_memory_fd %d, external_memory_dma_buf %d\n", have_fd, have_dma);
    if (!have_fd || !have_dma) { say(sock, "X"); return 2; }
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL); VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    float prio = 1.0f; const char *exts[] = { "VK_KHR_external_memory", "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf", "VK_KHR_external_semaphore", "VK_KHR_external_semaphore_fd" };
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .enabledExtensionCount = 5, .ppEnabledExtensionNames = exts };
    CK(vkCreateDevice(phys, &dci, NULL, &dev)); vkGetDeviceQueue(dev, qfam, 0, &queue); vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfam };
    VkCommandPool pool; CK(vkCreateCommandPool(dev, &pci, NULL, &pool));
    uint32_t S = (K + 4) * 256, SO = N * 256;                 // floats: A, ONES (S), O (SO)
    size_t total = (size_t)(2 * S + SO + 4096) * 4;
    Buf A = mkbuf(total, 1), W = mkbuf(4096, 0);
    fprintf(stderr, "exported arena: %zu bytes, dma-buf fd %d\n", total, A.fd);
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
    VkDescriptorBufferInfo bi0 = { .buffer = A.buf, .range = VK_WHOLE_SIZE }, bi1 = { .buffer = W.buf, .range = VK_WHOLE_SIZE };
    VkWriteDescriptorSet wr[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi0 },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi1 } };
    vkUpdateDescriptorSets(dev, 2, wr, 0, NULL);
    VkPipeline padd = mkpipe("add");
    VkCommandBuffer cb1 = record(pool, padd, nadd, S, 0, S);              // A += 1 * ONES, nadd times
    VkCommandBuffer cb2 = record(pool, padd, 1, SO, 2 * S, S);            // O += 1 * ONES
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    send_fd(sock, A.fd, "R");
    for (int round = 0; round < rounds; round++) {
        char c = hear(sock); if (c != '1') return 4;
        for (uint32_t k = 0; k < K + 4; k++) for (uint32_t m = 0; m < 256; m++) A.map[k * 256 + m] = (k < K && m < 251) ? ((k * 31 + m * 7) % 17) * 0.125f : 0.0f;
        for (uint32_t i = 0; i < S; i++) A.map[S + i] = 1.0f;
        for (uint32_t i = 0; i < SO; i++) A.map[2 * S + i] = -1.0f;
        VkFence f1, f2; CK(vkCreateFence(dev, &fci, NULL, &f1)); CK(vkCreateFence(dev, &fci, NULL, &f2));
        if (explicit_sync) {
            // seg 1 signals a SYNC_FD semaphore; the fd goes to python, which makes the assembly CSD job wait on it (syncobj) and
            // returns the CSD job's completion as a sync file that seg 2 waits on. No CPU wait anywhere on the GPU path.
            VkExportSemaphoreCreateInfo esi = { .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO, .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
            VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &esi };
            VkSemaphore sem1, sem2; CK(vkCreateSemaphore(dev, &sci, NULL, &sem1)); CK(vkCreateSemaphore(dev, &sci, NULL, &sem2));
            VkSubmitInfo s1 = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb1, .signalSemaphoreCount = 1, .pSignalSemaphores = &sem1 };
            CK(vkQueueSubmit(queue, 1, &s1, f1));
            PFN_vkGetSemaphoreFdKHR gsf = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(dev, "vkGetSemaphoreFdKHR");
            PFN_vkImportSemaphoreFdKHR isf = (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(dev, "vkImportSemaphoreFdKHR");
            VkSemaphoreGetFdInfoKHR gi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, .semaphore = sem1, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
            int sfd = -1; CK(gsf(dev, &gi, &sfd));
            send_fd(sock, sfd, "S"); close(sfd);
            int cfd = recv_fd(sock);                                   // sync file of the CSD job
            VkImportSemaphoreFdInfoKHR ii = { .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR, .semaphore = sem2, .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, .fd = cfd };
            CK(isf(dev, &ii));
            VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            VkSubmitInfo s2 = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &sem2, .pWaitDstStageMask = &ws, .commandBufferCount = 1, .pCommandBuffers = &cb2 };
            CK(vkQueueSubmit(queue, 1, &s2, f2));
            CK(vkWaitForFences(dev, 1, &f2, VK_TRUE, 60ull * 1000000000ull));
            CK(vkWaitForFences(dev, 1, &f1, VK_TRUE, 60ull * 1000000000ull));
            vkDestroySemaphore(dev, sem1, NULL); vkDestroySemaphore(dev, sem2, NULL);
        } else {
        VkSubmitInfo s1 = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb1 };
        CK(vkQueueSubmit(queue, 1, &s1, f1));
        if (wait1) CK(vkWaitForFences(dev, 1, &f1, VK_TRUE, 60ull * 1000000000ull));   // experiment: CPU wait for seg 1
        say(sock, "S");
        c = hear(sock); if (c != '2') return 5;
        VkSubmitInfo s2 = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb2 };
        CK(vkQueueSubmit(queue, 1, &s2, f2));
        CK(vkWaitForFences(dev, 1, &f2, VK_TRUE, 60ull * 1000000000ull));
        CK(vkWaitForFences(dev, 1, &f1, VK_TRUE, 60ull * 1000000000ull));
        }
        FILE *f = fopen(outp, "wb"); fwrite(A.map + 2 * S, 4, SO, f); fclose(f);
        vkDestroyFence(dev, f1, NULL); vkDestroyFence(dev, f2, NULL);
        say(sock, "D");
    }
    return 0;
}
