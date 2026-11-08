// SPDX-License-Identifier: MIT
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
static double nowsec(void){struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed %d line %d\n", #x, r_, __LINE__); exit(1); } } while (0)
#define ARENA_FLOATS (24u << 20)
#define WEIGHT_FLOATS (48u << 20)
static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev; static VkQueue queue;
static uint32_t qfam; static VkCommandPool pool; static VkPhysicalDeviceMemoryProperties mp;
typedef struct { VkBuffer buf; VkDeviceMemory mem; float *map; size_t bytes; } Buf;
static Buf mkbuf(size_t bytes) {
    Buf b = {0}; b.bytes = bytes;
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    CK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(dev, b.buf, &rq);
    uint32_t ti = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((rq.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { ti = i; break; }
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = rq.size, .memoryTypeIndex = ti };
    CK(vkAllocateMemory(dev, &ai, NULL, &b.mem));
    CK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    CK(vkMapMemory(dev, b.mem, 0, bytes, 0, (void **)&b.map));
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
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: exec_server weights.bin < requests\n"); return 1; }
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ic = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    CK(vkCreateInstance(&ic, NULL, &inst));
    uint32_t np = 0; vkEnumeratePhysicalDevices(inst, &np, NULL); VkPhysicalDevice pd[4]; vkEnumeratePhysicalDevices(inst, &np, pd); phys = pd[0];
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL); VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
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
        CK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        CK(vkBeginCommandBuffer(cb, &cbi));
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        for (int i = 0; i < nops; i++) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, get_pipe(ops[i].name, ops[i].local));
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, ops[i].push);
            vkCmdDispatch(cb, ops[i].gx, ops[i].gy, 1);
            VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        }
        CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
        CK(vkResetFences(dev, 1, &fence));
        CK(vkQueueSubmit(queue, 1, &si, fence));
        CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 120ull * 1000000000ull));
        f = fopen(outpath, "wb"); fwrite(A.map + out_off, 4, out_n, f); fclose(f);
        free(ops);
        printf("done %.1f\n", (nowsec() - t0) * 1e3); fflush(stdout);
    }
    return 0;
}
