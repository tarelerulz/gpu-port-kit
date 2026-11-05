// Print the Vulkan device limits that matter for porting (same loader and ICD as exec).
// build: gcc -O2 -o devinfo devinfo.c -L$HOME/opt/pios-vulkan/lib -lvulkan
#include <stdio.h>
#include <vulkan/vulkan.h>
int main(void) {
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ic = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    VkInstance inst; if (vkCreateInstance(&ic, NULL, &inst)) { puts("no instance"); return 1; }
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice pd[4]; vkEnumeratePhysicalDevices(inst, &n, pd);
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd[0], &p);
    VkPhysicalDeviceSubgroupProperties sg = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sg };
    vkGetPhysicalDeviceProperties2(pd[0], &p2);
    const VkPhysicalDeviceLimits *l = &p.limits;
    printf("device: %s\n", p.deviceName);
    printf("api %u.%u.%u  driver 0x%x  vendor 0x%x device 0x%x\n", VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion),
           VK_VERSION_PATCH(p.apiVersion), p.driverVersion, p.vendorID, p.deviceID);
    printf("maxComputeWorkGroupInvocations %u\n", l->maxComputeWorkGroupInvocations);
    printf("maxComputeWorkGroupSize %u x %u x %u\n", l->maxComputeWorkGroupSize[0], l->maxComputeWorkGroupSize[1], l->maxComputeWorkGroupSize[2]);
    printf("maxComputeWorkGroupCount %u x %u x %u\n", l->maxComputeWorkGroupCount[0], l->maxComputeWorkGroupCount[1], l->maxComputeWorkGroupCount[2]);
    printf("maxComputeSharedMemorySize %u bytes\n", l->maxComputeSharedMemorySize);
    printf("maxStorageBufferRange %u bytes\n", l->maxStorageBufferRange);
    printf("maxPushConstantsSize %u bytes\n", l->maxPushConstantsSize);
    printf("maxDescriptorSetStorageBuffers %u\n", l->maxPerStageDescriptorStorageBuffers);
    printf("subgroupSize %u  subgroup ops 0x%x\n", sg.subgroupSize, sg.supportedOperations);
    printf("minStorageBufferOffsetAlignment %llu bytes\n", (unsigned long long)l->minStorageBufferOffsetAlignment);
    VkPhysicalDeviceMemoryProperties m; vkGetPhysicalDeviceMemoryProperties(pd[0], &m);
    for (uint32_t i = 0; i < m.memoryHeapCount; i++)
        printf("heap %u: %.0f MB%s\n", i, m.memoryHeaps[i].size / 1048576.0, (m.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " device-local" : "");
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        printf("memtype %u: heap %u%s%s%s%s\n", i, m.memoryTypes[i].heapIndex,
               (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : "",
               (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " HOST_VISIBLE" : "",
               (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " HOST_COHERENT" : "",
               (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " HOST_CACHED" : "");
    }
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd[0], &nq, NULL);
    VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(pd[0], &nq, qp);
    for (uint32_t i = 0; i < nq; i++) printf("queue family %u: flags 0x%x count %u\n", i, qp[i].queueFlags, qp[i].queueCount);
    return 0;
}
