// Ask the GPU which number formats and math features its shaders can use (Vulkan feature flags).
// build: gcc -O2 -o gpufeatures gpufeatures.c -lvulkan -L$HOME/opt/pios-vulkan/lib
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>
int main(void) {
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ic = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    VkInstance inst; if (vkCreateInstance(&ic, NULL, &inst)) { puts("no instance"); return 1; }
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice pd[4]; vkEnumeratePhysicalDevices(inst, &n, pd);
    VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &f12 };
    VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13 };
    vkGetPhysicalDeviceFeatures2(pd[0], &f2);
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd[0], &p);
    printf("NOTE: these flags are not reliable. float16 and int8 read \"no\" but run correctly (see GPU_MAP.md section 14).\n");
    printf("GPU: %s (Vulkan %u.%u)\n", p.deviceName, VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion));
    printf("16-bit float math in shaders (shaderFloat16):      %s\n", f12.shaderFloat16 ? "yes" : "no");
    printf("8-bit integer math in shaders (shaderInt8):        %s\n", f12.shaderInt8 ? "yes" : "no");
    printf("16-bit integer math (shaderInt16):                 %s\n", f2.features.shaderInt16 ? "yes" : "no");
    printf("64-bit float (shaderFloat64):                      %s\n", f2.features.shaderFloat64 ? "yes" : "no");
    printf("8-bit values can be stored in buffers:              %s\n", f12.storageBuffer8BitAccess ? "yes" : "no");
    printf("integer dot product (shaderIntegerDotProduct):     %s\n", f13.shaderIntegerDotProduct ? "yes" : "no");
    return 0;
}
