# Patches for hang reporting on the Pi 400 GPU (V3D)

Without these patches, a GPU job that never finishes is reset by the kernel, but Vulkan reports
**success** and the output is garbage. With them, the job's fence fails and `vkWaitForFences`
returns `VK_ERROR_DEVICE_LOST`.

Two parts are needed:

| Part | Patch | Base |
|---|---|---|
| Kernel `v3d` module | `kernel/v3d-fence-error.patch` | raspberrypi/linux commit `89050b1059997d38d55462b323b099a6436dc10d` (Raspberry Pi kernel 6.12.75, `drivers/gpu/drm/v3d`) |
| Mesa Vulkan driver (broadcom) | `mesa/vk-drm-syncobj-fence-error.patch` | Mesa 25.3.6, `src/vulkan/runtime/vk_drm_syncobj.c` |

## What the patches change

- **Kernel:** every V3D job records the fences of the jobs it waits for at submit time. If one of
  them failed, the job fails too. Previously the scheduler did not propagate that error, so the
  no-op job that carries the user's fence succeeded. The cache-clean job also fails when the compute
  job it follows was dropped as hung.
- **Mesa:** after a successful wait on a binary fence, the driver reads the fence status through its
  sync file. A failed fence returns `VK_ERROR_DEVICE_LOST`.

## Build the kernel module

Use the same kernel version as the running system (`uname -r`).

```sh
# 1. Get the kernel source at the base commit (sparse checkout of the V3D driver is enough)
git clone --filter=blob:none --sparse https://github.com/raspberrypi/linux.git kernel-src
cd kernel-src
git checkout 89050b1059997d38d55462b323b099a6436dc10d
git sparse-checkout set drivers/gpu/drm/v3d

# 2. Apply the patch from the kernel source root
git apply /path/to/patches/kernel/v3d-fence-error.patch

# 3. Build against kernel headers for the same version (must match `uname -r`).
#    The headers tree needs a Makefile and scripts/ (a complete build tree).
#    Check first:  ls /lib/modules/$(uname -r)/build/Makefile
cd drivers/gpu/drm/v3d
make -C /path/to/kernel-headers-for-$(uname -r) M=$PWD CONFIG_DRM_V3D=m modules
```

If `/lib/modules/$(uname -r)/build` is missing or is a dangling link (it was on the test system),
get a complete headers tree for the same version from a system that has one (for example a
Raspberry Pi OS root filesystem's `/usr/src/linux-headers-$(uname -r)`), copy it, and point `-C`
at the copy. The test build used a copied tree this way.

The result is `v3d.ko` in that directory.

## Load and unload (needs root)

```sh
# Stop anything that uses the GPU first (for example, GPU-based services). Then:
sudo rmmod v3d
sudo insmod /absolute/path/to/v3d.ko
cat /sys/module/v3d/srcversion     # check the version you built

# Back to the stock module: reboot, or
sudo rmmod v3d && sudo modprobe v3d
```

Use an absolute path with `insmod`. Under `sudo`, `~` expands to `/root`.

Notes: an out-of-tree module taints the kernel, and the patched module is not persistent across
reboots unless installed separately.

## Build the Mesa driver

```sh
# Mesa 25.3.6 source (or a checkout with the same file), then:
patch -p1 < /path/to/patches/mesa/vk-drm-syncobj-fence-error.patch
meson setup build -Dvulkan-drivers=broadcom -Dgallium-drivers= \
      -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dllvm=disabled \
      -Dopengl=false -Dgles1=disabled -Dgles2=disabled -Dvideo-codecs= \
      --prefix=$HOME/opt/mesa-v3dv
ninja -C build install
```

Point the Vulkan loader at the result:

```sh
export VK_ICD_FILENAMES=$HOME/opt/mesa-v3dv/icd.json
```

The kit reads this path from `gpuport/config.py` (`VK_ENV`).

## Verify

With both parts in place, a hanging shader makes `vkWaitForFences` return `VK_ERROR_DEVICE_LOST`
(-4). The kit's test suite (`tests/run_all.py`) passes on this combination. Without the patches,
the same test returns success.

Caveats: this is not an upstream change. It was tested on this one kernel and Mesa build only.
