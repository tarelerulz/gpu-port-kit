#!/bin/sh
# Build tools/exec_hybrid: runner/exec_hybrid.c (exec_server2 + raw V3D CSD jobs for the hand-written QPU matmul, see work/asm/qpu_mm.py).
# Needs libdrm headers (-I/usr/include/libdrm) and shaders/qpu/mm16.qpu (python3 work/asm/qpu_mm.py build, then copy).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VKL=""; [ -d "$HOME/opt/pios-vulkan/lib" ] && VKL="-L$HOME/opt/pios-vulkan/lib"   # only when the author's Vulkan folder exists; otherwise the system libvulkan
sed -e 's/uint32_t sd = local;/uint32_t sd = 0xffffffffu;/' "$HERE/../runner/exec_hybrid.c" > "$HERE/exec_hybrid_tmp.c"
gcc -O2 -I/usr/include/libdrm -o "$HERE/exec_hybrid" "$HERE/exec_hybrid_tmp.c" -lvulkan $VKL
rm -f "$HERE/exec_hybrid_tmp.c"
echo "built $HERE/exec_hybrid"
