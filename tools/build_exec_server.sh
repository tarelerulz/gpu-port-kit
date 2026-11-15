#!/bin/sh
# Build tools/exec_server_fixed: the resident server (runner/exec_server2.c). Set EXEC_SRC to build another copy.
# Spec constant 0 must not be the local size: gemm12 reads M from it (see GPU_MAP.md section 16).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VKL=""; [ -d "$HOME/opt/pios-vulkan/lib" ] && VKL="-L$HOME/opt/pios-vulkan/lib"   # only when the author's Vulkan folder exists; otherwise the system libvulkan
SRC="${EXEC_SRC:-$HERE/../runner/exec_server2.c}"
sed -e 's/uint32_t sd = local;/uint32_t sd = 0xffffffffu;   \/* fix: spec constant 0 is not the local size *\//' "$SRC" > "$HERE/exec_server_fixed.c"
gcc -O2 -o "$HERE/exec_server_fixed" "$HERE/exec_server_fixed.c" -lvulkan $VKL
rm -f "$HERE/exec_server_fixed.c"
echo "built $HERE/exec_server_fixed"
