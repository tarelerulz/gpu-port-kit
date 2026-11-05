#!/bin/sh
# Build tools/exec_server_fixed: oido's exec_server2.c (resident server) with the specialization-constant fix.
# Spec constant 0 must not be the local size: gemm12 reads M from it (see GPU_MAP.md section 16).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HOME/oido/port/gpu/exec_server2.c"
sed -e 's/uint32_t sd = local;/uint32_t sd = 0xffffffffu;   \/* fix: spec constant 0 is not the local size *\//' "$SRC" > "$HERE/exec_server_fixed.c"
gcc -O2 -o "$HERE/exec_server_fixed" "$HERE/exec_server_fixed.c" -lvulkan -L"$HOME/opt/pios-vulkan/lib" 2>/dev/null
rm -f "$HERE/exec_server_fixed.c"
echo "built $HERE/exec_server_fixed"
