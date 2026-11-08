#!/bin/sh
# Build tools/exec_big: the speech-app exec.c batch runner with a 24M-float arena and 48M-float weights.
# Needed by tests/nemo8_rebuild.py (a 10 s clip needs 18.7M arena floats).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HOME/speech-app/port/gpu/exec.c"
sed -e 's/#define ARENA_FLOATS (8u << 20)/#define ARENA_FLOATS (24u << 20)/' \
    -e 's/#define WEIGHT_FLOATS (32u << 20)/#define WEIGHT_FLOATS (48u << 20)/' \
    -e 's/uint32_t sd = local;/uint32_t sd = 0xffffffffu;   \/* fix: spec constant 0 must not be the local size (shaders read M from it) *\//' "$SRC" > "$HERE/exec_big.c"
gcc -O2 -o "$HERE/exec_big" "$HERE/exec_big.c" -lvulkan -L"$HOME/opt/pios-vulkan/lib" 2>/dev/null
rm -f "$HERE/exec_big.c"
echo "built $HERE/exec_big"
