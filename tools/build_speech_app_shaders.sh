#!/bin/sh
# Build the older speech-app shaders (shaders/speech-app/*.comp) into SPIR-V.
# Output: work/speech-app-shaders/*.spv (or the folder given as $1). Needs glslc.
# Point GPUPORT_SHADERS at the output folder. The old gemm12 shader is not published: the kit's own
# shaders/gemm12 replaces it (copied into the output folder).
# Built with: glslc --target-env=vulkan1.1 -O (47 of these rebuilt byte-identical to the author's originals).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../shaders/speech-app"
OUT="${1:-$HERE/../work/speech-app-shaders}"
mkdir -p "$OUT"
for f in "$SRC"/*.comp; do
  b="$(basename "$f" .comp)"
  glslc --target-env=vulkan1.1 -O -I "$HERE/../shaders" -I "$SRC" "$f" -o "$OUT/$b.spv"
  cp "$f" "$OUT/"
done
# the old gemm12 is replaced by the kit's own (shaders/gemm12.*); programs that load only this folder (port/gpu_encoder2.py) need it here
cp "$HERE/../shaders/gemm12.spv" "$HERE/../shaders/gemm12.comp" "$OUT/"
echo "built $(ls "$OUT"/*.spv | wc -l) shaders in $OUT"
