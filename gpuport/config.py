"""Paths and limits. Override with environment variables."""
import os

# Directory holding exec.c's binary and the *.spv shaders (exec loads <name>.spv from its cwd).
SHADER_DIR = os.environ.get('GPUPORT_SHADERS', os.path.expanduser('~/oido/port/gpu'))
EXEC_BIN = os.environ.get('GPUPORT_EXEC', os.path.join(SHADER_DIR, 'exec'))
# Scratch files (arena/plan/out). Not in /tmp on purpose.
WORK_DIR = os.environ.get('GPUPORT_WORK', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work'))
# PiOS Vulkan loader + v3dv driver (same as ~/opt/pios-vulkan/env.sh).
VK_ENV = {
    'LD_LIBRARY_PATH': os.path.expanduser('~/opt/pios-vulkan/lib'),
    'VK_ICD_FILENAMES': os.path.expanduser('~/opt/pios-vulkan/icd/broadcom_icd.json'),
}
# exec.c fixed buffer sizes (floats). Plans must stay inside them.
EXEC_ARENA_FLOATS = 8 << 20
EXEC_WEIGHT_FLOATS = 32 << 20
# Rules measured on this Pi (see ~/doc/agent-memory/2026-11-03-gpu-porting-rules.md).
REGION_GAP = 131072       # floats after every arena region
ALIGN = 64                # floats, arena region start
MIN_WORKGROUPS = 24       # fewer gave wrong gemm12 output at some shapes
MAX_GROUP_DIM = 65535     # dispatch limit per dimension
MAX_PUSH = 32             # exec sends 32 uint push words
GEMM12_MAX_ROWS = 251     # gemm12 only correct for row blocks <= 251
