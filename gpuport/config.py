"""Paths and limits. Override with environment variables."""
import os

# Directory holding exec.c's binary and the *.spv shaders (exec loads <name>.spv from its cwd).
SHADER_DIR = os.environ.get('GPUPORT_SHADERS', os.path.expanduser('~/speech-app/port/gpu'))
# Default runner: the kit's fixed build (tools/build_exec_big.sh); speech-app's exec has the specialization-constant bug.
EXEC_BIN = os.environ.get('GPUPORT_EXEC', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools', 'exec_big'))
# Scratch files (arena/plan/out). Not in /tmp on purpose.
WORK_DIR = os.environ.get('GPUPORT_WORK', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work'))
# PiOS Vulkan loader + v3dv driver (same as ~/opt/pios-vulkan/env.sh).
VK_ENV = {
    'LD_LIBRARY_PATH': os.path.expanduser('~/opt/pios-vulkan/lib'),
    'VK_ICD_FILENAMES': os.path.expanduser('~/opt/mesa-v3dv/icd.json'),   # patched Mesa (fence-error fix)
}
# exec.c fixed buffer sizes (floats). Plans must stay inside them.
EXEC_ARENA_FLOATS = 8 << 20
EXEC_WEIGHT_FLOATS = 32 << 20
# Rules measured on this Pi (see <notes>/2026-11-03-gpu-porting-rules.md).
REGION_GAP = 131072       # floats after every arena region
ALIGN = 64                # floats, arena region start
MIN_WORKGROUPS = 24       # fewer gave wrong gemm12 output at some shapes
MAX_GROUP_DIM = 65535     # dispatch limit per dimension
MAX_PUSH = 32             # exec sends 32 uint push words
GEMM12_MAX_ROWS = 251     # speed choice, not a correctness limit (2026-11-05): gemm12 is correct at any row count with
                            # the fixed runner, but time per call jumps above ~251 rows (conv2 at 5020 rows took 2.6 s in one call).

KIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_SHADERS = os.path.join(KIT_ROOT, 'shaders')
STREAM_SHADERS = os.path.expanduser('~/speech-app/port/stream')   # gemm_k5 and other streaming shaders


def merged_shader_dir(extra=(KIT_SHADERS,), base=None):
    """Folder with symlinks to the speech-app shaders plus the kit's own shaders (kit wins on a name clash).
    exec loads <name>.spv from its cwd, so plans that use both need this folder."""
    import glob
    out = os.path.join(WORK_DIR, 'shaders')
    os.makedirs(out, exist_ok=True)
    for src in [base or SHADER_DIR, *extra]:
        for f in glob.glob(os.path.join(src, '*.spv')) + glob.glob(os.path.join(src, '*.comp')):
            dst = os.path.join(out, os.path.basename(f))
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(f), dst)
    return out
