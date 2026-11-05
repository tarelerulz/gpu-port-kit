"""gpuport: helpers for porting a model to the Pi 400 GPU (V3D via Vulkan) with the exec batch runner.

Modules:
  arena.py    arena regions with the 64-float alignment and 131072-float gap rule
  weights.py  weight blob with 16-float alignment and gemm12 matmul layout (N padded to 12)
  plan.py     plan writer + linter (the measured rules from the oido port)
  ops.py      push-constant layouts for shaders already in use (gemm12, add, ln, glu, silu)
  run.py      run a plan through gpu/exec, prefix runs, per-op timing
  check.py    compare a region against numpy; per-op isolated test; prefix bisect
"""

from .arena import Arena  # noqa: E402
from .weights import Weights  # noqa: E402
from .plan import Plan  # noqa: E402
from . import ops, check  # noqa: E402,F401
