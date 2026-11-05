"""Run a plan through gpu/exec (one command buffer, barrier after each dispatch)."""
import os
import re
import subprocess
import tempfile
import numpy as np
from .config import SHADER_DIR, EXEC_BIN, WORK_DIR, VK_ENV, EXEC_ARENA_FLOATS
from .plan import Plan

_SUBMIT = re.compile(r'gpu submit\+wait ([0-9.]+) ms')
_seq = [0]


def _zeros_weights(work):
    p = os.path.join(work, 'zero_weights.bin')
    if not os.path.exists(p):
        os.makedirs(work, exist_ok=True)
        np.zeros(16, '<f4').tofile(p)
    return p


def run(plan, arena_init, out_off, out_n, weights=None, shader_dir=SHADER_DIR, exec_bin=EXEC_BIN,
        work=WORK_DIR, lint=True, arena_floats=EXEC_ARENA_FLOATS):
    """plan: Plan. arena_init: float32 array (words before its end are the start arena; rest is zero).
    weights: path to a float32 .bin, or None. Returns (out float32[out_n], submit_ms)."""
    if isinstance(plan, (list, tuple)):
        plan = Plan(plan)
    if lint:
        errs = [m for lvl, m in plan.lint(shader_dir) if lvl == 'error']
        if errs:
            raise ValueError('plan lint errors:\n  ' + '\n  '.join(errs))
    arena_init = np.asarray(arena_init, np.float32).ravel()
    if arena_init.size > arena_floats or out_off + out_n > arena_floats:
        raise MemoryError('arena or output beyond the exec batch arena')
    os.makedirs(work, exist_ok=True)
    _seq[0] += 1
    base = os.path.join(work, f'run{os.getpid()}_{_seq[0] % 4}')
    plan.write(base + '.plan')
    arena_init.astype('<f4').tofile(base + '.arena_in')
    wpath = weights or _zeros_weights(work)
    env = dict(os.environ, **VK_ENV)
    r = subprocess.run([exec_bin, base + '.plan', base + '.arena_in', wpath, base + '.arena_out',
                        str(out_off), str(out_n)], env=env, cwd=shader_dir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'exec failed ({r.returncode}): {r.stderr.strip()}')
    m = _SUBMIT.search(r.stderr)
    ms = float(m.group(1)) if m else float('nan')
    out = np.fromfile(base + '.arena_out', '<f4')
    if out.size != out_n:
        raise RuntimeError(f'exec wrote {out.size} floats, expected {out_n}')
    return out, ms


def run_prefix(plan, n, arena_init, out_off, out_n, **kw):
    return run(plan.prefix(n), arena_init, out_off, out_n, **kw)
