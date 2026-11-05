"""Runner health check: run before any port. Fails loudly if the GPU runner computes matrix multiplies wrongly.
Covers the bug found 2026-11-05 (specialization constant 0 read as the row count): rows past 256 were never
computed. Checks the row counts and output widths that a port uses. Exit 1 on failure.
usage: python3 ~/gpu-port-kit/tests/check_runner.py"""
import os
import sys
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport import Arena, Weights, Plan, ops
from gpuport.run import run
from gpuport.config import WORK_DIR, merged_shader_dir, STREAM_SHADERS, EXEC_BIN

SHADERS = merged_shader_dir(extra=(os.path.join(KIT, 'shaders'), STREAM_SHADERS))
rng = np.random.default_rng(21)
bad = []
print(f'runner: {EXEC_BIN}')
for M, K, N in [(7, 16, 12), (251, 176, 704), (260, 176, 33), (502, 512, 257), (1001, 64, 40)]:
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    W = (rng.standard_normal((N, K)) * 0.1).astype(np.float32); b = (rng.standard_normal(N) * 0.1).astype(np.float32)
    want = A @ W.T + b
    w = Weights(); mm = w.put_mm(W, b); path = w.save(os.path.join(WORK_DIR, 'runner_check'))
    ar = Arena(); ia = ar.alloc('A', M * K); io = ar.alloc('O', M * N)
    arena = np.zeros(ar.used(), np.float32); arena[ia:ia + M * K] = A.ravel()
    # one unsplit call: ops.gemm12 would split above 251 rows and hide the bug, so dispatch directly
    from gpuport.plan import split_groups
    RG = (M + 3) // 4; groups = max(24, ((mm['np'] // 12) * RG + 255) // 256); gx, gy = split_groups(groups)
    p = Plan(); p.add('gemm12', 256, gx, gy, [M, N, K, mm['np'], ia, K, 1, mm['wo'], mm['bo'], io, N, 1, 0])
    got = run(p, arena, io, M * N, weights=path, shader_dir=SHADERS, lint=False)[0]
    err = np.abs(got - want.ravel()).max()
    ok = err < 1e-3
    print(f'{"ok  " if ok else "FAIL"} gemm12 one call M={M} K={K} N={N}: max err {err:.2e}')
    if not ok:
        bad.append((M, K, N))
if bad:
    print(f'RUNNER BROKEN for {bad}: do not trust GPU results. See GPU_MAP.md section 16.')
    raise SystemExit(1)
print('runner OK')
