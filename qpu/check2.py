"""Check 2: throughput of back-to-back CSD jobs through the DRM interface (no waiting between submits). 2026-11-12.
usage: PYTHONPATH=<py-videocore6>/src python3 check2.py"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_asm as M
from videocore6.driver import Driver

def run(K, NT, njobs, drv, code, unif):
    best = 1e9
    for rep in range(4):
        t0 = time.monotonic()
        with drv.compute_shader_dispatcher(timeout_sec=60) as csd:
            for _ in range(njobs):
                csd.dispatch(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=8)
            t_sub = time.monotonic() - t0
        best = min(best, time.monotonic() - t0)
    return best, t_sub

with Driver() as drv:
    code = drv.program(M.kernel)
    rng = np.random.default_rng(1)
    for K, NT in [(2, 1), (704, 1), (704, 11)]:
        MP = 256
        A = drv.alloc((K + 4, MP), dtype=np.float32); A[:] = 0.0
        Wt = drv.alloc((NT + 1, K, 16), dtype=np.float32); Wt[:] = 0.0
        O = drv.alloc((16 * NT, MP), dtype=np.float32)
        u = [A.addresses()[0, 0], MP * 4, MP * 4 + 64, 2 * MP * 4, 64, O.addresses()[0, 0], MP * 4, MP * 4 - 64, 16 * MP * 4,
             Wt.addresses()[0, 0, 0], K * 16 * 4, K // 2, NT]
        unif = drv.alloc(len(u) + 2, dtype=np.uint32); unif[:len(u)] = np.array(u, dtype=np.uint64).astype(np.uint32)
        one, _ = run(K, NT, 1, drv, code, unif)
        for n in (50, 342):
            tot, tsub = run(K, NT, n, drv, code, unif)
            print(f'K={K:4d} NT={NT:2d}: {n:3d} jobs: total {tot * 1e3:7.2f} ms = {tot / n * 1e3:.3f} ms/job (python submit loop {tsub * 1e3:.1f} ms); single job incl. wait {one * 1e3:.2f} ms', flush=True)
