"""Baseline for mm44.py: the production 2-vector x 16-column kernel (qpu_mm.kernel), same shape, best of 12. usage: python3 bench_base.py [K] [N] [M]"""
import sys, time, numpy as np
sys.path.insert(0, '.')
import qpu_mm as Q
import _videocore6.driver as D
_o = D.DRM_V3D
D.DRM_V3D = lambda *a, **k: _o('/dev/dri/renderD128')
from videocore6.driver import Driver
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1584; N = int(sys.argv[2]) if len(sys.argv) > 2 else 176; M = int(sys.argv[3]) if len(sys.argv) > 3 else 256
TP = 256; OPITCH = 256
rng = np.random.default_rng(8)
Wm = (rng.standard_normal((N, K)) * 0.03).astype(np.float32); bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
A0 = (rng.standard_normal((K, M)) * 0.5).astype(np.float32)
stream = Q.pack_weights(Wm, bias)
with Driver() as drv:
    code = drv.program(Q.kernel)
    A = drv.alloc(((K + 4) * TP + 16,), dtype=np.float32); Af = np.zeros(A.shape[0], np.float32)
    for k in range(K): Af[k * TP:k * TP + M] = A0[k]
    A[:] = Af
    O = drv.alloc((N * OPITCH + 64,), dtype=np.float32); O[:] = -7.0
    Ws = drv.alloc((stream.size + 16,), dtype=np.uint32); Ws[:stream.size] = stream
    u = [A.addresses()[0], TP * 4, TP * 4 + 64, 2 * TP * 4, 64, O.addresses()[0], OPITCH * 4, OPITCH * 4 - 64, 16 * OPITCH * 4,
         Ws.addresses()[0], (16 + 8 * K) * 4, K // 2, N // 16, M]
    TB = drv.alloc((8, 8), dtype=np.uint32); TB[:] = Q.make_table(M, N // 16); u[12] = TB.addresses()[0, 0]; u[13] = 0
    unif = drv.alloc(len(u) + 2, dtype=np.uint32); unif[:len(u)] = np.array(u, dtype=np.uint64).astype(np.uint32)
    best = 1e9
    for _ in range(int(__import__("os").environ.get("REPS", "40"))):
        t0 = time.monotonic(); drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=8); best = min(best, time.monotonic() - t0)
    got = np.array(O)[:N * OPITCH].reshape(N, OPITCH)[:, :M]
    want = Wm.astype(np.float16).astype(np.float32) @ A0 + bias[:, None]
    print(f'base K={K} N={N} M={M}: best {best*1e3:.2f} ms ({M*N*K/best/1e9:.2f} GMAC/s incl. launch), max err {np.abs(got - want).max():.3g}')
