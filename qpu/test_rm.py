import sys, numpy as np, time
sys.path.insert(0, '.')
import qpu_mm as Q
import _videocore6.driver as D
_o = D.DRM_V3D
D.DRM_V3D = lambda *a, **k: _o('/dev/dri/renderD128')      # render node: works while another process holds card0
from videocore6.driver import Driver
K, N, M, TP = 176, 528, 251, 252
M = int(sys.argv[1]) if len(sys.argv) > 1 else M
RB = N * 4
rng = np.random.default_rng(9)
Wm = (rng.standard_normal((N, K)) * 0.05).astype(np.float32); bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
A0 = (rng.standard_normal((K, M)) * 0.5).astype(np.float32)
stream = Q.pack_weights(Wm, bias)
with Driver() as drv:
    code = drv.program(Q.kernel_rm)
    A = drv.alloc(((K + 4) * TP + 16,), dtype=np.float32); Af = np.full(A.shape[0], np.nan, np.float32)
    for k in range(K): Af[k * TP:k * TP + M] = A0[k]
    A[:] = Af
    O = drv.alloc((251 * N + 4096,), dtype=np.float32); O[:] = -7.0
    Ws = drv.alloc((stream.size + 16,), dtype=np.uint32); Ws[:stream.size] = stream
    u = [A.addresses()[0], TP * 4, TP * 4 + 64, 2 * TP * 4, 64, O.addresses()[0], RB, (4 - 16 * RB) & 0xffffffff, 64,
         Ws.addresses()[0], (16 + 8 * K) * 4, K // 2, N // 16, M, 16 * RB]
    TB = drv.alloc((8, 8), dtype=np.uint32); TB[:] = Q.make_table(M, N // 16); u[12] = TB.addresses()[0, 0]; u[13] = 0
    unif = drv.alloc(len(u) + 2, dtype=np.uint32); unif[:len(u)] = np.array(u, dtype=np.uint64).astype(np.uint32)
    for rep in range(3):
        t0 = time.monotonic(); drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=8); dt = time.monotonic() - t0
    got = np.array(O)[:M * N].reshape(M, N)
    want = (Wm.astype(np.float16).astype(np.float32) @ A0 + bias[:, None]).T
    print(f'row-major out K={K} N={N}: {dt*1e3:.2f} ms, max err {np.abs(got - want).max():.3g}, untouched tail {bool((np.array(O)[M*N:M*N+4096] == -7.0).all())}')
