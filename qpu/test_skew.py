import sys, numpy as np, time
sys.path.insert(0, '.')
import qpu_mm as Q
Q.use_render_node()
from videocore6.driver import Driver
T = int(sys.argv[1]) if len(sys.argv) > 1 else 251
e = int(sys.argv[2]) if len(sys.argv) > 2 else 2
K = 44; TP = (T + 11) // 12 * 12; GR = 2 * T - 1 + e; R0 = T - 1 + e
ntiles = (GR + 15) // 16
rng = np.random.default_rng(11)
Wm = (rng.standard_normal((ntiles * 16, K)) * 0.1).astype(np.float32); Wm[GR:] = 0
A0 = (rng.standard_normal((K, T)) * 0.5).astype(np.float32)
stream = Q.pack_weights(Wm, np.zeros(ntiles * 16, np.float32))
LS = (TP + 1) * 4; NS = TP * 4
with Driver() as drv:
    code = drv.program(Q.kernel_skew)
    A = drv.alloc(((K + 4) * TP + 16,), dtype=np.float32); Af = np.full(A.shape[0], np.nan, np.float32)
    for k in range(K): Af[k * TP:k * TP + T] = A0[k]
    A[:] = Af
    OUT = drv.alloc((T * TP + 64 * TP,), dtype=np.float32); OUT[:] = -9.0
    Ws = drv.alloc((stream.size + 16,), dtype=np.uint32); Ws[:stream.size] = stream
    out_base = OUT.addresses()[0] + 32 * TP * 4          # leave 32 rows of margin on both sides: nothing may be written there
    ob = (int(out_base) - R0 * TP * 4) & 0xffffffff
    u = [A.addresses()[0], TP * 4, TP * 4 + 64, 2 * TP * 4, 64, ob, LS, (NS - 16 * LS) & 0xffffffff, 16 * NS,
         Ws.addresses()[0], (16 + 8 * K) * 4, K // 2, 0, 0, 16 * LS]
    TB = drv.alloc((8, 8), dtype=np.uint32); TB[:] = Q.make_table(T, ntiles, T=T, R0=R0); u[12] = TB.addresses()[0, 0]
    unif = drv.alloc(len(u) + 2, dtype=np.uint32); unif[:len(u)] = np.array(u, dtype=np.uint64).astype(np.uint32)
    for rep in range(3):
        t0 = time.monotonic(); drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=8); dt = time.monotonic() - t0
    full = np.array(OUT)
    got = full[32 * TP:32 * TP + T * TP].reshape(T, TP)
    Wq = Wm.astype(np.float16).astype(np.float32)
    P = Wq @ A0                                                   # [row r][i]
    want = np.zeros((T, T), np.float32)
    for j in range(T):
        for i in range(T):
            want[j, i] = P[j + R0 - i, i]
    err = np.abs(got[:, :T] - want).max()
    margin_ok = bool((full[:32 * TP] == -9.0).all() and (full[32 * TP + T * TP:] == -9.0).all())
    print(f'skew T={T} e={e} GR={GR} tiles={ntiles}: {dt*1e3:.2f} ms, max err {err:.2e}, padding cols untouched {bool((got[:, T:] == -9.0).all())}, margins untouched {margin_ok}')
