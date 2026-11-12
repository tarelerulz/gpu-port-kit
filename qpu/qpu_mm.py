"""Production-shaped assembly matmul for the nemo8 hybrid runner (2026-11-12). Builds mm16.qpu (machine code) and self-tests it.

out[n][m] = bias[n] + sum_k W[k][n] * A[k][m]      A k-major [K][pitch], out [N][pitch] (m contiguous), m < M only (lanes >= M are not stored)
8 threads (1 per QPU); thread t owns m = 32t .. 32t+31 (so M <= 256). Every thread loops over the n tiles (16 outputs each).
Weight stream per tile (uniform memory read with unifa + ldunifa): 16 bias words (f32), then K * 8 words, each word = two f16 weights
(low half = even n, high half = odd n). Tile stride = (16 + 8 K) words.
Uniform parameters (14 words): A_base, PI, PI+64, 2*PI, 64, O_base, OP, OP-64, 16*OP, W_addr, tile_bytes, K/2, ntiles, M
(addresses are GPU virtual addresses, PI / OP row pitches in bytes).

usage: PYTHONPATH=<py-videocore6>/src python3 qpu_mm.py [build|test]
"""
import os
import sys
import numpy as np
from _videocore6.assembler import Instruction, qpu, assemble
_R = Instruction.REGISTERS

NA = 32
ACC = [_R["rf%d" % i] for i in range(NA)]
(P0, PI, PI64, PI2, C64, Z) = [_R["rf%d" % i] for i in range(32, 38)]
(OB, OCUR, OP, OS2, OTILE) = [_R["rf%d" % i] for i in range(38, 43)]
(WADDR, WT, KP, NT, KCNT, TCNT, LANE, ABASE, MV, LM) = [_R["rf%d" % i] for i in range(43, 53)]
XSET = [(_R["r0"], _R["r1"]), (_R["r2"], _R["r4"])]
NPARAM = 14


def emit_block(s, odd, lk=None):
    X = XSET[s]
    Xn = XSET[1 - s]
    for t in range(32):
        n, mv = divmod(t, 2)
        prev = ACC[(2 * n + mv - 1) % NA]
        kw = {}
        if t % 4 == 3:
            kw['sig'] = ldunifa                       # next f16 pair of weights (two n values)
        elif t == 28:
            kw['sig'] = ldtmu(Xn[0])
        elif t == 30:
            kw['sig'] = ldtmu(Xn[1])
        if odd and t == 29:
            lk.b(cond='anyna')                        # t = 29, 30, 31 are its delay slots
        wreg = r5.unpack('l') if t % 4 < 2 else r5.unpack('h')
        fadd(prev, prev, r3, **kw).fmul(r3, wreg, X[mv])
        if t == 5:
            add(tmua, P0, PI if odd else Z)
        if t == 9:
            add(tmua, P0, PI64 if odd else C64)
        if odd and t == 13:
            add(P0, P0, PI2)
        if odd and t == 17:
            sub(KCNT, KCNT, 1, cond='pushz')


@qpu
def kernel(asm):
    for reg in [ABASE, PI, PI64, PI2, C64, OB, OP, OS2, OTILE, WADDR, WT, KP, NT, MV]:
        nop(sig=ldunifrf(reg))
    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111)
    shl(r0, r0, 5)
    eidx(r1)
    add(LM, r0, r1)                                   # m of lane (vector 0)
    shl(LANE, LM, 2)
    add(OB, OB, LANE)
    add(ABASE, ABASE, LANE)
    mov(Z, 0)
    mov(TCNT, NT)
    nop(sig=thrsw)
    nop()
    nop()

    with loop as lt:                                  # over n tiles
        mov(unifa, WADDR)
        add(WADDR, WADDR, WT)
        mov(KCNT, KP)
        add(tmua, ABASE, Z)                           # x requests for k = 0 and k = 1
        add(tmua, ABASE, C64)
        add(P0, ABASE, PI)
        add(tmua, P0, Z)
        add(tmua, P0, C64)
        add(P0, P0, PI)                               # P0 = k 2
        mov(r3, 0.0)
        nop(sig=ldtmu(r0))
        nop(sig=ldtmu(r1))
        nop()
        nop(sig=ldunifa)                              # bias 0 -> r5
        for n in range(16):                           # accumulators start at the bias; the last signal fetches weight pair 0
            mov(ACC[2 * n], r5, sig=ldunifa).mov(ACC[2 * n + 1], r5)

        with loop as lk:
            emit_block(0, False)
            emit_block(1, True, lk)

        fadd(ACC[NA - 1], ACC[NA - 1], r3)
        nop(sig=ldtmu(r0))
        nop(sig=ldtmu(r1))

        mov(r1, 1)
        shl(r1, r1, 4)                                # 16 (small immediates stop at 15)
        add(r0, LM, r1)                               # flags: A = vector 0 lanes with m < M, B = vector 1 lanes
        sub(null, r0, MV, cond='pushn')
        sub(null, LM, MV, cond='pushn')
        mov(OCUR, OB)
        for j in range(NA):
            mov(tmud, ACC[j])
            mov(tmua, OCUR, cond='ifa' if j % 2 == 0 else 'ifb')
            add(OCUR, OCUR, C64 if j % 2 == 0 else OS2)
            tmuwt()
        add(OB, OB, OTILE)
        sub(TCNT, TCNT, 1, cond='pushz')
        lt.b(cond='anyna')
        nop()
        nop()
        nop()

    barrierid(syncb, sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def pack_weights(W, bias):
    """W [N][K] float (torch layout), bias [N]. N % 16 == 0. Returns a uint32 stream of tiles: 16 bias words + K * 8 f16-pair words."""
    N, K = W.shape
    assert N % 16 == 0
    h = np.asarray(W, np.float32).astype(np.float16).view(np.uint16).reshape(N // 16, 8, 2, K)     # [tile][pair][lo/hi][k]
    words = (h[:, :, 0, :].astype(np.uint32) | (h[:, :, 1, :].astype(np.uint32) << 16))               # [tile][pair][k]
    words = np.transpose(words, (0, 2, 1))                                                             # [tile][k][pair]
    b = np.asarray(bias, np.float32).reshape(N // 16, 16).view(np.uint32)
    return np.concatenate([b, words.reshape(N // 16, K * 8)], axis=1).reshape(-1).astype(np.uint32)


def build(path):
    code = np.array(assemble(kernel), dtype=np.uint64)
    code.tofile(path)
    print(f'{len(code)} instructions -> {path}')
    return code


def test():
    import time
    from videocore6.driver import Driver
    K, N, M, TP = 176, 704, 251, 252
    rng = np.random.default_rng(5)
    Wm = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
    A0 = (rng.standard_normal((K, M)) * 0.5).astype(np.float32)
    stream = pack_weights(Wm, bias)
    tile_words = 16 + 8 * K
    with Driver() as drv:
        code = drv.program(kernel)
        A = drv.alloc((K + 4, TP + 8), dtype=np.float32); A[:] = 7.0         # junk beyond M, pitch TP + 8 would break the kernel: use pitch TP below
        A = drv.alloc(((K + 4) * TP + 16,), dtype=np.float32); A[:] = np.nan  # NaN everywhere first: lanes >= M must not leak into the result
        Af = np.array(A).reshape(-1)
        for k in range(K):
            Af[k * TP:k * TP + M] = A0[k]
        A[:] = Af
        O = drv.alloc((N * TP + 64,), dtype=np.float32); O[:] = -5.0
        Ws = drv.alloc((stream.size + 16,), dtype=np.uint32); Ws[:stream.size] = stream
        u = [A.addresses()[0], TP * 4, TP * 4 + 64, 2 * TP * 4, 64, O.addresses()[0], TP * 4, TP * 4 - 64, 16 * TP * 4,
             Ws.addresses()[0], tile_words * 4, K // 2, N // 16, M]
        unif = drv.alloc(len(u) + 2, dtype=np.uint32); unif[:len(u)] = np.array(u, dtype=np.uint64).astype(np.uint32)
        for rep in range(3):
            t0 = time.monotonic()
            drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=8)
            dt = time.monotonic() - t0
        got = np.array(O)[:N * TP].reshape(N, TP)
        Wq = Wm.astype(np.float16).astype(np.float32)
        want = Wq @ A0 + bias[:, None]
        err = np.abs(got[:, :M] - want).max()
        pad_untouched = bool((got[:, M:] == -5.0).all())
        print(f'FF1 shape K={K} N={N} M={M}: {dt * 1e3:.2f} ms, max err vs f16-weight reference {err:.2e}, padding columns untouched: {pad_untouched}, finite: {bool(np.isfinite(got[:, :M]).all())}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    here = os.path.dirname(os.path.abspath(__file__))
    if mode == 'build':
        build(os.path.join(here, 'mm16.qpu'))
    else:
        test()
