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
RSC = _R['rf53']                                  # residual scale (rs kernel)
SE = _R['rf54']                                   # row-major kernel: 16 * row bytes
GG, GN = _R['rf55'], _R['rf56']
PASS, WSAVE, TCNT0, OBS = [_R['rf%d' % i] for i in range(57, 61)]
R0, TTV0, TTV1 = _R['rf61'], _R['rf62'], _R['rf63']      # skew kernel: row shift, per-lane key limit for vector 0 / 1   # passes over extra row blocks (M > 256) and the values to restore each pass                   # adaptive partition: this thread's first tile group / number of tile groups
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


def make_kernel(rs, relu=False, rm=False, skew=False):
    @qpu
    def kernel(asm):
        for reg in [ABASE, PI, PI64, PI2, C64, OB, OP, OS2, OTILE, WADDR, WT, KP, NT, MV] + ([RSC] if rs else []) + ([SE] if rm else []):
            nop(sig=ldunifrf(reg))
        # per-thread table (8 words per thread at address NT = param 12): m block base, first tile group g, tile groups G, tiles, valid m
        tidx(r0)
        shr(r0, r0, 2)
        band(r0, r0, 0b1111)
        shl(r0, r0, 5)
        add(r0, r0, NT)
        mov(unifa, r0)
        nop()
        nop()
        nop()
        nop(sig=ldunifarf(LM))
        nop(sig=ldunifarf(GG))
        nop(sig=ldunifarf(GN))
        nop(sig=ldunifarf(TCNT))
        nop(sig=ldunifarf(MV))
        nop(sig=ldunifarf(PASS))                          # table word 5: passes over blocks of 256 rows
        if skew and os.environ.get('SKEWDBG') != 'noload':
            nop(sig=ldunifarf(NT))                        # table word 6: T keys (NT register is free now)
            nop(sig=ldunifarf(R0))                        # table word 7: row shift R0 = T - 1 + e
        eidx(r1)
        add(LM, LM, r1)                                   # m of lane (vector 0)
        shl(LANE, LM, 2)
        if rm:
            umul24(r0, LM, OP)                            # row-major output: lane m starts at m * row bytes
            add(OB, OB, r0)
        else:
            add(OB, OB, LANE)
        add(ABASE, ABASE, LANE)
        umul24(r0, GG, WT)
        add(WADDR, WADDR, r0)                             # this thread starts at tile g ...
        umul24(r0, GG, OTILE)
        add(OB, OB, r0)
        umul24(WT, WT, GN)                                # ... and steps by G tiles
        umul24(OTILE, OTILE, GN)
        mov(WSAVE, WADDR)
        mov(TCNT0, TCNT)
        mov(OBS, OB)
        mov(Z, 0)
        if skew and os.environ.get('SKEWDBG') not in ('nottv', 'noprol'):
            mov(r0, 1)
            shl(r0, r0, -2)                                  # shift by 30 (shifts are modulo 32; small immediates stop at 15)
            sub(r2, Z, r0)                                # -2^30: lanes that must never store
            shr(r1, C64, 2)                               # 16
            sub(null, LM, MV, cond='pushn')               # vector 0 lane valid (m < M)
            mov(TTV0, r2)
            mov(TTV0, NT, cond='ifa')                     # valid lanes: key limit T
            add(r0, LM, r1)
            sub(null, r0, MV, cond='pushn')
            mov(TTV1, r2)
            mov(TTV1, NT, cond='ifa')
        nop(sig=thrsw)
        nop()
        nop()

        with loop as lp:                                  # over blocks of 256 rows (normally one pass)
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
                if skew and os.environ.get('SKEWDBG') != 'nostore':
                    # position term written skewed: entry (row r, lane i) goes to key j = r - R0 + i (valid 0 <= j < T), contiguous per key row
                    sub(r0, TCNT0, TCNT)                          # tiles done by this thread
                    umul24(r0, r0, GN)
                    add(r0, r0, GG)                               # first-n tile group index
                    shl(r0, r0, 4)
                    sub(r2, LM, R0)
                    add(r2, r2, r0)                               # J of lane (vector 0) for the tile's first row
                    shr(r3, C64, 2)                               # 16
                    for n in range(16):
                        add(r0, r2, r3)                           # J of vector 1
                        bxor(r1, r0, -1)                          # ~J < 0  <=>  J >= 0
                        sub(r0, r0, TTV1)                         # J - T >= 0  <=>  J >= T (or the lane is beyond M)
                        imax(r1, r1, r0, cond='pushn')             # A = vector 1 lane valid
                        bxor(r1, r2, -1)
                        sub(r0, r2, TTV0)
                        imax(r1, r1, r0, cond='pushn')             # A = vector 0 valid, B = vector 1 valid
                        mov(tmud, ACC[2 * n])
                        mov(tmua, OCUR, cond='ifa')
                        add(OCUR, OCUR, SE)
                        tmuwt()
                        mov(tmud, ACC[2 * n + 1])
                        mov(tmua, OCUR, cond='ifb')
                        add(OCUR, OCUR, OS2)
                        tmuwt()
                        add(r2, r2, 1)
                elif rs:
                    for j in range(NA):                       # out += scale * (acc incl. bias): read, multiply-add, store
                        mov(tmua, OCUR)
                        fmul(r3, ACC[j], RSC)
                        nop(sig=ldtmu(r0))
                        fadd(r1, r0, r3)
                        mov(tmud, r1)
                        mov(tmua, OCUR, cond='ifa' if j % 2 == 0 else 'ifb')
                        add(OCUR, OCUR, C64 if j % 2 == 0 else OS2)
                        tmuwt()
                else:
                    for j in range(NA):
                        if relu:
                            fmax(tmud, ACC[j], 0.0)
                        else:
                            mov(tmud, ACC[j])
                        mov(tmua, OCUR, cond='ifa' if j % 2 == 0 else 'ifb')
                        add(OCUR, OCUR, (SE if rm else C64) if j % 2 == 0 else OS2)
                        tmuwt()
                add(OB, OB, OTILE)
                sub(TCNT, TCNT, 1, cond='pushz')
                lt.b(cond='anyna')
                nop()
                nop()
                nop()
            # next pass: 256 rows further
            sub(PASS, PASS, 1, cond='pushz')
            mov(WADDR, WSAVE)
            mov(TCNT, TCNT0)
            shl(r0, C64, 4)                                   # 1024 bytes = 256 floats
            shl(r1, C64, 2)                                   # 256 rows
            add(ABASE, ABASE, r0)
            add(LM, LM, r1)
            if rm:
                umul24(r2, r1, OP)
                add(OBS, OBS, r2)
            else:
                add(OBS, OBS, r0)
            mov(OB, OBS)
            lp.b(cond='anyna')
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


    return kernel


kernel = make_kernel(False)
kernel_rs = make_kernel(True)
kernel_relu = make_kernel(False, True)
kernel_rm = make_kernel(False, False, True)
kernel_skew = make_kernel(False, False, True, skew=True)


def use_render_node():
    import _videocore6.driver as D
    if not getattr(D, '_patched', False):
        _o = D.DRM_V3D
        D.DRM_V3D = lambda *a, **k: _o('/dev/dri/renderD128')     # works even while a Vulkan process holds card0
        D._patched = True


def make_table(M, NT, nthreads=8, T=0, R0=0):
    """Per-thread work table (8 words each): [m block base, first tile group g, tile groups G, tiles, valid m, passes, 0, 0].
    nblk = ceil(M / 32) row blocks of 32. nblk <= 8: G = 8 // nblk tile groups share the n tiles, leftover threads idle (one masked tile).
    nblk > 8: every thread owns block t and repeats for passes of 256 rows (blocks t + 8 p)."""
    nblk = max(1, (M + 31) // 32)
    tab = np.zeros((nthreads, 8), np.uint32)
    if nblk > nthreads:
        P = (nblk + nthreads - 1) // nthreads
        for t in range(nthreads):
            tab[t] = [t * 32, 0, 1, NT, M, P, 0, 0]
        tab[:, 6] = T; tab[:, 7] = R0
        return tab
    G = nthreads // nblk
    for t in range(nthreads):
        mb, g = t % nblk, t // nblk
        ntl = (NT - g + G - 1) // G if t < G * nblk else 0
        if ntl <= 0 or t >= G * nblk:
            tab[t] = [0, 0, 1, 1, 0, 1, 0, 0]
        else:
            tab[t] = [mb * 32, g, G, ntl, M, 1, 0, 0]
    tab[:, 6] = T; tab[:, 7] = R0
    return tab


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
    for k_, p_ in ((kernel, path), (kernel_rs, path.replace('.qpu', 'rs.qpu')), (kernel_relu, path.replace('.qpu', 'relu.qpu')), (kernel_rm, path.replace('.qpu', 'rm.qpu')), (kernel_skew, path.replace('.qpu', 'sk.qpu'))):
        code = np.array(assemble(k_), dtype=np.uint64)
        code.tofile(p_)
        print(f'{len(code)} instructions -> {p_}')


def test():
    import time
    use_render_node()
    from videocore6.driver import Driver
    K, N = 176, 704
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 251
    TP = max(252, (M + 3) // 4 * 4)
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
        TB = drv.alloc((8, 8), dtype=np.uint32); TB[:] = make_table(M, N // 16); u[12] = TB.addresses()[0, 0]; u[13] = 0
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


def test_rs():
    import time
    use_render_node()
    from videocore6.driver import Driver
    TP, M = 252, 251
    rng = np.random.default_rng(6)
    with Driver() as drv:
        code = drv.program(kernel_rs)
        for K, N, scale in [(704, 176, 0.5), (176, 176, 1.0)]:
            Wm = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
            bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
            A0 = (rng.standard_normal((K, M)) * 0.5).astype(np.float32)
            X0 = rng.standard_normal((N, TP)).astype(np.float32)
            stream = pack_weights(Wm, bias)
            A = drv.alloc(((K + 4) * TP + 16,), dtype=np.float32); A[:] = np.nan
            Af = np.array(A).reshape(-1)
            for k in range(K):
                Af[k * TP:k * TP + M] = A0[k]
            A[:] = Af
            O = drv.alloc((N * TP + 64,), dtype=np.float32)
            Ws = drv.alloc((stream.size + 16,), dtype=np.uint32); Ws[:stream.size] = stream
            u = [A.addresses()[0], TP * 4, TP * 4 + 64, 2 * TP * 4, 64, O.addresses()[0], TP * 4, TP * 4 - 64, 16 * TP * 4,
                 Ws.addresses()[0], (16 + 8 * K) * 4, K // 2, N // 16, M, int(np.float32(scale).view(np.uint32))]
            TB = drv.alloc((8, 8), dtype=np.uint32); TB[:] = make_table(M, N // 16); u[12] = TB.addresses()[0, 0]; u[13] = 0
            unif = drv.alloc(len(u) + 2, dtype=np.uint32); unif[:len(u)] = np.array(u, dtype=np.uint64).astype(np.uint32)
            for rep in range(3):
                O[:N * TP] = X0.reshape(-1)
                t0 = time.monotonic()
                drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=8)
                dt = time.monotonic() - t0
            got = np.array(O)[:N * TP].reshape(N, TP)
            want = X0[:, :M] + scale * ((Wm.astype(np.float16).astype(np.float32)) @ A0 + bias[:, None])
            err = np.abs(got[:, :M] - want).max()
            print(f'RS K={K} N={N} scale={scale}: {dt * 1e3:.2f} ms, max err {err:.2e}, padding columns untouched: {bool((got[:, M:] == X0[:, M:]).all())}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    here = os.path.dirname(os.path.abspath(__file__))
    if mode == 'testrs':
        test_rs()
    elif mode == 'build':
        build(os.path.join(here, 'mm16.qpu'))
    else:
        test()
