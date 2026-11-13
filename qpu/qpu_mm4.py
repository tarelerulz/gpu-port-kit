"""QPU matmul, 4-vector x 8-column tiling (2026-11-12). Replaces the 2 x 16 tiling of qpu_mm.py for matmuls with static weights:
uniform (weight stream) traffic per MAC is halved, which was the bottleneck (10 -> ~11.5-13 GMAC/s).

out[n][m] = bias[n] + sum_k W[k][n] A[k][m] for m < M (<= 512). A k-major [K][pitch], pitch >= 64 * ceil(M / 64) (lanes beyond M are read, never stored).
8 threads (one per QPU). nblk = ceil(M / 64) blocks of 64 rows (4 vectors of 16 lanes), G = 8 // nblk tile groups; thread t < G * nblk owns block t % nblk and
tiles g, g + G, ... (g = t // nblk) of 8 columns. Weight stream per 8-column tile: 8 bias words (f32), then K * 4 words (f16 pairs: low half = even n, high half = odd n).
Variants: plain, relu, silu (SFU exp + recip), rs (out += scale * (acc + bias), RSC), rm (row-major output [m][n], row pitch OP bytes).
Uniform words (ldunif): PI (A pitch bytes), OP (out pitch bytes), WT (tile step bytes = tile bytes * G), OTILE (out step per tile, bytes), KP (K / 2), NT (thread table address), RSC.
Thread table (8 words at NT + 32 t): A base (address of A[0][m0]), weight stream start, out base, tiles, valid rows V (rows of this block < M; <= 0: idle, all stores masked).
usage: PYTHONPATH=shim:<py-videocore6>/src python3 qpu_mm4.py [build | test [variant] [M] [K] [N]]"""
import os
import sys
import time
import numpy as np
from _videocore6.assembler import Instruction, qpu, assemble
_R = Instruction.REGISTERS
rfr = lambda i: _R['rf%d' % i]
NA = 32
ACC = [rfr(i) for i in range(NA)]
XA = [_R['r0'], _R['r1'], _R['r2'], _R['r4']]
XB = [rfr(i) for i in range(32, 36)]
XS = [XA, XB]
(P0, PI, PI64, PI128, PI192, PI2, Z, C64, C128, C192, KCNT, KP) = [rfr(i) for i in range(36, 48)]
(WADDR, WT, ABASE, TCNT, OB, OTILE, OCUR, STEPN, V, RSC, STEPV, NT, OP) = [rfr(i) for i in range(48, 61)]
VARIANTS = ('plain', 'relu', 'silu', 'rs', 'rm')
FILES = {'plain': 'mm4.qpu', 'relu': 'mm4relu.qpu', 'silu': 'mm4silu.qpu', 'rs': 'mm4rs.qpu', 'rm': 'mm4rm.qpu'}


EXP = set(os.environ.get('Q4EXP', '').split(','))
NREQ = 0 if 'nox' in EXP else (2 if 'halfx' in EXP else 4)      # timing experiments: noepi noloop nowt batchwt nopro


def emit_block(s, odd, lk=None):
    X, Xn = XS[s], XS[1 - s]
    req = [PI, PI64, PI128, PI192] if odd else [Z, C64, C128, C192]
    for t in range(32):
        kw = {}
        if t % 8 == 7 and 'nowt' not in EXP:
            kw['sig'] = ldunifa                          # next f16 pair word (two n values)
        elif t in (24, 25, 26, 28) and {24: 0, 25: 1, 26: 2, 28: 3}[t] < NREQ:
            kw['sig'] = ldtmu(Xn[{24: 0, 25: 1, 26: 2, 28: 3}[t]])
        if t in (4, 8, 12, 16) and (t - 4) // 4 < NREQ:
            add(tmua, P0, req[(t - 4) // 4])             # x requests for k + 2, 4 vectors
        if odd and t == 18:
            add(P0, P0, PI2)
        if odd and t == 20:
            sub(KCNT, KCNT, 1, cond='pushz')
        if odd and t == 29:
            lk.b(cond='anyna')                           # t = 29, 30, 31 are its delay slots
        prev = ACC[(t - 1) % NA]
        wreg = _R['r5'].unpack('l') if (t // 4) % 2 == 0 else _R['r5'].unpack('h')
        fadd(prev, prev, r3, **kw).fmul(r3, wreg, X[t % 4])


def make_kernel(variant):
    assert variant in VARIANTS
    rm = variant == 'rm'

    @qpu
    def kernel(asm):
        for reg in [PI, OP, WT, OTILE, KP, NT] + ([RSC] if variant in ('rs', 'silu') else []):
            nop(sig=ldunifrf(reg))
        tidx(r0)
        shr(r0, r0, 2)
        band(r0, r0, 0b1111)
        shl(r0, r0, 5)
        add(r0, r0, NT)
        mov(unifa, r0)
        nop()
        nop()
        nop()
        nop(sig=ldunifarf(ABASE))
        nop(sig=ldunifarf(WADDR))
        nop(sig=ldunifarf(OB))
        nop(sig=ldunifarf(TCNT))
        nop(sig=ldunifarf(V))
        eidx(r1)
        shl(r2, r1, 2)
        add(ABASE, ABASE, r2)
        if rm:
            umul24(r0, r1, OP)                           # row-major output: lane m starts at m * row bytes
            add(OB, OB, r0)
        else:
            add(OB, OB, r2)
        mov(r0, 1)
        shl(C64, r0, 6)
        shl(C128, r0, 7)
        add(C192, C64, C128)
        mov(Z, 0)
        add(PI64, PI, C64)
        add(PI128, PI, C128)
        add(PI192, PI, C192)
        add(PI2, PI, PI)
        if rm:
            shl(STEPV, OP, 4)                            # 16 rows
            shl(r0, r0, 2)                               # 4 bytes
            sub(STEPN, r0, STEPV)
        else:
            sub(STEPN, OP, C64)
        nop(sig=thrsw)
        nop()
        nop()
        with loop as lt:                                     # over n tiles
            mov(unifa, WADDR)
            add(WADDR, WADDR, WT)
            mov(KCNT, KP)
            add(P0, ABASE, Z)
            for c in (Z, C64, C128, C192):                   # k = 0
                add(tmua, P0, c)
            add(P0, ABASE, PI)
            for c in (Z, C64, C128, C192):                   # k = 1
                add(tmua, P0, c)
            add(P0, P0, PI)                                  # P0 = k 2
            mov(r3, 0.0)
            for i in range(4):
                nop(sig=ldtmu(XA[i]))
            nop()
            nop(sig=ldunifa)                                 # bias 0 -> r5
            for n in range(8):                               # accumulators start at the bias; last signal fetches weight pair 0
                mov(ACC[4 * n], r5).mov(ACC[4 * n + 1], r5)
                mov(ACC[4 * n + 2], r5, sig=ldunifa).mov(ACC[4 * n + 3], r5)
            if 'noloop' not in EXP:
              with loop as lk:
                emit_block(0, False)
                emit_block(1, True, lk)
            fadd(ACC[NA - 1], ACC[NA - 1], r3)
            for i in range(NREQ):
                nop(sig=ldtmu(r0))                           # drain the requests beyond the end
            for ph in (range(2) if 'noepi' not in EXP else []):                              # vectors (0, 1) then (2, 3): flags A, B = lanes of the first / second vector valid
                eidx(r1)
                # lane offsets of the two vectors: (16 + 32 ph) and 32 ph
                shr(r2, (C64 if ph == 0 else C192), 2)       # 16 (ph 0) or 48 (ph 1): vector 2ph + 1
                add(r0, r1, r2)
                sub(null, r0, V, cond='pushn')
                if ph == 0:
                    mov(r0, r1)
                else:
                    shr(r2, C128, 2)                         # 32: vector 2
                    add(r0, r1, r2)
                sub(null, r0, V, cond='pushn')
                if ph == 0:
                    mov(OCUR, OB)
                else:
                    add(r2, STEPV if rm else C64, STEPV if rm else C64)
                    add(OCUR, OB, r2)
                for n in range(8):
                    for vv in range(2):
                        j = 4 * n + 2 * ph + vv
                        cnd = 'ifa' if vv == 0 else 'ifb'
                        step = (STEPV if rm else C64) if vv == 0 else STEPN
                        if variant == 'rs':
                            mov(tmua, OCUR)
                            fmul(r3, ACC[j], RSC)
                            nop(sig=ldtmu(r0))
                            fadd(r1, r0, r3)
                            mov(tmud, r1)
                        elif variant == 'silu':
                            fmul(r0, ACC[j], RSC)            # RSC = -log2(e)
                            mov(exp, r0)
                            nop()
                            nop()
                            fadd(r0, r4, 1.0)
                            mov(recip, r0)
                            nop()
                            nop()
                            fmul(tmud, ACC[j], r4)
                        elif variant == 'relu':
                            fmax(tmud, ACC[j], 0.0)
                        else:
                            mov(tmud, ACC[j])
                        mov(tmua, OCUR, cond=cnd)
                        add(OCUR, OCUR, step)
                        if 'batchwt' not in EXP:
                            tmuwt()
            if 'batchwt' in EXP:
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
    return kernel


def pack8(W, bias):
    """W [N][K], bias [N], N % 8 == 0 -> uint32 stream: per 8-column tile 8 bias words then K * 4 words (low half even n, high half odd n)."""
    N, K = W.shape
    h = np.asarray(W, np.float32).astype(np.float16).view(np.uint16).reshape(N // 8, 4, 2, K)
    words = h[:, :, 0, :].astype(np.uint32) | (h[:, :, 1, :].astype(np.uint32) << 16)       # [tile][pair][k]
    words = np.transpose(words, (0, 2, 1))                                                    # [tile][k][pair]
    b = np.asarray(bias, np.float32).reshape(N // 8, 8).view(np.uint32)
    return np.concatenate([b, words.reshape(N // 8, K * 4)], axis=1).reshape(-1).astype(np.uint32)


def make_table(M, NTL, A, W, O, tile_words, pitch_out, rm=False, nthreads=8):
    """Per-thread table rows (A, W, O are base addresses in bytes; O already includes nothing else): [A base, weight stream, out base, tiles, valid rows, 0, 0, 0].
    Returns (table uint64 [8][8], G)."""
    nblk = max(1, (M + 63) // 64)
    assert nblk <= nthreads, 'M > 512'
    G = nthreads // nblk
    tab = np.zeros((nthreads, 8), np.uint64)
    for t in range(nthreads):
        mb, g = t % nblk, t // nblk
        active = t < G * nblk and g < NTL
        if not active:
            mb, g = 0, 0
        ntl = (NTL - g + G - 1) // G if active else 1
        v = max(0, min(M - 64 * mb, 1 << 20)) if active else 0
        ob = O + 4 * (64 * mb + g * 8 * pitch_out) if not rm else O + 4 * (64 * mb * pitch_out + g * 8)
        tab[t] = [A + 4 * 64 * mb, W + g * tile_words * 4, ob, ntl, v, 0, 0, 0]
    return tab, G


def build(outdir):
    for v in VARIANTS:
        code = np.array(assemble(make_kernel(v)), dtype=np.uint64)
        code.tofile(os.path.join(outdir, FILES[v]))
        print(f'{len(code)} instructions -> {os.path.join(outdir, FILES[v])}')


def test(variant='plain', M=251, K=176, N=176, reps=int(os.environ.get('REPS', '40'))):
    import _videocore6.driver as D
    _o = D.DRM_V3D
    D.DRM_V3D = lambda *a, **k: _o('/dev/dri/renderD128')
    from videocore6.driver import Driver
    rm = variant == 'rm'
    TP = (M + 63) // 64 * 64
    OPITCH = TP if not rm else N + 16
    rng = np.random.default_rng(5)
    Wm = (rng.standard_normal((N, K)) * 0.03).astype(np.float32); bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
    A0 = (rng.standard_normal((K, M)) * 0.5).astype(np.float32)
    stream = pack8(Wm, bias)
    tile_words = 8 + 4 * K
    scale = 0.5 if variant == 'rs' else (-1.4426950408889634 if variant == 'silu' else 0.0)
    with Driver() as drv:
        code = drv.program(make_kernel(variant))
        A = drv.alloc(((K + 4) * TP + 64,), dtype=np.float32); Af = np.zeros(A.shape[0], np.float32)
        Af[:K * TP].reshape(K, TP)[:, :M] = A0; A[:] = Af
        osz = (N * OPITCH + 64) if not rm else (M + 64) * OPITCH + 64
        O = drv.alloc((osz,), dtype=np.float32); O[:] = -7.0
        O0 = np.array(O).copy()
        if variant == 'rs':
            R0 = (rng.standard_normal(osz) * 0.3).astype(np.float32); O[:] = R0; O0 = R0.copy()
        Ws = drv.alloc((stream.size + 16,), dtype=np.uint32); Ws[:stream.size] = stream
        TB = drv.alloc((8, 8), dtype=np.uint32)
        tab, G = make_table(M, N // 8, int(A.addresses()[0]), int(Ws.addresses()[0]), int(O.addresses()[0]), tile_words, OPITCH, rm)
        TB[:] = tab.astype(np.uint32)
        OTILE = (8 * G * 4) if rm else (8 * G * OPITCH * 4)
        sb = np.array([scale], np.float32).view(np.uint32)[0]
        u = [0 if os.environ.get('Q4PI0') else TP * 4, OPITCH * 4, tile_words * 4 * G, OTILE, K // 2, int(TB.addresses()[0, 0])] + ([int(sb)] if variant in ('rs', 'silu') else [])
        unif = drv.alloc(len(u) + 2, dtype=np.uint32); unif[:len(u)] = np.array(u, dtype=np.uint64).astype(np.uint32)
        best = 1e9
        for _ in range(reps):
            O[:] = O0
            t0 = time.monotonic(); drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=8); best = min(best, time.monotonic() - t0)
        Of = np.array(O)
        acc = Wm.astype(np.float16).astype(np.float32) @ A0 + bias[:, None]          # [N][M]
        if variant == 'relu': want = np.maximum(acc, 0)
        elif variant == 'silu': want = acc / (1 + np.exp(-acc))
        elif variant == 'rs': want = O0[:N * OPITCH].reshape(N, OPITCH)[:, :M] + scale * acc
        else: want = acc
        if rm:
            got = Of[:M * OPITCH].reshape(M, OPITCH)[:, :N].T
        else:
            got = Of[:N * OPITCH].reshape(N, OPITCH)[:, :M]
        err = float(np.abs(got - want).max())
        # untouched padding check
        if rm:
            pad_ok = bool((Of[:M * OPITCH].reshape(M, OPITCH)[:, N:] == O0[:M * OPITCH].reshape(M, OPITCH)[:, N:]).all() and (Of[M * OPITCH:] == O0[M * OPITCH:]).all())
        else:
            pad_ok = bool((Of[:N * OPITCH].reshape(N, OPITCH)[:, M:] == O0[:N * OPITCH].reshape(N, OPITCH)[:, M:]).all() and (Of[N * OPITCH:] == O0[N * OPITCH:]).all())
        print(f'{variant:6s} M={M:4d} K={K:4d} N={N:4d}: {best*1e3:6.2f} ms ({M*N*K/best/1e9:5.2f} GMAC/s incl. launch)  max err {err:.3g}  padding untouched {pad_ok}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        build(sys.argv[2] if len(sys.argv) > 2 else '.')
    else:
        a = sys.argv[2:]
        test(a[0] if a else 'plain', *[int(x) for x in a[1:]])
