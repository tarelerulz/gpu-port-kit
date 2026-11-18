"""QPU matmul, 1 vector x 8 columns per tile (2026-11-16): row tails of at most 16 rows.

Why: the row-tiling kernels compute whole blocks (32 rows in qpu_mm.py, 64 in qpu_mm4.py) whatever the number of valid rows, so the few rows
left after 64 / 128 / 256 cost a whole tile pass per thread. This kernel has one 16-lane vector of rows per thread and reads the same weight
stream as qpu_mm4.py (8-column tiles: 8 bias words, then K * 4 f16-pair words), so 8 threads share the N / 8 tiles of one 16-row block.

out[n][m] = bias[n] + sum_k W[k][n] A[k][m] for m < M <= 16. A k-major [K][pitch]; out k-major [N][pitch].
Variants: plain, relu, rs (out += scale * (acc + bias)).
Uniform words (ldunif): PI (A pitch bytes), OP (out pitch bytes), WT (tile step bytes = tile bytes * G), OTILE (out step per tile step, bytes), KP (K / 2),
NT (thread table address), RSC (rs only).
Thread table (8 words at NT + 32 t): A base, weight stream start, out base, tiles, valid rows V (<= 0: idle, all stores masked). Same as qpu_mm4.py.
x loads: Q1PD rows of k are requested ahead (default 4; the TMU holds 8 pending vectors, so up to 8 fit here; qpu_mm4 can only do 2).
usage: PYTHONPATH=shim:<py-videocore6>/src python3 qpu_mm1.py [build <dir> | test [variant] [M] [K] [N]]"""
import os
import sys
import time
import numpy as np
from _videocore6.assembler import Instruction, qpu, assemble
_R = Instruction.REGISTERS
rfr = lambda i: _R['rf%d' % i]
NA = 8
ACC = [rfr(i) for i in range(NA)]
XS = [_R['r0'], _R['r1']]
(P0, PI, PI2, Z, KCNT, KP, WADDR, WT, ABASE, TCNT, OB, OTILE, OCUR, V, RSC, NT, OP) = [rfr(i) for i in range(8, 25)]
VARIANTS = ('plain', 'relu', 'rs')
FILES = {'plain': 'mm1.qpu', 'relu': 'mm1relu.qpu', 'rs': 'mm1rs.qpu'}
PD = int(os.environ.get('Q1PD', '4'))           # measured 2026-11-16, K=1584: PD 2 0.84 ms, 4 0.73, 8 0.73
assert 2 <= PD <= 8
EXP = set(os.environ.get('Q1EXP', '').split(','))          # timing experiments only: nowt (no weight stream reads in the loop)


def emit_block(s, odd, lk=None):
    """One k: 8 multiply-adds (8 columns x 1 vector), 4 weight words, 1 x request (k + PD), 1 x pop (k + 1)."""
    X, Xn = XS[s], XS[1 - s]
    for t in range(8):
        kw = {}
        if t % 2 == 1:
            if 'nowt' not in EXP:
                kw['sig'] = ldunifa                      # next f16 pair word (two n values)
        elif t == 6:
            kw['sig'] = ldtmu(Xn)
        if t == 0:
            add(tmua, P0, PI if odd else Z)              # x request for k + PD
        if odd and t == 1:
            add(P0, P0, PI2)
        if odd and t == 3:
            sub(KCNT, KCNT, 1, cond='pushz')
        if odd and t == 5:
            lk.b(cond='anyna')                           # t = 5, 6, 7 are its delay slots
        prev = ACC[(t - 1) % NA]
        wreg = _R['r5'].unpack('l') if t % 2 == 0 else _R['r5'].unpack('h')
        fadd(prev, prev, r3, **kw).fmul(r3, wreg, X)


def make_kernel(variant):
    assert variant in VARIANTS

    @qpu
    def kernel(asm):
        for reg in [PI, OP, WT, OTILE, KP, NT] + ([RSC] if variant == 'rs' else []):
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
        add(OB, OB, r2)
        mov(Z, 0)
        add(PI2, PI, PI)
        nop(sig=thrsw)
        nop()
        nop()
        with loop as lt:                                     # over n tiles
            mov(unifa, WADDR)
            add(WADDR, WADDR, WT)
            mov(KCNT, KP)
            mov(P0, ABASE)
            for _ in range(PD):                              # x requests for k = 0 .. PD - 1
                add(tmua, P0, Z)
                add(P0, P0, PI)                              # P0 = k PD at the end
            mov(r3, 0.0)
            nop(sig=ldtmu(XS[0]))                            # k = 0
            nop()
            nop(sig=ldunifa)                                 # bias 0 -> r5
            for n in range(8):                               # accumulators start at the bias; the last signal fetches weight pair 0
                mov(ACC[n], r5, sig=ldunifa)
            with loop as lk:
                emit_block(0, False)
                emit_block(1, True, lk)
            fadd(ACC[NA - 1], ACC[NA - 1], r3)
            for i in range(PD - 1):
                nop(sig=ldtmu(r0))                           # drain the requests beyond the end
            eidx(r1)
            sub(null, r1, V, cond='pushn')                   # flag A: lane m < V
            mov(OCUR, OB)
            for n in range(8):
                if variant == 'rs':
                    mov(tmua, OCUR)
                    fmul(r3, ACC[n], RSC)
                    nop(sig=ldtmu(r0))
                    fadd(r1, r0, r3)
                    mov(tmud, r1)
                elif variant == 'relu':
                    fmax(tmud, ACC[n], 0.0)
                else:
                    mov(tmud, ACC[n])
                mov(tmua, OCUR, cond='ifa')
                add(OCUR, OCUR, OP)
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
    """Same stream as qpu_mm4.pack8: per 8-column tile 8 bias words then K * 4 words (low half even n, high half odd n)."""
    N, K = W.shape
    h = np.asarray(W, np.float32).astype(np.float16).view(np.uint16).reshape(N // 8, 4, 2, K)
    words = h[:, :, 0, :].astype(np.uint32) | (h[:, :, 1, :].astype(np.uint32) << 16)
    words = np.transpose(words, (0, 2, 1))
    b = np.asarray(bias, np.float32).reshape(N // 8, 8).view(np.uint32)
    return np.concatenate([b, words.reshape(N // 8, K * 4)], axis=1).reshape(-1).astype(np.uint32)


def make_table(M, NTL, A, W, O, tile_words, pitch_out, nthreads=8):
    """[A base, weight stream, out base, tiles, valid rows, 0, 0, 0] per thread; one 16-row block, G = 8 tile groups. Returns (table, G)."""
    assert 1 <= M <= 16
    G = nthreads
    tab = np.zeros((nthreads, 8), np.uint64)
    for g in range(nthreads):
        active = g < NTL
        gg = g if active else 0
        ntl = (NTL - gg + G - 1) // G if active else 1
        tab[g] = [A, W + gg * tile_words * 4, O + 4 * gg * 8 * pitch_out, ntl, M if active else 0, 0, 0, 0]
    return tab, G


def build(outdir):
    for v in VARIANTS:
        code = np.array(assemble(make_kernel(v)), dtype=np.uint64)
        code.tofile(os.path.join(outdir, FILES[v]))
        print(f'{len(code)} instructions -> {os.path.join(outdir, FILES[v])}')


def test(variant='plain', M=7, K=176, N=176, reps=int(os.environ.get('REPS', '40'))):
    import _videocore6.driver as D
    _o = D.DRM_V3D
    D.DRM_V3D = lambda *a, **k: _o('/dev/dri/renderD128')
    from videocore6.driver import Driver
    TP = 64
    OPITCH = 64
    rng = np.random.default_rng(5)
    Wm = (rng.standard_normal((N, K)) * 0.03).astype(np.float32); bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
    A0 = (rng.standard_normal((K, M)) * 0.5).astype(np.float32)
    stream = pack8(Wm, bias)
    tile_words = 8 + 4 * K
    scale = 0.5 if variant == 'rs' else 0.0
    with Driver() as drv:
        code = drv.program(make_kernel(variant))
        A = drv.alloc(((K + 16) * TP + 64,), dtype=np.float32); Af = np.zeros(A.shape[0], np.float32)
        Af[:K * TP].reshape(K, TP)[:, :M] = A0; A[:] = Af
        osz = N * OPITCH + 64
        O = drv.alloc((osz,), dtype=np.float32); O[:] = -7.0
        O0 = np.array(O).copy()
        if variant == 'rs':
            R0 = (rng.standard_normal(osz) * 0.3).astype(np.float32); O[:] = R0; O0 = R0.copy()
        Ws = drv.alloc((stream.size + 16,), dtype=np.uint32); Ws[:stream.size] = stream
        TB = drv.alloc((8, 8), dtype=np.uint32)
        tab, G = make_table(M, N // 8, int(A.addresses()[0]), int(Ws.addresses()[0]), int(O.addresses()[0]), tile_words, OPITCH)
        TB[:] = tab.astype(np.uint32)
        sb = np.array([scale], np.float32).view(np.uint32)[0]
        u = [TP * 4, OPITCH * 4, tile_words * 4 * G, 8 * G * OPITCH * 4, K // 2, int(TB.addresses()[0, 0])] + ([int(sb)] if variant == 'rs' else [])
        unif = drv.alloc(len(u) + 2, dtype=np.uint32); unif[:len(u)] = np.array(u, dtype=np.uint64).astype(np.uint32)
        best = 1e9
        for _ in range(reps):
            O[:] = O0
            t0 = time.monotonic(); drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=8); best = min(best, time.monotonic() - t0)
        Of = np.array(O)
        acc = Wm.astype(np.float16).astype(np.float32) @ A0 + bias[:, None]          # [N][M]
        want = np.maximum(acc, 0) if variant == 'relu' else (O0[:N * OPITCH].reshape(N, OPITCH)[:, :M] + scale * acc if variant == 'rs' else acc)
        got = Of[:N * OPITCH].reshape(N, OPITCH)[:, :M]
        err = float(np.abs(got - want).max())
        pad_ok = bool((Of[:N * OPITCH].reshape(N, OPITCH)[:, M:] == O0[:N * OPITCH].reshape(N, OPITCH)[:, M:]).all() and (Of[N * OPITCH:] == O0[N * OPITCH:]).all())
        print(f'{variant:6s} M={M:4d} K={K:4d} N={N:4d} PD={PD}: {best*1e3:6.2f} ms  max err {err:.3g}  padding untouched {pad_ok}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        build(sys.argv[2] if len(sys.argv) > 2 else '.')
    else:
        a = sys.argv[2:]
        test(a[0] if a else 'plain', *[int(x) for x in a[1:]])
