"""Self-test on the real GPU. Each check is numpy vs GPU for one op, then a small plan with bisect.
usage: python3 ~/gpu-port-kit/selftest.py [--timing] [--speech-app]
--speech-app also lints the real nemo8 plan built by ~/speech-app/port/gpu_encoder2.py (if present)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuport import Arena, Weights, Plan, ops, check
from gpuport.timing import time_op
from gpuport.config import merged_shader_dir
SHADER_DIR = merged_shader_dir()

rng = np.random.default_rng(0)
allok = True


def gemm_ref(A, Wm, b, epi):
    y = A @ Wm.T + b
    if epi == 2:
        y = y / (1 + np.exp(-y))
    return y.astype(np.float32)


def test_add_silu():
    global allok
    n = 1000
    ar = Arena(); a = ar.alloc('a', n); b = ar.alloc('b', n); s = ar.alloc('s', n)
    x = rng.standard_normal(n).astype(np.float32); y = rng.standard_normal(n).astype(np.float32)
    arena = np.zeros(ar.used(), np.float32); arena[a:a + n] = x; arena[b:b + n] = y
    p = Plan(); ops.add(p, n, a, b, 0.5)
    ok, _ = check.op_test('add (a += 0.5*b)', p, arena, a, x + 0.5 * y, 1e-6)
    allok &= ok
    arena = np.zeros(ar.used(), np.float32); arena[a:a + n] = x
    p = Plan(); ops.silu(p, n, a, s)
    ok, _ = check.op_test('silu', p, arena, s, x / (1 + np.exp(-x)), 1e-5)
    allok &= ok


def test_gemm12():
    global allok
    for (M, K, N, epi) in [(37, 50, 20, 0), (37, 50, 20, 2), (300, 64, 33, 0)]:
        A = rng.standard_normal((M, K)).astype(np.float32)
        Wm = rng.standard_normal((N, K)).astype(np.float32) * 0.2
        b = rng.standard_normal(N).astype(np.float32)
        w = Weights(); mm = w.put_mm(Wm, b)
        wpath = w.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work', 'gemm_test'))
        ar = Arena(); ia = ar.alloc('A', M * K); io = ar.alloc('O', M * N)
        arena = np.zeros(ar.used(), np.float32); arena[ia:ia + M * K] = A.ravel()
        p = Plan(); ops.gemm12(p, ia, M, mm, io, epi=epi)
        ok, _ = check.op_test(f'gemm12 M={M} K={K} N={N} epi={epi} ({len(p)} blocks)', p, arena, io,
                              gemm_ref(A, Wm, b, epi), 1e-3, weights=wpath)
        allok &= ok


def test_gemm12_kscratch():
    """gemm12 through the k-major path (tr_km + gemm12av4 / gemm12av4o4), vs numpy."""
    global allok
    for M, K, N, epi, stride in [(37, 52, 20, 0, 52), (37, 52, 20, 2, 52),
                                  (300, 64, 40, 0, 64), (300, 64, 33, 0, 64),
                                  (77, 512, 257, 0, 516), (300, 257, 80, 0, 261)]:
        A = rng.standard_normal((M, K)).astype(np.float32)
        Wm = rng.standard_normal((N, K)).astype(np.float32) * 0.2
        b = rng.standard_normal(N).astype(np.float32)
        w = Weights(); mm = w.put_mm(Wm, b)
        wpath = w.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work', 'gemm_test'))
        ar = Arena(); ia = ar.alloc('A', M * stride); io = ar.alloc('O', M * N); ik = ar.alloc('TK', K * 252)
        arena = np.zeros(ar.used(), np.float32); padded = np.full((M, stride), np.nan, np.float32)
        padded[:, :K] = A
        arena[ia:ia + M * stride] = padded.ravel()
        p = Plan(); ops.gemm12(p, ia, M, mm, io, epi=epi, sAm=stride, kscratch=ik)
        assert any(op.shader.startswith('gemm12av4') for op in p.ops)
        ok, _ = check.op_test(f'gemm12 kscratch M={M} K={K} N={N} stride={stride} epi={epi} ({len(p)} ops)', p, arena, io,
                              gemm_ref(A, Wm, b, epi), 1e-3, weights=wpath)
        allok &= ok


def test_bisect_and_lint():
    global allok
    n = 512
    ar = Arena(); a = ar.alloc('a', n); s = ar.alloc('s', n)
    x = rng.standard_normal(n).astype(np.float32)
    arena = np.zeros(ar.used(), np.float32); arena[a:a + n] = x
    p = Plan(); ops.add(p, n, a, a, 1.0)     # op 0: a = 2x
    ops.silu(p, n, a, s)                       # op 1: s = silu(2x)
    want0 = 2 * x; want1 = 2 * x / (1 + np.exp(-2 * x))
    bad = check.bisect('two-op', p, arena, [(0, 'doubled', a, want0, 1e-6), (1, 'silu', s, want1, 1e-5)])
    allok &= bad is None
    # lint must reject a 1-workgroup dispatch and a local-size mismatch
    bad_plan = Plan().add('add', 128, 1, 1, [n, a, a, 0])
    errs = [m for lvl, m in bad_plan.lint(SHADER_DIR) if lvl == 'error']
    print('lint catches 1 workgroup:', bool(errs))
    allok &= bool(errs)
    mism = Plan().add('add', 64, 24, 1, [n, a, a, 0])
    errs = [m for lvl, m in mism.lint(SHADER_DIR) if lvl == 'error']
    print('lint catches local mismatch:', any('local' in e for e in errs))
    allok &= any('local' in e for e in errs)


    # lint must warn about a power-of-two row stride on a row-major matmul (7x slower at 512, GPU_MAP section 10)
    w = Weights(); mm = w.put_mm(np.zeros((12, 512), np.float32))
    slow = ops.gemm12(Plan(), 0, 8, mm, 600000, sAm=512)
    fine = ops.gemm12(Plan(), 0, 8, mm, 600000, sAm=516)
    hit = any('power-of-two' in m for _, m in slow.lint(SHADER_DIR)) and not any('power-of-two' in m for _, m in fine.lint(SHADER_DIR))
    print('lint warns about a 512 row stride (and not 516):', hit)
    allok &= hit


def test_timing():
    n = 16384
    ar = Arena(); a = ar.alloc('a', n); b = ar.alloc('b', n)
    arena = np.zeros(ar.used(), np.float32); arena[a:a + n] = 1.0
    p = Plan(); ops.add(p, n, a, b, 0.0)
    time_op(p, arena, a, n, repeats=20)


if __name__ == '__main__':
    test_add_silu()
    test_gemm12()
    test_gemm12_kscratch()
    test_bisect_and_lint()
    if '--timing' in sys.argv:
        test_timing()
    if '--speech-app' in sys.argv:
        sys.path.insert(0, os.path.expanduser('~/speech-app/port'))
        import json
        import gpu_encoder2 as g
        L = json.load(open(os.path.expanduser('~/speech-app/port/cache2/layout.json')))
        ops_, size, R, Tm, T = g.build(16000, L)
        p = Plan([])
        for name, local, gx, gy, push in ops_:
            p.add(name, local, gx, gy, push)
        msgs = p.lint(SHADER_DIR)
        errs = [m for lvl, m in msgs if lvl == 'error']
        print(f'nemo8 plan (1 s audio): {len(p)} ops, arena {size} floats, lint errors {len(errs)}, warnings {len(msgs) - len(errs)}')
        for m in errs[:10]:
            print('  ', m)
        for lvl, m in msgs[:3]:
            print('  ', lvl, m)
    print('ALL OK' if allok else 'SOME CHECKS FAILED')
    raise SystemExit(0 if allok else 1)
