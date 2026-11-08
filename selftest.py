"""Self-test on the real GPU. Each check is numpy vs GPU for one op, then a small plan with bisect.
usage: python3 ~/gpu-port-kit/selftest.py [--timing] [--oido]
--oido also lints the real nemo8 plan built by ~/oido/port/gpu_encoder2.py (if present)."""
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


def test_timing():
    n = 16384
    ar = Arena(); a = ar.alloc('a', n); b = ar.alloc('b', n)
    arena = np.zeros(ar.used(), np.float32); arena[a:a + n] = 1.0
    p = Plan(); ops.add(p, n, a, b, 0.0)
    time_op(p, arena, a, n, repeats=20)


if __name__ == '__main__':
    test_add_silu()
    test_gemm12()
    test_bisect_and_lint()
    if '--timing' in sys.argv:
        test_timing()
    if '--oido' in sys.argv:
        sys.path.insert(0, os.path.expanduser('~/oido/port'))
        import json
        import gpu_encoder2 as g
        L = json.load(open(os.path.expanduser('~/oido/port/cache2/layout.json')))
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
