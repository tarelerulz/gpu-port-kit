"""Memory guard checks (added 2026-11-12): every matmul path and the elementwise ops must write only their output
region and fill it completely; the guard must also catch a deliberate overrun and a short write.
usage: python3 tests/test_memory_guard.py"""
import os
import sys
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport import Arena, Weights, Plan, ops, check
from gpuport.config import WORK_DIR, merged_shader_dir

SHADERS = merged_shader_dir()
rng = np.random.default_rng(5)
allok = True


def matmul_case(M, K, N, stride, epi, kmajor, poison):
    global allok
    A = rng.standard_normal((M, K)).astype(np.float32)
    Wm = rng.standard_normal((N, K)).astype(np.float32) * 0.2
    b = rng.standard_normal(N).astype(np.float32)
    w = Weights(); mm = w.put_mm(Wm, b); wpath = w.save(os.path.join(WORK_DIR, 'guard_test'))
    ar = Arena(); ar.alloc('A', (M - 1) * stride + K); ar.alloc('O', M * N)
    scratch = ()
    if kmajor:
        ar.alloc('TK', K * 252); scratch = ('TK',)
    rows = np.full((M, stride), check.CANARY if not poison else np.nan, np.float32); rows[:, :K] = A
    p = Plan(); ops.gemm12(p, ar['A'], M, mm, ar['O'], epi=epi, sAm=stride, kscratch=ar['TK'] if kmajor else None)
    label = f'{"k-major" if kmajor else "row-major"} M={M} K={K} N={N} stride={stride} epi={epi}'
    ok, out, _ = check.guard_run(label, p, ar, {'A': rows.ravel()[:(M - 1) * stride + K]}, ['O'], scratch,
                                 weights=wpath, poison=poison, shader_dir=SHADERS)
    y = A @ Wm.T + b
    if epi == 2:
        y = y / (1 + np.exp(-y))
    ok &= check.compare('     values', out['O'], y, 1e-3, quiet=True)
    allok &= ok


# shapes: M not a multiple of 4, N not a multiple of 12 or 4, two row blocks, padded stride, odd K
for kmajor in (False, True):
    for poison in (False, True):
        for M, K, N, stride, epi in [(37, 52, 20, 52, 0), (37, 52, 20, 56, 2), (300, 64, 33, 64, 0),
                                      (253, 257, 80, 261, 0), (5, 16, 12, 16, 0), (251, 176, 704, 176, 0)]:
            matmul_case(M, K, N, stride, epi, kmajor, poison)

# elementwise ops
n = 1000
ar = Arena(); ar.alloc('a', n); ar.alloc('b', n); ar.alloc('s', n)
x = rng.standard_normal(n).astype(np.float32); y = rng.standard_normal(n).astype(np.float32)
ok, out, _ = check.guard_run('add', ops.add(Plan(), n, ar['a'], ar['b'], 0.5), ar, {'a': x, 'b': y}, ['a'], shader_dir=SHADERS)
allok &= ok and check.compare('     values', out['a'], x + 0.5 * y, 1e-6, quiet=True)
ok, out, _ = check.guard_run('silu', ops.silu(Plan(), n, ar['a'], ar['s']), ar, {'a': x}, ['s'], shader_dir=SHADERS)
allok &= ok and check.compare('     values', out['s'], x / (1 + np.exp(-x)), 1e-5, quiet=True)

# the guard itself: an overrun and a short write must both be reported
print('-- two deliberate faults follow (FAIL lines are expected) --')
ok, _, probs = check.guard_run('overrun (silu told n+7)', ops.silu(Plan(), n + 7, ar['a'], ar['s']), ar, {'a': x}, ['s'], shader_dir=SHADERS)
caught = (not ok) and any('gap after s' in p for p in probs)
print('guard catches a write past the region:', caught); allok &= caught
ok, _, probs = check.guard_run('short write (silu told n-9)', ops.silu(Plan(), n - 9, ar['a'], ar['s']), ar, {'a': x}, ['s'], shader_dir=SHADERS)
caught = (not ok) and any('9 of 1000 floats never written' in p for p in probs)
print('guard catches an unwritten tail:', caught); allok &= caught

print('ALL MEMORY GUARD TESTS PASS' if allok else 'SOME MEMORY GUARD TESTS FAILED')
raise SystemExit(0 if allok else 1)
