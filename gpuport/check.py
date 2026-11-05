"""Correctness helpers: compare GPU regions with numpy, test one op in isolation, bisect a plan."""
import numpy as np
from .run import run, run_prefix
from .plan import Plan


def compare(label, got, want, tol=1e-4, quiet=False):
    """Max abs diff and where it is. tol is absolute. Returns True when within tol."""
    got = np.asarray(got, np.float32).ravel(); want = np.asarray(want, np.float32).ravel()
    if got.shape != want.shape:
        print(f'FAIL {label}: shape {got.shape} vs {want.shape}')
        return False
    d = np.abs(got - want)
    worst = int(np.argmax(d)) if d.size else 0
    mx = float(d.max()) if d.size else 0.0
    ok = bool(np.isfinite(got).all() and mx <= tol)
    if not quiet or not ok:
        scale = float(np.abs(want).mean()) or 1.0
        print(f'{"ok  " if ok else "FAIL"} {label}: max diff {mx:.3g} (at {worst}: got {got[worst]:.6g} want {want[worst]:.6g}), '
              f'mean diff {d.mean():.3g}, ref |mean| {scale:.3g}')
    return ok


def op_test(label, plan, arena_init, out_off, want, tol=1e-4, weights=None, **kw):
    """Run one plan (usually one op) and compare the output region with the numpy result."""
    want = np.asarray(want, np.float32).ravel()
    got, ms = run(plan, arena_init, out_off, want.size, weights=weights, **kw)
    ok = compare(label, got, want, tol)
    print(f'     gpu submit {ms:.2f} ms')
    return ok, got


def bisect(label, plan, arena_init, checkpoints, weights=None, **kw):
    """checkpoints: list of (after_op_index, name, out_off, want, tol) in plan order. For each one,
    runs the plan prefix that ends at after_op_index and compares that region. Prints every result and
    returns the first failing name (or None). This finds the first wrong stage (the method that worked)."""
    first_bad = None
    for after, name, off, want, tol in checkpoints:
        want = np.asarray(want, np.float32).ravel()
        got, _ = run_prefix(plan, after + 1, arena_init, off, want.size, weights=weights, **kw)
        ok = compare(f'{label} [{name} after op {after}]', got, want, tol)
        if not ok and first_bad is None:
            first_bad = name
            print(f'     first wrong stage: {name}; check the op that produced it and its inputs')
            break
    return first_bad
