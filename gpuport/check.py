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


# Guard value for memory checks: finite, never produced by a model, compared bit for bit.
CANARY = np.float32(-7.7771e33)


def guard_run(label, plan, arena, inputs, writes, scratch=(), weights=None, poison=False, runner=None, **kw):
    """Run a plan and check WHERE it wrote, not only what (added 2026-11-12).
    arena: the Arena object. inputs: {region name: array}. writes: regions the plan must fill completely.
    scratch: regions the plan may write in part (temporaries, padded rows).
    Everything else (other regions, every gap, the unfilled tail of an input region) is filled with CANARY
    and must come back bit-identical. Catches the three bugs from the notes: rows written past the end of an
    output (runner bug 2026-11-05), a scratch buffer on top of live data (2026-11-09), and outputs that were
    never computed (rows 256+).
    poison=True fills the guard areas with NaN instead, so a READ outside the inputs shows up as NaN in the
    output (shaders that load padding on purpose and discard it stay clean; ones that use it do not).
    runner: optional callable(plan, full_arena, out_off, out_n) -> (out, ms), e.g. ExecServer.run.
    Returns (ok, {name: array for writes + scratch}, problems list)."""
    size = arena.used()
    guard = np.float32(np.nan) if poison else CANARY
    full = np.full(size, guard, np.float32)
    for name, val in inputs.items():
        val = np.asarray(val, np.float32).ravel()
        o, n = arena.regions[name]
        if val.size > n:
            raise ValueError(f'input {name}: {val.size} floats > region {n}')
        full[o:o + val.size] = val
    before = full.view(np.uint32).copy()
    if runner is not None:
        got, ms = runner(plan, full, 0, size)
    else:
        got, ms = run(plan, full, 0, size, weights=weights, **kw)
    gb = got.view(np.uint32)
    may_write = np.zeros(size, bool)
    problems, out = [], {}
    for name in list(writes) + list(scratch):
        o, n = arena.regions[name]
        may_write[o:o + n] = True
        out[name] = got[o:o + n].copy()
    stray = np.flatnonzero((gb != before) & ~may_write)
    if stray.size:
        runs = np.split(stray, np.flatnonzero(np.diff(stray) != 1) + 1)
        for r in runs[:5]:
            problems.append(f'wrote outside its regions: {r.size} floats at {arena.where(int(r[0]))} '
                            f'(offset {int(r[0])}..{int(r[-1])})')
        if len(runs) > 5:
            problems.append(f'... and {len(runs) - 5} more stray ranges ({stray.size} floats in total)')
    gbits = np.float32(guard).view(np.uint32)
    for name in writes:
        o, n = arena.regions[name]
        if name in inputs:
            continue                               # in-place: cannot tell written from kept
        left = np.flatnonzero(gb[o:o + n] == gbits) if not poison else np.flatnonzero(np.isnan(got[o:o + n]))
        if left.size:
            problems.append(f'{name}: {left.size} of {n} floats never written'
                            f'{" (or NaN)" if poison else ""}, first at {name}+{int(left[0])}')
    ok = not problems
    print(f'{"ok  " if ok else "FAIL"} {label}: memory guard ({"NaN poison" if poison else "canary"}), '
          f'{size} floats, gpu {ms:.1f} ms')
    for p in problems:
        print('     ' + p)
    return ok, out, problems
