"""Per-dispatch GPU profile of any plan (tools/exec_prof: Vulkan timestamps around each dispatch).
The generic form of tools/profile_nemo8.py (added 2026-11-12). Build the runner once: python3 tools/build_exec_prof.py

    from gpuport.profile import profile
    rows = profile(plan, arena_init, weights=wpath, marks=[('front', 0), ('layers', 40)])

Matmul shaders (gemm12*: push M, N, K) also get their rate in GMAC/s, so a slow call stands out against the
measured ceilings (row-major gemm12 about 2, k-major AV4 about 5-6)."""
import collections
import os
from .config import KIT_ROOT, WORK_DIR
from .run import run
from .server import SERVER_ARENA_FLOATS

EXEC_PROF = os.path.join(KIT_ROOT, 'tools', 'exec_prof')


def _gmac(op):
    if op.shader.startswith('gemm12') or op.shader in ('gemm4', 'gemm_k5'):
        return op.push[0] * op.push[1] * op.push[2] / 1e9
    return None


def profile(plan, arena_init, weights=None, marks=None, top=10, quiet=False, **kw):
    """Runs the plan once on exec_prof. Returns a list of (index, shader, ms, GMAC or None).
    marks: [(stage name, first op index), ...] for a by-stage table (optional)."""
    if not os.path.exists(EXEC_PROF):
        raise FileNotFoundError(f'{EXEC_PROF} missing: run python3 tools/build_exec_prof.py')
    pf = os.path.join(WORK_DIR, f'prof{os.getpid()}.txt')
    kw.setdefault('arena_floats', SERVER_ARENA_FLOATS)
    run(plan, arena_init, 0, 1, weights=weights, exec_bin=EXEC_PROF, prof=pf, **kw)
    times = [float(l.split()[2]) for l in open(pf)]
    os.remove(pf)
    rows = [(i, op.shader, t, _gmac(op)) for i, (op, t) in enumerate(zip(plan.ops, times))]
    if quiet:
        return rows
    total = sum(times) or 1e-9
    print(f'{len(rows)} dispatches, sum of GPU times {total:.1f} ms')
    by = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for _, name, t, g in rows:
        by[name][0] += 1; by[name][1] += t; by[name][2] += g or 0.0
    print('by shader:')
    for name, (c, t, g) in sorted(by.items(), key=lambda x: -x[1][1]):
        rate = f'  {g / (t / 1e3):5.2f} GMAC/s' if g and t > 0 else ''
        print(f'  {name:16s} {c:4d} x {t / c:8.3f} ms = {t:8.1f} ms {100 * t / total:5.1f}%{rate}')
    if marks:
        print('by stage:')
        ends = list(marks[1:]) + [('end', len(rows))]
        for (name, a), (_, b) in zip(marks, ends):
            t = sum(times[a:b])
            print(f'  {name:12s} {t:8.1f} ms {100 * t / total:5.1f}%')
    print(f'top {top} dispatches:')
    for i, name, t, g in sorted(rows, key=lambda r: -r[2])[:top]:
        rate = f'  {g / (t / 1e3):5.2f} GMAC/s' if g and t > 0 else ''
        print(f'  op {i:4d} {name:16s} {t:8.2f} ms{rate}')
    return rows
