"""Per-op GPU time from exec's submit+wait. Method: run the op once, then repeated R times in one plan;
per-op time = (t_R - t_1) / (R - 1). Ops are not reset between repeats, so use ops whose values
do not matter for timing (values may grow for accumulating ops such as add)."""
from .plan import Plan
from .run import run


def time_op(plan_single, arena_init, out_off, out_n, repeats=20, weights=None, **kw):
    """plan_single: Plan with the op(s) to time (usually one op). Returns ms per repetition."""
    one = plan_single.ops
    _, t1 = run(Plan(one), arena_init, out_off, out_n, weights=weights, **kw)
    _, tr = run(Plan(one * repeats), arena_init, out_off, out_n, weights=weights, **kw)
    per = (tr - t1) / (repeats - 1)
    print(f'time {len(one)} op(s) x{repeats}: 1x {t1:.2f} ms, {repeats}x {tr:.2f} ms -> {per:.3f} ms per rep')
    return per
