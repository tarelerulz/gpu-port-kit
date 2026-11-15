"""Attention on the matmul kernel (ops.attention_km, 2026-11-12) vs numpy, with the memory guard, then speed
against the old chain (att_scores5jf + att_softmax + att_pv6) at nemo8 size.
usage: python3 tests/test_attention_km.py [--time]"""
import os
import sys
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, 'tests'))
from gpuport import Arena, Weights, Plan, ops, check
from gpuport.config import WORK_DIR, MIN_WORKGROUPS
from test_attention_layout import pos_table, att_ref

rng = np.random.default_rng(4)
TMAX = 260        # cached position table size in this test: T = 8..251 give every window offset 0..3 mod 4


def setup(T, d, H, old=False):
    NPr = 2 * T - 1
    qkv = rng.standard_normal((T, 3 * d)).astype(np.float32)
    Wpos = (rng.standard_normal((d, d)) * 0.05).astype(np.float32); bpos = (rng.standard_normal(d) * 0.1).astype(np.float32)
    pbu = (rng.standard_normal(d) * 0.1).astype(np.float32); pbv = (rng.standard_normal(d) * 0.1).astype(np.float32)
    w = Weights(); mm = w.put_mm(Wpos, bpos); pbuO = w.put(pbu); pbvO = w.put(pbv)
    tab, pitch = ops.pos_table_km(Wpos, bpos, TMAX); tabO = w.put(tab); w.put(np.zeros(16, np.float32))
    path = w.save(os.path.join(WORK_DIR, f'attkm_{T}_{d}_{H}'))
    ar = Arena()
    for name, size in [('QKV', T * 3 * d), ('PE', NPr * d), ('ATT', T * d), ('TK', d * 252)] + ops.att_km_regions(T, d, H):
        ar.alloc(name, size)
    if old:
        ar.alloc('P', NPr * d); ar.alloc('SC', H * T * T); ar.alloc('ATT2', T * d)
    P_ref = (pos_table(T, d) @ Wpos.T + bpos).astype(np.float32)
    want = att_ref(qkv, P_ref, pbu, pbv, T, d, H, d // H)
    return dict(tab=dict(off=tabO, Tmax=TMAX, pitch=pitch), ar=ar, qkv=qkv, pe=pos_table(T, d), mm=mm, pbu=pbuO, pbv=pbvO, path=path, want=want)


def run_config(T, d, H):
    s = setup(T, d, H); ar = s['ar']
    ok = True
    for tab in (None, s['tab']):
        p = ops.attention_km(Plan(), T, d, H, ar['QKV'], ar['PE'], s['mm'], s['pbu'], s['pbv'], ar['ATT'],
                             {n: ar[n] for n in ar.regions}, pos_tab=tab)
        for poison in (False, True):
            g, out, _ = check.guard_run(f'attention_km T={T} d={d} H={H}{" cached positions" if tab else ""}', p, ar,
                                        {'QKV': s['qkv'], 'PE': s['pe']}, ['ATT'],
                                        ['TK', 'QU', 'QV', 'KT', 'PT', 'ST', 'GT'], weights=s['path'], poison=poison)
            ok &= g and check.compare('     values vs numpy', out['ATT'], s['want'], 2e-4, quiet=True)
    return ok


def timing(T=251, d=176, H=4):
    from gpuport.profile import profile
    s = setup(T, d, H, old=True); ar = s['ar']; dh = d // H; NPr = 2 * T - 1
    arena = np.zeros(ar.used(), np.float32)
    arena[ar['QKV']:ar['QKV'] + s['qkv'].size] = s['qkv'].ravel(); arena[ar['PE']:ar['PE'] + s['pe'].size] = s['pe'].ravel()
    new = ops.attention_km(Plan(), T, d, H, ar['QKV'], ar['PE'], s['mm'], s['pbu'], s['pbv'], ar['ATT'], {n: ar[n] for n in ar.regions})
    old = Plan(); ops.gemm12(old, ar['PE'], NPr, s['mm'], ar['P'], kscratch=ar['TK'])
    push = [T, d, H, dh, ar['QKV'], ar['P'], ar['SC'], s['pbu'], s['pbv']]
    old.dispatch('att_scores5jf', 64, H * T * ((T + 3) // 4), push + [0] * 23)
    old.dispatch('att_softmax', 64, H * T, push + [ar['ATT2']] + [0] * 22)
    old.add('att_pv6', 64, max(MIN_WORKGROUPS, (H * ((T + 1) // 2) * (dh // 4) + 63) // 64), 1, push + [ar['ATT2']] + [0] * 22)
    cached = ops.attention_km(Plan(), T, d, H, ar['QKV'], ar['PE'], s['mm'], s['pbu'], s['pbv'], ar['ATT'],
                              {n: ar[n] for n in ar.regions}, pos_tab=s['tab'])
    both = Plan(old.ops + new.ops + cached.ops)
    rows = profile(both, arena, weights=s['path'], quiet=True)
    for label, a, b in [('old chain', 0, len(old)), ('new chain', len(old), len(old) + len(new)), ('new chain, cached positions', len(old) + len(new), len(both))]:
        print(f'{label}: {sum(r[2] for r in rows[a:b]):.2f} ms')
        by = {}
        for _, name, t, _ in rows[a:b]:
            by.setdefault(name, [0, 0.0]); by[name][0] += 1; by[name][1] += t
        for name, (c, t) in by.items():
            print(f'    {name:16s} {c:2d} x  {t:7.2f} ms')
    got, _ = __import__('gpuport.run', fromlist=['run']).run(both, arena, ar['ATT'], T * d, weights=s['path'], arena_floats=24 << 20,
                                                             exec_bin=os.path.join(KIT, 'tools', 'exec_big'))
    check.compare('new chain vs numpy at full size', got, s['want'], 2e-4)


if __name__ == '__main__':
    res = [run_config(8, 64, 2), run_config(13, 176, 4), run_config(21, 176, 4), run_config(126, 176, 4), run_config(250, 176, 4), run_config(251, 176, 4)]
    if '--time' in sys.argv:
        timing()
    print('ALL ATTENTION KM TESTS PASS' if all(res) else 'ATTENTION KM TESTS FAILED')
    raise SystemExit(0 if all(res) else 1)
