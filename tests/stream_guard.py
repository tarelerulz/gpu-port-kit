"""Memory guard on the oido streaming plans (added 2026-11-12).
Part 1: streaming front end (gpuport/stream_fe.build), window sizes 1..252 mel rows, final and not final.
Part 2: the live encoder chunk plans (~/oido/port/stream_gpu.py chunk_ops + stream_gpu2 clear_tail / shift ops) with
        the shaders in ~/oido/port/gpu, state carried in the arena from chunk to chunk as the resident server does.
Each part runs three ways: clean (arena zero), canary (guard value everywhere the plan has not written) and NaN poison.
Pass = no write outside the regions, and the outputs are bit-identical in all three runs.
usage: python3 tests/stream_guard.py [fe|enc]"""
import os
import sys
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport import Arena, check
from gpuport.run import run
from gpuport.stream_fe import prepare, build, SHADERS, OIDO_PORT, ARENA_FLOATS

EXEC_BIG = os.path.join(KIT, 'tools', 'exec_big')
MODEL = os.path.join(OIDO_PORT, '..', 'models', 'oido_stream.tnm')
pcm = np.fromfile(os.path.join(OIDO_PORT, 'clip10s.pcm'), '<i2').astype(np.float32) / 32768
allok = True


def front_end():
    global allok
    weights, fe, sc, d = prepare(MODEL)
    kw = dict(weights=weights, shader_dir=SHADERS, exec_bin=EXEC_BIG, arena_floats=ARENA_FLOATS)
    for tm in (1, 21, 64, 76, 252):
        for final in (False, True):
            n = 160 * tm + 200
            x = np.resize(pcm, n).copy()
            if final:
                x[-280:] = 0
            plan, R, t3, _ = build(tm, n, tm - int(final), int(final), fe, sc, d)
            ar = plan.arena
            clean, _ = run(plan, x, R['X'], t3 * d, **kw)
            writes = [r for r in ar.regions if r not in ('XIN', 'FR', 'TK')]
            res = []
            for poison in (False, True):
                ok, out, _ = check.guard_run(f'stream front end Tm={tm} final={final}', plan, ar, {'XIN': x}, writes,
                                             ['FR', 'TK'], poison=poison, **kw)
                same = np.array_equal(out['X'], clean)
                if not same:
                    bad = np.flatnonzero(out['X'].view(np.uint32) != clean.view(np.uint32))
                    print(f'     X differs from the clean run: {bad.size} of {clean.size} floats, first at row {bad[0] // d}')
                res.append(ok and same)
            allok &= all(res)


def encoder(nframes=181):
    """11 full chunks + one 5-frame chunk: history passes 128 rows, so the shift copies run."""
    global allok
    sys.path.insert(0, OIDO_PORT)
    import stream_gpu as SG
    import stream_gpu2 as SG2
    import stream_ref as S
    sizes = []
    orig = SG.Plan.alloc
    def alloc(self, name, size):
        off = orig(self, name, size); sizes.append((name, off, size)); return off
    SG.Plan.alloc = alloc
    g = SG2.StreamGPU2()
    SG.Plan.alloc = orig
    ar = Arena(limit=ARENA_FLOATS)
    seen = {}
    for name, off, size in sizes:                 # a name allocated twice keeps its last offset in the engine
        seen[name] = seen.get(name, 0) + 1
    count = {}
    for name, off, size in sizes:
        count[name] = count.get(name, 0) + 1
        key = name if count[name] == seen[name] else f'{name}_unused{count[name]}'
        ar.regions[key] = (off, size)
    ar.cur = g.size
    dup = [k for k in ar.regions if '_unused' in k]
    print(f'encoder arena: {len(ar.regions)} regions, {g.size / 1e6:.2f}M floats; regions allocated twice: {dup or "none"}')
    L = g.L; d, V = L['d'], L['V']
    M = S.load(MODEL)
    X, T = S.frontend_X((pcm * 32768).astype(np.int16), M)
    X = X[:nframes]; T = len(X)
    wpath = os.path.join(g.cache, 'weights.bin')
    gdir = g.L_dir()
    kw = dict(weights=wpath, shader_dir=gdir, exec_bin=EXEC_BIG, arena_floats=ARENA_FLOATS, lint=False)
    fixed = [k for k in ar.regions if k.startswith('P') and k[1:].isdigit()] + dup     # must never change
    work = [k for k in ar.regions if k not in fixed]
    P = SG.Plan
    def as_plan(ops):
        from gpuport import Plan
        p = Plan()
        for name, local, gx, gy, push in ops:
            p.add(name, local, gx, gy, push[:32])
        return p
    def region(full, name):
        o, n = ar.regions[name]
        return full[o:o + n]
    # start of utterance, as init_state(): positional rows written by the host, history zeroed by the GPU
    zero_ops = []
    for lb in g.lay:
        for key, cnt in [('KC', (SG.LEFT + SG.C_CH) * d), ('VC', (SG.LEFT + SG.C_CH) * d), ('SEQ', (SG.HALO + SG.C_CH + SG.HALO) * d)]:
            zero_ops.append(('zero_f', 64, max(24, (cnt + 63) // 64), 1, [cnt, lb[key]] + [0] * 30))
    state = {}
    for mode, fill in [('clean', 0.0), ('canary', check.CANARY), ('poison', np.nan)]:
        full = np.full(g.size, fill, np.float32)
        for li, lb in enumerate(g.lay):
            full[lb['P']:lb['P'] + SG.NP * d] = g.posrows[li].ravel()
        state[mode] = full
    hist = [k for k in ar.regions if k[:2] in ('KC', 'VC') or k.startswith('SEQ')]
    for mode in ('canary', 'poison'):
        ok, out, _ = check.guard_run(f'stream encoder init ({mode})', as_plan(zero_ops), ar,
                                     {k: region(state[mode], k) for k in fixed}, hist, poison=mode == 'poison', **kw)
        allok &= ok
        for k in hist:
            region(state[mode], k)[:] = out[k]
    for k in hist:
        region(state['clean'], k)[:] = 0
    nkprev = 0
    for t0 in range(0, T, SG.C_CH):
        n = min(SG.C_CH, T - t0); nk = nkprev + n
        ops = g.clear_tail_ops(n) + SG.chunk_ops(L, g.P, g.lay, n, t0, t0 - nkprev, nk, nkprev) + g.shift_ops(n, nk)
        plan = as_plan(ops)
        logits = {}
        for mode in ('clean', 'canary', 'poison'):
            full = state[mode]
            region(full, 'X')[:n * d] = X[t0:t0 + n].ravel()
            if mode == 'clean':
                got, _ = run(plan, full, 0, g.size, **kw)
                ok = True
            else:
                ok, out, _ = check.guard_run(f'stream encoder chunk t0={t0} n={n} nk={nk} ({mode})', plan, ar,
                                             {k: region(full, k) for k in ar.regions}, [], work,
                                             poison=mode == 'poison', **kw)
                got = full.copy()
                for k in work:
                    region(got, k)[:] = out[k]
            state[mode] = got
            lo = ar.regions['LOG'][0]
            logits[mode] = got[lo:lo + n * (V + 1)].copy()
            allok &= ok
        for mode in ('canary', 'poison'):
            same = np.array_equal(logits[mode], logits['clean'])
            if not same:
                bad = np.flatnonzero(logits[mode].view(np.uint32) != logits['clean'].view(np.uint32))
                nan = int(np.isnan(logits[mode]).sum())
                print(f'     logits ({mode}) differ from the clean run: {bad.size} of {logits[mode].size} floats, {nan} NaN')
            allok &= same
        nkprev = min(nk, SG.LEFT)


which = sys.argv[1] if len(sys.argv) > 1 else 'both'
if which in ('fe', 'both'):
    front_end()
if which in ('enc', 'both'):
    encoder()
print('STREAM GUARD PASS' if allok else 'STREAM GUARD FAIL')
raise SystemExit(0 if allok else 1)
