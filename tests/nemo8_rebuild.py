"""Rebuild the nemo8 full-GPU encoder with the kit's helpers only (Weights, Arena, Plan, ops), then check it:
  1. weight blob: byte-identical to the production cache (~/oido/port/cache2/weights.bin)
  2. logits on tarel40 (10 s) and speak (5 s) vs the C reference (~/oido/port/*_logits.bin)
  3. collapsed token text vs the same reference
  4. logits vs the production GPU encoder (~/oido/port/gpu_encoder2.py): must be identical
The C reference is int8; the float GPU path agrees with it on ~98.8% of argmax frames (earlier notes), so the
C check uses a 98% floor. The production comparison is the exact one.
The plan is run through tools/exec_big (exec with a 24M-float arena, 48M-float weights), since a 10 s clip
needs 18.7M arena floats.
usage: python3 ~/gpu-port-kit/tests/nemo8_rebuild.py"""
import importlib.util
import os
import sys
import time
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.expanduser('~/oido/port'))
from gpuport import Arena, Weights, Plan, ops
from gpuport.run import run
from gpuport.config import WORK_DIR

OIDO = os.path.expanduser('~/oido/port')
MODEL = os.path.expanduser('~/oido/models/nemo8.tnm')
PROD_W = os.path.join(OIDO, 'cache2', 'weights.bin')
EXEC_BIG = os.path.join(KIT, 'tools', 'exec_big')
SHADERS = os.path.expanduser('~/oido/port/gpu')
ARENA_FLOATS = 24 << 20


def load_nemo_ref():
    spec = importlib.util.spec_from_file_location('nemo_ref', os.path.join(OIDO, 'nemo_ref.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def prepare(nr, model=MODEL):
    """Same reads and the same put order as ~/oido/port/gpu_encoder2.prepare(), but with kit Weights."""
    Bl = nr.Blob(model); hd = Bl.hd
    d, H, ff, k, nl, sc, V, bits = hd[1], hd[2], hd[3], hd[4], hd[5], hd[6], hd[7], hd[8]
    assert hd[9] == 0 and bits == 8
    Bl.align()
    window = Bl.f32(400); fb = Bl.f32(257 * 80).reshape(257, 80)
    c0_w = Bl.f32(sc * 9).reshape(sc, 9); c0_b = Bl.f32(sc)
    f2 = ((((80 + 2 - 3) // 2 + 1) + 2 - 3) // 2 + 1)
    c2_W, c2_b = Bl.qlin(sc, sc * 9, 8); sub_W, sub_b = Bl.qlin(d, sc * f2, 8)
    layers = []
    for _ in range(nl):
        L = {}
        L['n_ff1'] = (Bl.f32(d), Bl.f32(d)); L['ff1_1'] = Bl.qlin(ff, d, 8); L['ff1_2'] = Bl.qlin(d, ff, 8)
        L['n_att'] = (Bl.f32(d), Bl.f32(d)); L['qkv'] = Bl.qlin(3 * d, d, 8); L['out'] = Bl.qlin(d, d, 8)
        L['pos'] = Bl.qlin(d, d, 8); L['pbu'] = Bl.f32(d); L['pbv'] = Bl.f32(d)
        L['n_conv'] = (Bl.f32(d), Bl.f32(d)); L['pw1'] = Bl.qlin(2 * d, d, 8)
        L['dw_w'] = Bl.f32(d * k).reshape(d, k); L['dw_b'] = Bl.f32(d); L['pw2'] = Bl.qlin(d, d, 8)
        L['n_ff2'] = (Bl.f32(d), Bl.f32(d)); L['ff2_1'] = Bl.qlin(ff, d, 8); L['ff2_2'] = Bl.qlin(d, ff, 8)
        L['n_out'] = (Bl.f32(d), Bl.f32(d))
        layers.append(L)
    head_W, head_b = Bl.qlin(V + 1, d, 8)
    Bl.align(); toks = []; p = Bl.p
    for _ in range(V):
        n = Bl.b[p]; toks.append(Bl.b[p + 1:p + 1 + n].decode('utf-8', 'replace')); p += 1 + n

    w = Weights()
    kk = np.arange(512)[None, :]; nn = np.arange(257)[:, None]
    cosT = np.cos(2 * np.pi * nn * kk / 512.0).astype(np.float32)
    sinT = np.sin(2 * np.pi * nn * kk / 512.0).astype(np.float32)
    fe = dict(win=w.put(window), dft_c=w.put_mm(cosT, np.zeros(257)), dft_s=w.put_mm(sinT, np.zeros(257)),
              mel=w.put_mm(fb.T.astype(np.float32), np.zeros(80)),
              c0_w=w.put(c0_w), c0_b=w.put(c0_b), c2=w.put_mm(c2_W, c2_b),
              sub=w.put_mm(sub_W * np.sqrt(d), sub_b * np.sqrt(d)))
    lay = []
    for L in layers:
        lw = {}
        for key in ['ff1_1', 'ff1_2', 'ff2_1', 'ff2_2', 'qkv', 'out', 'pos', 'pw1', 'pw2']:
            lw[key] = w.put_mm(*L[key])
        for key in ['n_ff1', 'n_att', 'n_conv', 'n_ff2', 'n_out']:
            lw[key] = w.put(np.concatenate(L[key]))
        lw['pbu'] = w.put(L['pbu']); lw['pbv'] = w.put(L['pbv'])
        lw['dw_w'] = w.put(L['dw_w'].T.copy()); lw['dw_b'] = w.put(L['dw_b'])
        lay.append(lw)
    head = w.put_mm(head_W, head_b)
    return w, dict(d=d, H=H, ff=ff, k=k, nl=nl, sc=sc, V=V, f2=f2, fe=fe, layers=lay, head=head, tokens=toks)


def build(n, L):
    """Plan for n samples, same dispatches as gpu_encoder2.build(), written with kit helpers."""
    d, H, ff, k, sc, V, f2, fe = L['d'], L['H'], L['ff'], L['k'], L['sc'], L['V'], L['f2'], L['fe']
    V1 = V + 1; nvalid = n // 160; Tm = nvalid + 1
    T2 = (Tm + 2 - 3) // 2 + 1; T3 = (T2 + 2 - 3) // 2 + 1; T = T3; NP = 2 * T - 1
    dh = d // H
    ar = Arena(limit=ARENA_FLOATS)
    R = {}
    for name, size in [('XIN', n), ('FR', Tm * 512), ('RE', Tm * 257), ('IM', Tm * 257), ('PW', Tm * 257),
                       ('MEL', Tm * 80), ('C0', T2 * 40 * sc), ('PAT', T3 * 20 * sc * 9), ('C2', T3 * 20 * sc),
                       ('FL', T3 * sc * 20), ('X', T * d), ('HB', T * d), ('H2', T * ff), ('QKV', T * 3 * d),
                       ('P', NP * d), ('PE', NP * d), ('ATT', T * d), ('GL', T * d), ('CV', T * d),
                       ('SC', H * T * T), ('LOG', T * V1)]:
        R[name] = ar.alloc(name, size)
    p = Plan()
    XIN, FR, RE, IM, PW, MEL = (R[x] for x in ['XIN', 'FR', 'RE', 'IM', 'PW', 'MEL'])
    p.dispatch('fe_frame', 64, Tm * 512, [Tm, n, XIN, FR, fe['win']])
    ops.gemm12(p, FR, Tm, fe['dft_c'], RE, K=512, N=257, sAm=512)
    ops.gemm12(p, FR, Tm, fe['dft_s'], IM, K=512, N=257, sAm=512)
    p.dispatch('fe_pw', 64, Tm * 257, [Tm, RE, IM, PW])
    ops.gemm12(p, PW, Tm, fe['mel'], MEL, K=257, N=80, sAm=257)
    p.dispatch('fe_lognorm', 64, 80, [nvalid, MEL])
    C0 = R['C0']
    p.dispatch('fe_conv0', 64, T2 * 40 * sc, [Tm, MEL, C0, fe['c0_w'], fe['c0_b'], sc, T2])
    PAT, C2, FL = R['PAT'], R['C2'], R['FL']
    groups = (T3 * 20 * sc * 9 + 63) // 64          # 2-D split as in gpu_encoder2 (shader uses gid.x + gid.y*32768)
    p.add('fe_im2col', 64, min(groups, 32768), (groups + 32767) // 32768, [T3, T2, C0, PAT, sc] + [0] * 27)
    ops.gemm12(p, PAT, T3 * 20, fe['c2'], C2, K=sc * 9, N=sc, sAm=sc * 9, epi=3)
    p.dispatch('fe_flat', 64, T3 * sc * 20, [T3, sc, C2, FL])
    X, HB, H2, QKV, P, PE, ATT, GL, CV, SC, LOG = (R[x] for x in ['X', 'HB', 'H2', 'QKV', 'P', 'PE', 'ATT', 'GL', 'CV', 'SC', 'LOG'])
    ops.gemm12(p, FL, T3, fe['sub'], X, K=sc * 20, N=d, sAm=sc * 20)
    p.dispatch('pe_gen', 64, NP * d, [T, d, PE])
    JB = (T + 3) // 4; IB = (T + 1) // 2
    for lw in L['layers']:
        ops.ln(p, T, d, X, HB, lw['n_ff1']); ops.gemm12(p, HB, T, lw['ff1_1'], H2, epi=2)
        ops.gemm12(p, H2, T, lw['ff1_2'], HB); ops.add(p, T * d, X, HB, 0.5)
        ops.ln(p, T, d, X, HB, lw['n_att']); ops.gemm12(p, HB, T, lw['qkv'], QKV)
        ops.gemm12(p, PE, NP, lw['pos'], P)
        push = [T, d, H, dh, QKV, P, SC, lw['pbu'], lw['pbv']]
        p.dispatch('att_scores5', 64, H * T * JB, push + [0] * 23)
        p.dispatch('att_softmax', 64, H * T, push + [ATT] + [0] * 22)
        p.add('att_pv6', 64, max(24, (H * IB * (dh // 4) + 63) // 64), 1, push + [ATT] + [0] * 22)
        ops.gemm12(p, ATT, T, lw['out'], HB); ops.add(p, T * d, X, HB, 1.0)
        ops.ln(p, T, d, X, HB, lw['n_conv']); ops.gemm12(p, HB, T, lw['pw1'], H2)
        ops.glu(p, T, d, H2, GL)
        p.add('dw4', 64, max(24, (d // 4 + 63) // 64), T, [T, d, k, GL, CV, lw['dw_w'], lw['dw_b']] + [0] * 25)
        ops.silu(p, T * d, CV, CV)
        ops.gemm12(p, CV, T, lw['pw2'], HB); ops.add(p, T * d, X, HB, 1.0)
        ops.ln(p, T, d, X, HB, lw['n_ff2']); ops.gemm12(p, HB, T, lw['ff2_1'], H2, epi=2)
        ops.gemm12(p, H2, T, lw['ff2_2'], HB); ops.add(p, T * d, X, HB, 0.5)
        ops.ln(p, T, d, X, X, lw['n_out'])
    ops.gemm12(p, X, T, L['head'], LOG)
    return p, ar, R, T


def collapse(logits, toks, V):
    am = logits.argmax(1); out, prev = [], -1
    for a in am:
        if a != prev and a != V:
            out.append(toks[int(a)])
        prev = a
    s = ''.join(out)
    return s[1:] if s.startswith(' ') else s


def main():
    nr = load_nemo_ref()
    w, L = prepare(nr)
    os.makedirs(WORK_DIR, exist_ok=True)
    wpath = w.save(os.path.join(WORK_DIR, 'nemo8_kit'))
    mine = np.fromfile(wpath, '<f4'); prod = np.fromfile(PROD_W, '<f4')
    same = mine.size == prod.size and np.array_equal(mine, prod)
    print(f'weights: kit {mine.size} floats, production {prod.size}; byte-identical: {same}')
    ok_all = same
    V = L['V']; toks = L['tokens']
    for clip, ref_file in [('tarel40', 'tarel40_logits.bin'), ('speak', 'speak_logits.bin')]:
        pcm = np.fromfile(os.path.join(OIDO, clip + '.pcm'), '<i2')
        plan, ar, R, T = build(len(pcm), L)
        arena_in = pcm.astype(np.float32) / 32768.0
        t0 = time.time()
        out, ms = run(plan, arena_in, R['LOG'], T * (V + 1), weights=wpath, shader_dir=SHADERS,
                      exec_bin=EXEC_BIG, arena_floats=ARENA_FLOATS)
        wall = time.time() - t0
        logits = out.reshape(T, V + 1)
        ref = np.fromfile(os.path.join(OIDO, ref_file), '<f4').reshape(-1, V + 1)
        ok_shape = ref.shape[0] == T
        if not ok_shape:
            print(f'{clip}: frames {T} vs reference {ref.shape[0]}')
        d = np.abs(logits - ref[:T]).max()
        agree = (logits.argmax(1) == ref[:T].argmax(1)).mean()
        text_k = collapse(logits, toks, V); text_r = collapse(ref[:T], toks, V)
        same_text = text_k == text_r
        print(f'{clip}: {len(pcm)/16000:.0f} s, T={T}, arena {ar.used()/1e6:.2f}M floats, gpu {ms:.0f} ms, wall {wall:.2f} s')
        print(f'   logits max diff {d:.3g}, argmax agree {agree*100:.2f}%, tokens identical: {same_text}')
        print(f'   text: {text_k!r}')
        if not same_text:
            print(f'   reference: {text_r!r}')
        ok_all &= ok_shape and same_text and agree > 0.98
        try:
            from gpu_encoder2 import GpuEncoder2
            if not hasattr(main, 'enc'):
                main.enc = GpuEncoder2()
            prod = main.enc.logits(pcm)
            pdiff = np.abs(logits - prod).max()
            same_prod = pdiff == 0.0
            print(f'   vs production GPU encoder: max diff {pdiff:.3g}, identical: {same_prod}')
            ok_all &= same_prod
        except Exception as e:   # production path not available: say so, and count it as a failure
            print(f'   production comparison FAILED to run: {e}')
            ok_all = False
    print('NEMO8 REBUILD MATCHES' if ok_all else 'NEMO8 REBUILD DIFFERS (see above)')
    return 0 if ok_all else 1


if __name__ == '__main__':
    raise SystemExit(main())
