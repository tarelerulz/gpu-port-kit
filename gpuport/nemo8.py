"""nemo8 full-GPU encoder built only from gpuport helpers (weights, arena, plan, ops, resident server).
Used by the dictation menu (speech-app-type.py --gpu-full) and by tests/tools.
Class Nemo8Encoder has start(), transcribe(pcm_int16) -> text, close(), like the old gpu_encoder2.GpuEncoder2.
The weights are byte-identical to ~/speech-app/port/cache2/weights.bin (checked by tests/nemo8_rebuild.py)."""
import importlib.util
import json
import os
import numpy as np
from .weights import Weights
from .arena import Arena
from .plan import Plan
from . import ops
from .config import WORK_DIR, merged_shader_dir
from .server import ExecServer
from .int8 import read_q8, pack_linear, quant_op, lin_i8, quant_op_km, lin_i8_km

SPEECH_APP = os.path.expanduser('~/speech-app/port')
MODEL = os.path.expanduser('~/speech-app/models/nemo8.tnm')
PROD_W = os.path.join(SPEECH_APP, 'cache2', 'weights.bin')
SHADERS = merged_shader_dir()
ARENA_FLOATS = 24 << 20          # exec_server2 arena (about 10 s of audio fits)


def load_nemo_ref():
    spec = importlib.util.spec_from_file_location('nemo_ref', os.path.join(SPEECH_APP, 'nemo_ref.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def prepare(nr, model=MODEL, int8=False):
    """int8=True: encoder linears and head stored as packed int8 (kind 'i8'); else float32 (kind 'f32')."""
    """Same reads and the same put order as ~/speech-app/port/gpu_encoder2.prepare(), but with kit Weights."""
    Bl = nr.Blob(model); hd = Bl.hd
    d, H, ff, k, nl, sc, V, bits = hd[1], hd[2], hd[3], hd[4], hd[5], hd[6], hd[7], hd[8]
    assert hd[9] == 0 and bits == 8
    def lin(n_, k_):
        if int8:
            return ('i8',) + read_q8(Bl, n_, k_)
        W_, b_ = Bl.qlin(n_, k_, 8)
        return ('f32', W_, b_)
    Bl.align()
    window = Bl.f32(400); fb = Bl.f32(257 * 80).reshape(257, 80)
    c0_w = Bl.f32(sc * 9).reshape(sc, 9); c0_b = Bl.f32(sc)
    f2 = ((((80 + 2 - 3) // 2 + 1) + 2 - 3) // 2 + 1)
    c2_W, c2_b = Bl.qlin(sc, sc * 9, 8); sub_W, sub_b = Bl.qlin(d, sc * f2, 8)
    layers = []
    for _ in range(nl):
        L = {}
        L['n_ff1'] = (Bl.f32(d), Bl.f32(d)); L['ff1_1'] = lin(ff, d); L['ff1_2'] = lin(d, ff)
        L['n_att'] = (Bl.f32(d), Bl.f32(d)); L['qkv'] = lin(3 * d, d); L['out'] = lin(d, d)
        L['pos'] = lin(d, d); L['pbu'] = Bl.f32(d); L['pbv'] = Bl.f32(d)
        L['n_conv'] = (Bl.f32(d), Bl.f32(d)); L['pw1'] = lin(2 * d, d)
        L['dw_w'] = Bl.f32(d * k).reshape(d, k); L['dw_b'] = Bl.f32(d); L['pw2'] = lin(d, d)
        L['n_ff2'] = (Bl.f32(d), Bl.f32(d)); L['ff2_1'] = lin(ff, d); L['ff2_2'] = lin(d, ff)
        L['n_out'] = (Bl.f32(d), Bl.f32(d))
        layers.append(L)
    head_l = lin(V + 1, d)
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
            lw[key] = put_lin(w, L[key])
        for key in ['n_ff1', 'n_att', 'n_conv', 'n_ff2', 'n_out']:
            lw[key] = w.put(np.concatenate(L[key]))
        lw['pbu'] = w.put(L['pbu']); lw['pbv'] = w.put(L['pbv'])
        lw['dw_w'] = w.put(L['dw_w'].T.copy()); lw['dw_b'] = w.put(L['dw_b'])
        lay.append(lw)
    head = put_lin(w, head_l)
    return w, dict(d=d, H=H, ff=ff, k=k, nl=nl, sc=sc, V=V, f2=f2, fe=fe, layers=lay, head=head, tokens=toks, int8=int8)


def put_lin(w, x):
    if x[0] == 'i8':
        return pack_linear(w, x[1], x[2], x[3])
    return w.put_mm(x[1], x[2])


def build(n, L, optimized=True, av4=None):
    """Plan for n samples, same dispatches as gpu_encoder2.build(), written with kit helpers.
    optimized: padded frame stride (no power-of-two matmul row stride) and the kit's conv0 / im2col shaders.
    av4 (2026-11-09): matmuls read activations k-major through a transpose (ops.gemm12 kscratch). Default on for
    optimized plans; GPUPORT_AV4=0 turns it off (old row-major matmuls) for before/after numbers."""
    if av4 is None:
        av4 = os.environ.get('GPUPORT_AV4', '1') != '0'
    av4 = bool(av4) and optimized
    d, H, ff, k, sc, V, f2, fe = L['d'], L['H'], L['ff'], L['k'], L['sc'], L['V'], L['f2'], L['fe']
    V1 = V + 1; nvalid = n // 160; Tm = nvalid + 1
    T2 = (Tm + 2 - 3) // 2 + 1; T3 = (T2 + 2 - 3) // 2 + 1; T = T3; NP = 2 * T - 1
    dh = d // H
    FRS = 516 if optimized else 512          # padded frame row stride (512 was 7x slower in the DFT matmul)
    ar = Arena(limit=ARENA_FLOATS)
    R = {}
    for name, size in [('XIN', n), ('FR', Tm * FRS), ('RE', Tm * 257), ('IM', Tm * 257), ('PW', Tm * 257),
                       ('MEL', Tm * 80), ('C0', T2 * 40 * sc), ('PAT', T3 * 20 * sc * 9), ('C2', T3 * 20 * sc),
                       ('FL', T3 * sc * 20), ('X', T * d), ('HB', T * d), ('H2', T * ff), ('QKV', T * 3 * d),
                       ('P', NP * d), ('PE', NP * d), ('ATT', T * d), ('GL', T * d), ('CV', T * d),
                       ('SC', H * T * T), ('LOG', T * V1)]:
        R[name] = ar.alloc(name, size)
    if av4:
        # k-major scratch for the transposed matmul input: K <= max(d, ff, 9*sc, 20*sc), 252 rows per block
        R['TK'] = ar.alloc('TK', max(d, ff, 9 * sc, 20 * sc) * 252)
    p = Plan()
    XIN, FR, RE, IM, PW, MEL = (R[x] for x in ['XIN', 'FR', 'RE', 'IM', 'PW', 'MEL'])
    if optimized:
        p.dispatch('fe_frame_s', 64, Tm * 512, [Tm, n, XIN, FR, fe['win'], FRS])
    else:
        p.dispatch('fe_frame', 64, Tm * 512, [Tm, n, XIN, FR, fe['win']])
    ops.gemm12(p, FR, Tm, fe['dft_c'], RE, K=512, N=257, sAm=FRS)
    ops.gemm12(p, FR, Tm, fe['dft_s'], IM, K=512, N=257, sAm=FRS)
    p.dispatch('fe_pw', 64, Tm * 257, [Tm, RE, IM, PW])
    ops.gemm12(p, PW, Tm, fe['mel'], MEL, K=257, N=80, sAm=257)
    p.dispatch('fe_lognorm', 64, 80, [nvalid, MEL])
    C0 = R['C0']
    if optimized:
        p.dispatch('fe_conv0_v2', 64, T2 * 40, [Tm, MEL, C0, fe['c0_w'], fe['c0_b'], sc, T2])
    else:
        p.dispatch('fe_conv0', 64, T2 * 40 * sc, [Tm, MEL, C0, fe['c0_w'], fe['c0_b'], sc, T2])
    PAT, C2, FL = R['PAT'], R['C2'], R['FL']
    if optimized:
        p.dispatch('fe_im2col_v2', 64, T3 * 20 * sc, [T3, T2, C0, PAT, sc])
    else:
        groups = (T3 * 20 * sc * 9 + 63) // 64          # 2-D split as in gpu_encoder2 (shader uses gid.x + gid.y*32768)
        p.add('fe_im2col', 64, min(groups, 32768), (groups + 32767) // 32768, [T3, T2, C0, PAT, sc] + [0] * 27)
    ops.gemm12(p, PAT, T3 * 20, fe['c2'], C2, K=sc * 9, N=sc, sAm=sc * 9, epi=3, kscratch=R.get('TK'))
    p.dispatch('fe_flat', 64, T3 * sc * 20, [T3, sc, C2, FL])
    X, HB, H2, QKV, P, PE, ATT, GL, CV, SC, LOG = (R[x] for x in ['X', 'HB', 'H2', 'QKV', 'P', 'PE', 'ATT', 'GL', 'CV', 'SC', 'LOG'])
    ops.gemm12(p, FL, T3, fe['sub'], X, K=sc * 20, N=d, sAm=sc * 20, kscratch=R.get('TK'))
    p.dispatch('pe_gen', 64, NP * d, [T, d, PE])
    JB = (T + 3) // 4; IB = (T + 1) // 2
    i8km = bool(L.get('int8')) and os.environ.get('GPUPORT_I8KM', '1') != '0'   # 2026-11-09: k-major int8 activations
    if L.get('int8'):
        # k-major needs (K/4) * Mp words; Mp <= T + 3
        R['Q'] = ar.alloc('Q', (ff // 4) * (T + 4) if i8km else T * (ff // 4)); R['QS'] = ar.alloc('QS', T)
    def lin(p_, a_, M_, mm_, out_, epi=0):
        if mm_.get('kind') == 'i8':
            if i8km:
                quant_op_km(p_, a_, M_, mm_['k'], mm_['k'], R['Q'], R['QS'])
                lin_i8_km(p_, R['Q'], R['QS'], M_, mm_, out_, epi)
            else:
                quant_op(p_, a_, M_, mm_['k'], mm_['k'], R['Q'], R['QS'])
                lin_i8(p_, R['Q'], R['QS'], M_, mm_, out_, epi)
        else:
            ops.gemm12(p_, a_, M_, mm_, out_, epi=epi, kscratch=R.get('TK'))
    p.marks = [('front_end', 0)]                    # op index where each stage starts (for profiling)
    for li, lw in enumerate(L['layers']):
        p.marks.append((f'layer{li}', len(p.ops)))
        ops.ln(p, T, d, X, HB, lw['n_ff1']); lin(p, HB, T, lw['ff1_1'], H2, epi=2)
        lin(p, H2, T, lw['ff1_2'], HB); ops.add(p, T * d, X, HB, 0.5)
        ops.ln(p, T, d, X, HB, lw['n_att']); lin(p, HB, T, lw['qkv'], QKV)
        lin(p, PE, NP, lw['pos'], P)
        push = [T, d, H, dh, QKV, P, SC, lw['pbu'], lw['pbv']]
        p.dispatch('att_scores5jf', 64, H * T * JB, push + [0] * 23)
        p.dispatch('att_softmax', 64, H * T, push + [ATT] + [0] * 22)
        p.add('att_pv6', 64, max(24, (H * IB * (dh // 4) + 63) // 64), 1, push + [ATT] + [0] * 22)
        lin(p, ATT, T, lw['out'], HB); ops.add(p, T * d, X, HB, 1.0)
        ops.ln(p, T, d, X, HB, lw['n_conv']); lin(p, HB, T, lw['pw1'], H2)
        ops.glu(p, T, d, H2, GL)
        p.add('dw4', 64, max(24, (d // 4 + 63) // 64), T, [T, d, k, GL, CV, lw['dw_w'], lw['dw_b']] + [0] * 25)
        ops.silu(p, T * d, CV, CV)
        lin(p, CV, T, lw['pw2'], HB); ops.add(p, T * d, X, HB, 1.0)
        ops.ln(p, T, d, X, HB, lw['n_ff2']); lin(p, HB, T, lw['ff2_1'], H2, epi=2)
        lin(p, H2, T, lw['ff2_2'], HB); ops.add(p, T * d, X, HB, 0.5)
        ops.ln(p, T, d, X, X, lw['n_out'])
    p.marks.append(('head', len(p.ops)))
    lin(p, X, T, L['head'], LOG)
    return p, ar, R, T


def collapse(logits, toks, V):
    am = logits.argmax(1); out, prev = [], -1
    for a in am:
        if a != prev and a != V:
            out.append(toks[int(a)])
        prev = a
    s = ''.join(out)
    return s[1:] if s.startswith(' ') else s


class Nemo8Encoder:
    """Resident nemo8 encoder. First start() prepares the weights once (cached in WORK_DIR)."""
    def __init__(self, work=WORK_DIR):
        self.work = work
        self.L = None
        self.srv = None
        self.last_ms = None

    def _layout(self):
        if self.L is not None:
            return self.L
        os.makedirs(self.work, exist_ok=True)
        wprefix = os.path.join(self.work, 'nemo8_kit')
        lpath = wprefix + '.json'
        if not (os.path.exists(wprefix + '.bin') and os.path.exists(lpath)):
            w, L = prepare(load_nemo_ref())
            w.save(wprefix)
            json.dump(L, open(lpath, 'w'))
        self.L = json.load(open(lpath))
        self.weights = wprefix + '.bin'
        return self.L

    def start(self):
        L = self._layout()
        self.srv = ExecServer(self.weights, SHADERS).start()
        warm = np.zeros(16000, np.float32)                    # warm-up: pipelines and buffers
        plan, ar, R, T = build(len(warm), L)
        self.srv.run(plan, warm, R['LOG'], T * (L['V'] + 1))
        return self

    def transcribe(self, pcm):
        pcm = np.asarray(pcm, np.int16)
        if len(pcm) < 1600:                                   # under 0.1 s: nothing to say
            return ''
        L = self._layout()
        plan, ar, R, T = build(len(pcm), L)
        out, self.last_ms = self.srv.run(plan, pcm.astype(np.float32) / 32768.0, R['LOG'], T * (L['V'] + 1))
        return collapse(out.reshape(T, L['V'] + 1), L['tokens'], L['V'])

    def close(self):
        if self.srv is not None:
            self.srv.stop()
