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


FE2_MIN_T = 48        # frames (1.9 s): from here the matmul front end is faster (measured 2026-11-12)
POS_TMAX = 304        # frames covered by the cached position tables (12.1 s); longer clips compute them per run


def prepare(nr, model=MODEL, int8=False, pos_cache=True):
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
    pos_tab = None
    if pos_cache and not int8:          # appended after everything else: the production blob stays a prefix
        offs = []
        for L_ in layers:
            tab, pitch = ops.pos_table_km(L_['pos'][1], L_['pos'][2], POS_TMAX)
            offs.append(w.put(tab))
        w.put(np.zeros(16, np.float32))
        pos_tab = dict(Tmax=POS_TMAX, pitch=pitch, offs=offs)
    fe2 = fe3 = None
    if pos_cache and not int8:          # front-end convs as matmuls (2026-11-12): permuted weight copies, appended
        # conv0 for plane dt2 (conv0 row 2*t3 + dt2 - 1): K index b*8 + c8, c8 = 2*dt2 + a + 1 (a, b = conv0 taps dt, df)
        c0p = []
        for dt2 in range(3):
            Wp = np.zeros((sc, 24), np.float32)
            for a_ in range(3):
                for b_ in range(3):
                    Wp[:, b_ * 8 + 2 * dt2 + a_ + 1] = c0_w[:, a_ * 3 + b_]
            c0p.append(w.put_mm(Wp, c0_b))
        # conv2: K index df*3*sc + dt*sc + ci (file order is ci*9 + dt*3 + df)
        W2 = np.asarray(c2_W, np.float32).reshape(sc, sc, 3, 3).transpose(0, 3, 2, 1).reshape(sc, 9 * sc)
        fe2 = dict(c0p=c0p, c2p=w.put_mm(W2, c2_b))
        w.put(np.zeros(16, np.float32))
        # two-stage DFT (2026-11-12): 512 = 16 x 32, sample i = 32 n1 + n2 (n1 < 13: frames are 400 long), bin k = k1 + 16 k2
        n1_ = np.arange(13)[None, :]; k1_ = np.arange(16)[:, None]
        W1 = np.concatenate([np.cos(2 * np.pi * n1_ * k1_ / 16), -np.sin(2 * np.pi * n1_ * k1_ / 16)]).astype(np.float32)   # [32][13]
        n2_ = np.arange(32); k2_ = np.arange(17)
        cc = np.cos(2 * np.pi * np.outer(k2_, n2_) / 32); ss = np.sin(2 * np.pi * np.outer(k2_, n2_) / 32)
        W2d = np.zeros((34, 64), np.float32)             # rows k2 * 2 + {re, im}, columns n2 * 2 + {Yr, Yi}
        W2d[0::2, 0::2] = cc; W2d[0::2, 1::2] = ss; W2d[1::2, 0::2] = -ss; W2d[1::2, 1::2] = cc
        ph = 2 * np.pi * np.outer(np.arange(32), np.arange(16)) / 512
        tw = np.stack([np.cos(ph), -np.sin(ph)], -1).astype(np.float32).reshape(-1)
        fe3 = dict(w1=w.put_mm(W1, np.zeros(32)), w2=w.put_mm(W2d, np.zeros(34)), tw=w.put(tw))
        w.put(np.zeros(16, np.float32))
    asm = None
    if pos_cache and not int8 and os.environ.get('GPUPORT_ASM') == '1':     # QPU kernel weights (f16 pairs), appended
        asm = [{k_: put_asm(w, L_[k_]) for k_ in ASM_KEYS} for L_ in layers]
        if fe2:
            asm.append(dict(c2=put_asm(w, ('f32', W2, c2_b)), sub=put_asm(w, ('f32', sub_W * np.sqrt(d), sub_b * np.sqrt(d)))))   # last entry: conv2 (K = 9 sc, relu) and the subsampling matmul
        w.put(np.zeros(16, np.float32))
    pos_asm = None
    if asm is not None and pos_tab is not None:     # static position tables as f16 tile streams, four column phases (window starts are multiples of 4)
        dh_ = d // H; pos_asm = []
        for L_ in layers:
            tab, pitch_ = ops.pos_table_km(L_['pos'][1], L_['pos'][2], POS_TMAX)
            per = []
            for phi in (0, 4, 8, 12):
                row = []
                for h in range(H):
                    Wm = tab[h * dh_:(h + 1) * dh_, phi:].T
                    n16 = (Wm.shape[0] + 15) // 16 * 16
                    Wp = np.zeros((n16, dh_), np.float32); Wp[:Wm.shape[0]] = Wm
                    row.append(w.put(asm_stream(Wp, np.zeros(n16, np.float32)).view(np.float32)))
                per.append(row)
            pos_asm.append(per)
        w.put(np.zeros(16, np.float32))
    return w, dict(d=d, H=H, ff=ff, k=k, nl=nl, sc=sc, V=V, f2=f2, fe=fe, layers=lay, head=head, tokens=toks, int8=int8, pos_tab=pos_tab, fe2=fe2, fe3=fe3, asm=asm, pos_asm=pos_asm)


ASM_KEYS = ('ff1_1', 'ff2_1', 'ff1_2', 'ff2_2', 'out', 'pw2', 'qkv', 'pw1')   # matmuls run by the hand-written QPU kernel (GPUPORT_ASM=1, exec_hybrid runner)


def asm_stream(W_, b_):
    """uint32 stream for the QPU kernel from W_ [N][K] (N % 16 == 0) and bias [N]: per tile of 16 outputs 16 bias words + K * 8 f16-pair words."""
    N_, K_ = W_.shape
    h = W_.astype(np.float16).view(np.uint16).reshape(N_ // 16, 8, 2, K_)
    words = np.transpose(h[:, :, 0, :].astype(np.uint32) | (h[:, :, 1, :].astype(np.uint32) << 16), (0, 2, 1))
    return np.concatenate([np.asarray(b_, np.float32).reshape(N_ // 16, 16).view(np.uint32), words.reshape(N_ // 16, K_ * 8)], axis=1).reshape(-1).astype(np.uint32)


def put_asm(w, x):
    """Weights of one linear layer for the QPU kernel: tiles of 16 outputs, 16 bias words + K * 8 f16-pair words each (qpu_mm.pack_weights)."""
    W_, b_ = np.asarray(x[1], np.float32), np.asarray(x[2], np.float32)
    N_, K_ = W_.shape
    h = W_.astype(np.float16).view(np.uint16).reshape(N_ // 16, 8, 2, K_)
    words = np.transpose(h[:, :, 0, :].astype(np.uint32) | (h[:, :, 1, :].astype(np.uint32) << 16), (0, 2, 1))
    st = np.concatenate([b_.reshape(N_ // 16, 16).view(np.uint32), words.reshape(N_ // 16, K_ * 8)], axis=1).reshape(-1).astype(np.uint32)
    return dict(wo=w.put(st.view(np.float32)), k=K_, ntiles=N_ // 16, tile_words=16 + 8 * K_)


def put_lin(w, x):
    if x[0] == 'i8':
        return pack_linear(w, x[1], x[2], x[3])
    return w.put_mm(x[1], x[2])


def _fe_old(p, R, L, fe, optimized, km, Tm, T2, T3, Tp, sc, d, MEL, X):
    """conv0 shader + im2col + conv2 matmul + flatten + subsampling matmul (the front end before fe2)."""
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
    if km:
        ops.gemm_to_km(p, FL, T3, fe['sub'], X, Tp, R['TK'])
    else:
        ops.gemm12(p, FL, T3, fe['sub'], X, K=sc * 20, N=d, sAm=sc * 20, kscratch=R.get('TK'))


def d_ok(L):
    return L['d'] % 4 == 0 and (L['d'] // L['H']) % 4 == 0


def build(n, L, optimized=True, av4=None, att_km=None):
    """See _build. The matmul attention needs about 6% more arena, so the longest clip that fits drops from 13 s
    to 12 s; a clip that only fits without it is built with the old attention shaders."""
    try:
        return _build(n, L, optimized, av4, att_km)
    except MemoryError:
        if att_km is False:
            raise
        return _build(n, L, optimized, av4, False)


def _build(n, L, optimized=True, av4=None, att_km=None):
    """Plan for n samples, same dispatches as gpu_encoder2.build(), written with kit helpers.
    optimized: padded frame stride (no power-of-two matmul row stride) and the kit's conv0 / im2col shaders.
    av4 (2026-11-09): matmuls read activations k-major through a transpose (ops.gemm12 kscratch). Default on for
    optimized plans; GPUPORT_AV4=0 turns it off (old row-major matmuls) for before/after numbers."""
    if av4 is None:
        av4 = os.environ.get('GPUPORT_AV4', '1') != '0'
    av4 = bool(av4) and optimized
    # att_km (2026-11-12): attention as matmuls on the k-major kernel (ops.attention_km): 53 -> 22 ms per layer at
    # T=251. Same math, different summation order, so logits differ in the last bits (not bit-identical to the
    # production encoder). Default on with av4, float32 only; GPUPORT_ATT_KM=0 = old attention shaders.
    if att_km is None:
        att_km = os.environ.get('GPUPORT_ATT_KM', '1') != '0'
    att_km = bool(att_km) and av4 and not L.get('int8') and d_ok(L)
    pos_cache = os.environ.get('GPUPORT_POS_CACHE', '1') != '0'     # 2026-11-12: position tables from the weight blob
    Tq = (((n // 160 + 1) + 2 - 3) // 2 + 1 + 2 - 3) // 2 + 1
    pos_cache = bool(att_km and pos_cache and L.get('pos_tab') and Tq <= L['pos_tab']['Tmax'])
    d, H, ff, k, sc, V, f2, fe = L['d'], L['H'], L['ff'], L['k'], L['sc'], L['V'], L['f2'], L['fe']
    V1 = V + 1; nvalid = n // 160; Tm = nvalid + 1
    T2 = (Tm + 2 - 3) // 2 + 1; T3 = (T2 + 2 - 3) // 2 + 1; T = T3; NP = 2 * T - 1
    dh = d // H
    Tp = (T + 3) // 4 * 4                    # k-major row pitch (frames padded to a vec4)
    # km (2026-11-12): layer activations kept k-major ([c][t]), so no transposes inside the layers; ln, GLU and the
    # depthwise conv have k-major shaders. Needs the matmul attention. GPUPORT_KM=0 = row-major layers.
    km = att_km and os.environ.get('GPUPORT_KM', '1') != '0'
    # fe2 (2026-11-12): conv0 and conv2 as matmuls on regrouped data, no patches, output straight into the k-major
    # input of the subsampling matmul. GPUPORT_FE2=0 = conv0 shader + im2col.
    # Under about 1.8 s (48 frames) the 20 per-column conv2 matmuls cost more than they save, so short clips keep im2col.
    fe2 = bool(km and L.get('fe2') and os.environ.get('GPUPORT_FE2', '1') != '0' and Tq >= FE2_MIN_T)
    # fe3 (2026-11-12): the DFT as two small matmuls (16 x 32) instead of two dense 512 x 257 ones; GPUPORT_FE3=0 = dense DFT
    fe3 = bool(fe2 and L.get('fe3') and os.environ.get('GPUPORT_FE3', '1') != '0')
    use_asm = bool(L.get('asm')) and os.environ.get('GPUPORT_ASM') == '1' and 1 <= T <= 1024 and km and (os.environ.get('GPUPORT_ASM_GATE') == 'off' or not (129 <= T <= 136 or 257 <= T <= 328))
    att_qpu = bool(use_asm and att_km and os.environ.get('GPUPORT_ASM_ATT', '1') == '1')       # attention scores and A x V on the QPU kernel (packed f16 K / V)
    Tk = Tp if km else T                     # frames per row of the layer regions
    FRS = 516 if optimized else 512          # padded frame row stride (512 was 7x slower in the DFT matmul)
    ar = Arena(limit=ARENA_FLOATS)
    R = {}
    for name, size in [('XIN', n), ('FR', Tm * FRS), ('RE', Tm * 257), ('IM', Tm * 257), ('PW', Tm * 257),
                       ('MEL', Tm * 80), ('C0', T2 * 40 * sc), ('PAT', T3 * 20 * sc * 9), ('C2', T3 * 20 * sc),
                       ('FL', T3 * sc * 20), ('X', Tk * d), ('HB', Tk * d), ('H2', Tk * (max(ff, 2 * d) if km else ff)),
                       ('QKV', T * 3 * d), ('P', NP * d), ('PE', NP * d), ('ATT', Tk * d),
                       ('GL', (Tk + 32) * d if km else T * d), ('CV', Tk * d),
                       ('SC', H * T * T), ('LOG', T * V1)]:
        if att_km and name in ('P', 'SC') or pos_cache and name == 'PE':   # the matmul attention has its own regions
            continue
        if fe2 and name in ('C0', 'PAT', 'C2', 'FL'):
            continue
        if fe3 and name in ('FR', 'RE', 'IM'):
            continue
        R[name] = ar.alloc(name, size)
    if av4:
        # k-major scratch for the transposed matmul input: K <= max(d, ff, 9*sc, 20*sc), 252 rows per block
        R['TK'] = ar.alloc('TK', max(d, ff, 9 * sc, 20 * sc) * 252)
    if fe2:
        for name, size in [('Z', 82 * 8 * Tp), ('PL', 42 * 3 * sc * Tp), ('FLK', 20 * sc * Tp)]:
            R[name] = ar.alloc(name, size)
    if att_km:
        for name, size in ops.att_km_regions(T, d, H, pos=not pos_cache, qpu=att_qpu):
            R[name] = ar.alloc(name, size)
    p = Plan()
    XIN, FR, RE, IM, PW, MEL = (R.get(x) for x in ['XIN', 'FR', 'RE', 'IM', 'PW', 'MEL'])
    if fe3:
        # stage 1: M = Tm * 32 rows (t, n2), K = 13 (n1), N = 32 (re / im of k1); stage 2: M = Tm * 16 rows (t, k1), K = 64, N = 34
        Mp1, Mp2 = Tm * 32, Tm * 16
        sz = 2464 * Tm + 3 * 64
        assert sz <= 42 * 3 * sc * Tp, 'PL too small for the DFT scratch'
        Y_, A2_, A1_ = R['PL'], R['PL'] + 32 * Mp1 + 64, R['PL'] + 32 * Mp1 + 64 + 32 * Mp2 + 64    # O (stage-2 out) reuses Y_
        f3 = L['fe3']
        p.dispatch('fe_frame_k', 256, Tm * 416, [Tm, n, XIN, A1_, fe['win'], Mp1])
        ops.gemm_km1(p, A1_, Mp1, Mp1, f3['w1'], Y_, Mp1)
        p.dispatch('fe_tw', 256, 32 * Tm * 16, [Tm, Mp1, Mp2, Y_, A2_, f3['tw']])
        ops.gemm_km1(p, A2_, Mp2, Mp2, f3['w2'], Y_, Mp2)
        p.dispatch('fe_pw2', 256, Tm * 257, [Tm, Y_, Mp2, PW])
    else:
        if optimized:
            p.dispatch('fe_frame_s', 64, Tm * 512, [Tm, n, XIN, FR, fe['win'], FRS])
        else:
            p.dispatch('fe_frame', 64, Tm * 512, [Tm, n, XIN, FR, fe['win']])
        ops.gemm12(p, FR, Tm, fe['dft_c'], RE, K=512, N=257, sAm=FRS, kscratch=R.get('TK'))
        ops.gemm12(p, FR, Tm, fe['dft_s'], IM, K=512, N=257, sAm=FRS, kscratch=R.get('TK'))
        p.dispatch('fe_pw', 64, Tm * 257, [Tm, RE, IM, PW])
    ops.gemm12(p, PW, Tm, fe['mel'], MEL, K=257, N=80, sAm=257, kscratch=R.get('TK'))
    p.dispatch('fe_lognorm', 64, 80, [nvalid, MEL])
    X, HB, H2, QKV, P, PE, ATT, GL, CV, SC, LOG = (R.get(x) for x in ['X', 'HB', 'H2', 'QKV', 'P', 'PE', 'ATT', 'GL', 'CV', 'SC', 'LOG'])
    if fe2:
        Z, PL, FLK = R['Z'], R['PL'], R['FLK']
        p.dispatch('fe_phase', 256, 82 * 8 * Tp, [Tm, Tp, MEL, Z])
        if os.environ.get('GPUPORT_BATCH', '1') != '0':      # 40 column positions in one dispatch per plane (2026-11-12)
            for dt2 in range(3):
                ops.gemm_km_batch(p, Z, T3, Tp, L['fe2']['c0p'][dt2], PL + (3 + dt2) * sc * Tp, Tp, 40, 16 * Tp, 3 * sc * Tp, epi=3)
        else:
          for f0 in range(40):
            for dt2 in range(3):
                ops.gemm_km(p, Z + 2 * f0 * 8 * Tp, T3, Tp, L['fe2']['c0p'][dt2], PL + ((f0 + 1) * 3 + dt2) * sc * Tp, Tp, epi=3)
        p.dispatch('fe_c0fix', 64, 42 * 3 * sc, [T3, T2, sc, Tp, PL])
        if use_asm and os.environ.get('GPUPORT_ASM_CONV', '1') == '1':     # conv2: 20 QPU jobs (one per column position), relu epilogue
            for f in range(20):
                ops.asm_mm(p, PL + 2 * f * 3 * sc * Tp, T3, Tp, L['asm'][-1]['c2'], FLK + f * Tp, 20 * Tp, relu=True)
        elif os.environ.get('GPUPORT_BATCH', '1') != '0':
            ops.gemm_km_batch(p, PL, T3, Tp, L['fe2']['c2p'], FLK, 20 * Tp, 20, 6 * sc * Tp, Tp, epi=3)
        else:
          for f in range(20):
            ops.gemm_km(p, PL + 2 * f * 3 * sc * Tp, T3, Tp, L['fe2']['c2p'], FLK + f * Tp, 20 * Tp, epi=3)
        if use_asm and os.environ.get('GPUPORT_ASM_CONV', '1') == '1':
            ops.asm_mm(p, FLK, T3, Tp, L['asm'][-1]['sub'], X, Tp)
        else:
            ops.gemm_km(p, FLK, T3, Tp, fe['sub'], X, Tp)
    else:
        _fe_old(p, R, L, fe, optimized, km, Tm, T2, T3, Tp, sc, d, MEL, X)
    if not pos_cache:
        p.dispatch('pe_gen', 64, NP * d, [T, d, PE])
    JB = (T + 3) // 4; IB = (T + 1) // 2
    i8km = bool(L.get('int8')) and os.environ.get('GPUPORT_I8KM', '1') != '0'   # 2026-11-09: k-major int8 activations
    if L.get('int8'):
        # k-major needs (K/4) * Mp words; Mp <= T + 3
        R['Q'] = ar.alloc('Q', (ff // 4) * (T + 4) if i8km else T * (ff // 4)); R['QS'] = ar.alloc('QS', NP)   # one scale per row; the pos projection has NP = 2T-1 rows (was T: overran, found by the guard 2026-11-12)
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
    def ASMFF(lw_, key_, li_):        # FF first matmul (+ bias): QPU kernel, then silu on the Vulkan side; else the Mesa kernel
        if use_asm:
            exp_ = os.environ.get('GPUPORT_ASM_EXP')        # timing experiments only: 'none' = no matmul, 'tiny' = one tile
            a_ = dict(L['asm'][li_][key_])
            if exp_ == 'tiny':
                a_['ntiles'] = 1
            if exp_ != 'none':
                ops.asm_mm(p, HB, T, Tp, a_, H2, Tp)
            ops.silu(p, lw_[key_]['n'] * Tp, H2, H2)
        else:
            ops.gemm_km(p, HB, T, Tp, lw_[key_], H2, Tp, epi=2)
    def ASMROW(lw_, key_, li_, out_, ncols_):    # matmul with a row-major [T][ncols] output (qkv)
        if use_asm and os.environ.get('GPUPORT_ASM_ROW', '1') == '1':
            ops.asm_mm_rm(p, HB, T, Tp, L['asm'][li_][key_], out_, ncols_)
        else:
            ops.gemm_km(p, HB, T, Tp, lw_[key_], out_)
    def ASMPLAIN(lw_, key_, li_, out_):          # k-major output, bias only (pw1)
        if use_asm and os.environ.get('GPUPORT_ASM_ROW', '1') == '1':
            ops.asm_mm(p, HB, T, Tp, L['asm'][li_][key_], out_, Tp)
        else:
            ops.gemm_km(p, HB, T, Tp, lw_[key_], out_, Tp)
    def ASMRS(lw_, key_, li_, src_, scale_):   # matmul with the residual folded in (X += scale * result)
        if use_asm and os.environ.get('GPUPORT_ASM_RS', '1') == '1':
            ops.asm_mm_rs(p, src_, T, Tp, L['asm'][li_][key_], X, Tp, scale_)
        else:
            ops.gemm_km(p, src_, T, Tp, lw_[key_], X, Tp, resid=scale_)
    p.marks = [('front_end', 0)]                    # op index where each stage starts (for profiling)
    for li, lw in enumerate(L['layers']):
        p.marks.append((f'layer{li}', len(p.ops)))
        if km:
            tab = dict(off=L['pos_tab']['offs'][li], Tmax=L['pos_tab']['Tmax'], pitch=L['pos_tab']['pitch']) if pos_cache else None
            G = ops.gemm_km; n_ = d * Tp
            ops.ln_km(p, T, d, Tp, X, HB, lw['n_ff1']); ASMFF(lw, 'ff1_1', li)
            ASMRS(lw, 'ff1_2', li, H2, 0.5)
            ops.ln_km(p, T, d, Tp, X, HB, lw['n_att']); ASMROW(lw, 'qkv', li, QKV, 3 * d)
            ops.attention_km(p, T, d, H, QKV, PE, lw['pos'], lw['pbu'], lw['pbv'], ATT, R, pos_tab=tab, att_pitch=Tp, qpu=att_qpu, pos_asm=(L['pos_asm'][li] if att_qpu and L.get('pos_asm') else None))
            ASMRS(lw, 'out', li, ATT, 1.0)
            ops.ln_km(p, T, d, Tp, X, HB, lw['n_conv']); ASMPLAIN(lw, 'pw1', li, H2)
            if k <= 33:
                ops.glu_dw_km(p, T, d, k, Tp, H2, GL, CV, lw['dw_w'], lw['dw_b'], act=1)
            else:
                ops.glu_km(p, d, Tp, H2, GL)
                ops.dw_km(p, T, d, k, Tp, GL, CV, lw['dw_w'], lw['dw_b'], act=1)
            ASMRS(lw, 'pw2', li, CV, 1.0)
            ops.ln_km(p, T, d, Tp, X, HB, lw['n_ff2']); ASMFF(lw, 'ff2_1', li)
            ASMRS(lw, 'ff2_2', li, H2, 0.5)
            ops.ln_km(p, T, d, Tp, X, X, lw['n_out'])
            continue
        ops.ln(p, T, d, X, HB, lw['n_ff1']); lin(p, HB, T, lw['ff1_1'], H2, epi=2)
        lin(p, H2, T, lw['ff1_2'], HB); ops.add(p, T * d, X, HB, 0.5)
        ops.ln(p, T, d, X, HB, lw['n_att']); lin(p, HB, T, lw['qkv'], QKV)
        if att_km:
            pt = L.get('pos_tab')
            tab = dict(off=pt['offs'][li], Tmax=pt['Tmax'], pitch=pt['pitch']) if pos_cache else None
            ops.attention_km(p, T, d, H, QKV, PE, lw['pos'], lw['pbu'], lw['pbv'], ATT, R, pos_tab=tab)
        else:
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
    if km:
        ops.gemm_km(p, X, T, Tp, L['head'], LOG)
    else:
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
        fresh = os.path.exists(wprefix + '.bin') and os.path.exists(lpath)
        if fresh:
            j_ = json.load(open(lpath))
            # the blob file must be the one this layout was written with (tests and other tools also save work/nemo8_kit.bin: 2026-11-12 an
            # overwritten blob without the QPU weights made every transcript empty)
            fresh = ('fe3' in j_ and j_.get('blob_floats') == os.path.getsize(wprefix + '.bin') // 4
                     and (os.environ.get('GPUPORT_ASM') != '1' or ('asm' in j_ and 'pos_asm' in j_)))
        if not fresh:                                         # also rebuilds a cache from before the position tables
            w, L = prepare(load_nemo_ref())
            w.save(wprefix)
            L['blob_floats'] = w.size
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
