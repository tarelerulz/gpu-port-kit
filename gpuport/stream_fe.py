"""GPU front end for the oido streaming model (2026-11-05): log-mel, two stride-2 convs, subsampling -> X rows.

Windowed: each call runs the front end on a window of mel rows [a, b) (a a multiple of 4) and returns the X rows
whose receptive field is complete. The window's sample 0 is absolute sample 160*a, so the unchanged fe_frame_s
shader computes the same frames as the whole-utterance front end. Fixed normalization (fe_lognorm_fixed.comp),
not the per-utterance one in fe_lognorm.comp (nemo8)."""
import os, sys, numpy as np
from .arena import Arena
from .plan import Plan
from .weights import Weights
from . import ops
from .server import ExecServer
from .config import WORK_DIR, merged_shader_dir
from .nemo8 import SHADERS  # merged shader folder (kit + oido shaders)

OIDO_PORT = os.path.expanduser('~/oido/port')
ARENA_FLOATS = 24 << 20          # exec_server2 arena (SERVER_ARENA_FLOATS)


def prepare(model, work=WORK_DIR):
    """Front-end weights from the streaming model file. Returns (weights file, fe dict, sc, d)."""
    sys.path.insert(0, OIDO_PORT)
    import stream_ref as S
    M = S.load(model)
    d, sc = M['d'], M['sc']
    kk = np.arange(512)[None, :]; nn = np.arange(257)[:, None]
    cosT = np.cos(2 * np.pi * nn * kk / 512.0).astype(np.float32)
    sinT = np.sin(2 * np.pi * nn * kk / 512.0).astype(np.float32)
    w = Weights()
    c0_w, c0_b = M['c0']; c2_W, c2_b = M['c2']; sub_W, sub_b = M['sub']
    fe = dict(win=w.put(M['win']), dft_c=w.put_mm(cosT, np.zeros(257)), dft_s=w.put_mm(sinT, np.zeros(257)),
              mel=w.put_mm(M['fb'].T.astype(np.float32), np.zeros(80)),
              c0_w=w.put(np.asarray(c0_w, np.float32).reshape(sc, 9)), c0_b=w.put(np.asarray(c0_b, np.float32)),
              c2=w.put_mm(c2_W, c2_b), sub=w.put_mm(sub_W * np.sqrt(d), sub_b * np.sqrt(d)),
              nmean=w.put(np.asarray(M['nmean'], np.float32)), nstd=w.put(np.asarray(M['nstd'], np.float32)))
    os.makedirs(work, exist_ok=True)
    prefix = os.path.join(work, 'oido_stream_fe')
    w.save(prefix)
    return prefix + '.bin', fe, sc, d


def build(Tm, n, nrows, zero_row, fe, sc, d):
    """Plan for one window: Tm mel rows, n window samples (sample 0 = absolute 160*a), normalize nrows rows."""
    T2 = (Tm + 2 - 3) // 2 + 1; T3 = (T2 + 2 - 3) // 2 + 1
    FRS = 516
    ar = Arena(limit=ARENA_FLOATS)
    R = {}
    for name, size in [('XIN', n), ('FR', Tm * FRS), ('RE', Tm * 257), ('IM', Tm * 257), ('PW', Tm * 257),
                       ('MEL', Tm * 80), ('C0', T2 * 40 * sc), ('PAT', T3 * 20 * sc * 9), ('C2', T3 * 20 * sc),
                       ('FL', T3 * sc * 20), ('X', T3 * d)]:
        R[name] = ar.alloc(name, size)
    p = Plan()
    XIN, FR, RE, IM, PW, MEL = (R[x] for x in ['XIN', 'FR', 'RE', 'IM', 'PW', 'MEL'])
    p.dispatch('fe_frame_s', 64, Tm * 512, [Tm, n, XIN, FR, fe['win'], FRS])
    ops.gemm12(p, FR, Tm, fe['dft_c'], RE, K=512, N=257, sAm=FRS)
    ops.gemm12(p, FR, Tm, fe['dft_s'], IM, K=512, N=257, sAm=FRS)
    p.dispatch('fe_pw', 64, Tm * 257, [Tm, RE, IM, PW])
    ops.gemm12(p, PW, Tm, fe['mel'], MEL, K=257, N=80, sAm=257)
    p.dispatch('fe_lognorm_fixed', 64, 80, [nrows, MEL, zero_row, fe['nmean'], fe['nstd']])
    C0, PAT, C2, FL, X = (R[x] for x in ['C0', 'PAT', 'C2', 'FL', 'X'])
    p.dispatch('fe_conv0_v2', 64, T2 * 40, [Tm, MEL, C0, fe['c0_w'], fe['c0_b'], sc, T2])
    p.dispatch('fe_im2col_v2', 64, T3 * 20 * sc, [T3, T2, C0, PAT, sc])
    ops.gemm12(p, PAT, T3 * 20, fe['c2'], C2, K=sc * 9, N=sc, sAm=sc * 9, epi=3)
    p.dispatch('fe_flat', 64, T3 * sc * 20, [T3, sc, C2, FL])
    ops.gemm12(p, FL, T3, fe['sub'], X, K=sc * 20, N=d, sAm=sc * 20)
    return p, R, T3, n


class GpuFront:
    def __init__(self, model):
        self.weights, self.fe, self.sc, self.d = prepare(model)
        self.srv = ExecServer(self.weights, SHADERS).start()
        self.ms_last = 0.0

    def window(self, x, a, b, final):
        """X rows for absolute mel rows [a, b). x: float32 samples so far (abs). Returns (X rows, abs first X row)."""
        assert a % 4 == 0
        Tm = b - a
        n_win = 160 * Tm + 200
        xin = np.zeros(n_win, np.float32)
        s0 = 160 * a
        seg = x[s0:s0 + n_win]
        xin[:len(seg)] = seg
        nvalid = len(x) // 160
        if final:
            nrows, zero_row = nvalid - a, 1
        else:
            nrows, zero_row = Tm, 0
        plan, R, T3, n = build(Tm, n_win, nrows, zero_row, self.fe, self.sc, self.d)
        out, ms = self.srv.run(plan, xin, R['X'], T3 * self.d)
        self.ms_last = ms
        return out.reshape(T3, self.d), a // 4
