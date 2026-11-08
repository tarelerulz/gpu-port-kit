# SPDX-License-Identifier: MIT
# Oido port scripts (nemo8 on the Pi GPU). Copyright (c) 2026 gpu-port-kit contributors. See LICENSE.
"""Oido nemo8 on the Pi GPU, front end included: audio -> framing -> FFT as DFT matmul -> mel -> log/normalize ->
conv0 -> conv2 (im2col matmul) -> subsampling projection -> 16 conformer layers -> head. The CPU only builds the plan
and decodes the text. Weights are cached (cache2/); the server is gpu/exec_server (resident)."""
import json, os, subprocess, sys, threading, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nemo_ref import Blob

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get('OIDO_MODEL', os.path.join(HERE, '..', 'models', 'nemo8.tnm'))   # the nemo8 weights file
CACHE = os.path.join(HERE, 'cache2')
GPU_DIR = os.environ.get('OIDO_GPU_DIR', os.path.join(HERE, 'gpu'))   # folder with the GPU server binary and its shaders
SERVER = os.environ.get('OIDO_GPU_SERVER', os.path.join(GPU_DIR, 'exec_server'))
ARENA_LIMIT = 24 << 20
REGION_GAP = 131072   # gap after every arena region: a matmul output placed just below its input was read back wrong (2026-11-03)
def roundup(a, b): return (a + b - 1) // b * b
def fbits(x):
    import struct
    return struct.unpack('<I', struct.pack('<f', x))[0]

def prepare(model=MODEL, cache=CACHE):
    os.makedirs(cache, exist_ok=True)
    Bl = Blob(model); hd = Bl.hd
    d, H, ff, k, nl, sc, V, bits = hd[1], hd[2], hd[3], hd[4], hd[5], hd[6], hd[7], hd[8]
    assert hd[9] == 0 and bits == 8
    Bl.align()
    window = Bl.f32(400); fb = Bl.f32(257 * 80).reshape(257, 80)
    c0_w = Bl.f32(sc * 9).reshape(sc, 9); c0_b = Bl.f32(sc)
    f1 = (80 + 2 - 3) // 2 + 1; f2 = (f1 + 2 - 3) // 2 + 1
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

    W = []; woff = [0]
    def put(arr):
        arr = np.asarray(arr, np.float32).ravel()
        pad = roundup(woff[0], 16) - woff[0]
        if pad: W.append(np.zeros(pad, np.float32)); woff[0] += pad
        off = woff[0]; W.append(arr); woff[0] += arr.size
        return off
    def put_mm(Wm, bias):
        n, kdim = Wm.shape; NPn = roundup(n, 12)
        Wt = np.zeros((kdim, NPn), np.float32); Wt[:, :n] = Wm.T
        bp = np.zeros(NPn, np.float32); bp[:n] = bias
        return dict(wo=put(Wt), bo=put(bp), n=n, k=kdim)
    # front end matrices
    kk = np.arange(512)[None, :]; nn = np.arange(257)[:, None]
    cosT = np.cos(2 * np.pi * nn * kk / 512.0).astype(np.float32)   # [257][512]
    sinT = np.sin(2 * np.pi * nn * kk / 512.0).astype(np.float32)
    fe = dict(win=put(window), dft_c=put_mm(cosT, np.zeros(257)), dft_s=put_mm(sinT, np.zeros(257)),
              mel=put_mm(fb.T.astype(np.float32), np.zeros(80)),
              c0_w=put(c0_w), c0_b=put(c0_b), c2=put_mm(c2_W, c2_b),
              sub=put_mm(sub_W * np.sqrt(d), sub_b * np.sqrt(d)))
    lay = []
    for L in layers:
        lw = {}
        for key in ['ff1_1', 'ff1_2', 'ff2_1', 'ff2_2', 'qkv', 'out', 'pos', 'pw1', 'pw2']:
            lw[key] = put_mm(*L[key])
        for key in ['n_ff1', 'n_att', 'n_conv', 'n_ff2', 'n_out']:
            lw[key] = put(np.concatenate(L[key]))
        lw['pbu'] = put(L['pbu']); lw['pbv'] = put(L['pbv'])
        lw['dw_w'] = put(L['dw_w'].T.copy()); lw['dw_b'] = put(L['dw_b'])
        lay.append(lw)
    head = put_mm(head_W, head_b)
    np.concatenate(W).astype('<f4').tofile(os.path.join(cache, 'weights.bin'))
    json.dump(dict(d=d, H=H, ff=ff, k=k, nl=nl, sc=sc, V=V, f2=f2, fe=fe, layers=lay, head=head, tokens=toks),
              open(os.path.join(cache, 'layout.json'), 'w'))
    return cache

def build(n, L):
    """Plan for n audio samples. Returns (ops, arena_size, regions)."""
    d, H, ff, k, sc, V, f2, fe = L['d'], L['H'], L['ff'], L['k'], L['sc'], L['V'], L['f2'], L['fe']
    V1 = V + 1; nvalid = n // 160; Tm = nvalid + 1
    T2 = (Tm + 2 - 3) // 2 + 1; T3 = (T2 + 2 - 3) // 2 + 1; T = T3; NP = 2 * T - 1
    dh = d // H
    R = {}
    cur = [0]
    def alloc(name, size):
        cur[0] = roundup(cur[0], 64); R[name] = cur[0]; cur[0] += size + REGION_GAP
        return R[name]
    alloc('XIN', n); alloc('FR', Tm * 512); alloc('RE', Tm * 257); alloc('IM', Tm * 257); alloc('PW', Tm * 257)
    alloc('MEL', Tm * 80); alloc('C0', T2 * 40 * sc); alloc('PAT', T3 * 20 * sc * 9); alloc('C2', T3 * 20 * sc)
    alloc('FL', T3 * sc * 20)
    alloc('X', T * d); alloc('HB', T * d); alloc('H2', T * ff); alloc('QKV', T * 3 * d); alloc('P', NP * d)
    alloc('PE', NP * d); alloc('ATT', T * d); alloc('GL', T * d); alloc('CV', T * d); alloc('SC', H * T * T)
    alloc('LOG', T * V1)
    assert cur[0] <= ARENA_LIMIT, f'arena needs {cur[0]} floats (limit {ARENA_LIMIT}); utterance too long for the GPU path'
    ops = []
    def gemm(Ab, M, mm, ob, N=None, K=None, sAm=None, epi=0):
        N = mm['n'] if N is None else N; K = mm['k'] if K is None else K
        sAm = K if sAm is None else sAm
        NPn = roundup(mm['n'], 12)
        for r0 in range(0, M, 251):
            mb = min(251, M - r0)
            gx = max(24, ((NPn // 12) * ((mb + 3) // 4) + 255) // 256)
            ops.append(('gemm12', 256, gx, 1, [mb, N, K, NPn, Ab + r0 * sAm, sAm, 1, mm['wo'], mm['bo'], ob + r0 * N, N, 1, epi] + [0] * 19))
    def ew(name, total, push, local=64):
        ops.append((name, local, max(24, (total + local - 1) // local), 1, push + [0] * (32 - len(push))))
    XIN = R['XIN']
    ew('fe_frame', Tm * 512, [Tm, n, XIN, R['FR'], fe['win']])
    gemm(R['FR'], Tm, fe['dft_c'], R['RE'], N=257, K=512, sAm=512)
    gemm(R['FR'], Tm, fe['dft_s'], R['IM'], N=257, K=512, sAm=512)
    ew('fe_pw', Tm * 257, [Tm, R['RE'], R['IM'], R['PW']])
    gemm(R['PW'], Tm, fe['mel'], R['MEL'], N=80, K=257, sAm=257)
    ew('fe_lognorm', 80, [nvalid, R['MEL']])
    ew('fe_conv0', T2 * 40 * sc, [Tm, R['MEL'], R['C0'], fe['c0_w'], fe['c0_b'], sc, T2])
    groups = (T3 * 20 * sc * 9 + 63) // 64
    ops.append(('fe_im2col', 64, min(groups, 32768), (groups + 32767) // 32768, [T3, T2, R['C0'], R['PAT'], sc] + [0] * 27))
    gemm(R['PAT'], T3 * 20, fe['c2'], R['C2'], N=sc, K=sc * 9, sAm=sc * 9, epi=3)
    ew('fe_flat', T3 * sc * 20, [T3, sc, R['C2'], R['FL']])
    gemm(R['FL'], T3, fe['sub'], R['X'], N=d, K=sc * 20, sAm=sc * 20)
    ew('pe_gen', NP * d, [T, d, R['PE']])
    X, HB, H2, QKV, P, PE, ATT, GL, CV, SC, LOG = (R[x] for x in ['X', 'HB', 'H2', 'QKV', 'P', 'PE', 'ATT', 'GL', 'CV', 'SC', 'LOG'])
    for lw in L['layers']:
        def ln(inO, outO, go):
            ew('ln', T, [T, d, inO, outO, go, go + d], local=64)
        def add(dst, src, scale):
            ew('add', T * d, [T * d, dst, src, fbits(scale)], local=128)
        ln(X, HB, lw['n_ff1']); gemm(HB, T, lw['ff1_1'], H2, epi=2)
        gemm(H2, T, lw['ff1_2'], HB); add(X, HB, 0.5)
        ln(X, HB, lw['n_att']); gemm(HB, T, lw['qkv'], QKV)
        gemm(PE, NP, lw['pos'], P)
        ew('att_scores5', H * T * ((T + 3) // 4), [T, d, H, dh, QKV, P, SC, lw['pbu'], lw['pbv']])
        ew('att_softmax', H * T, [T, d, H, dh, QKV, P, SC, lw['pbu'], lw['pbv'], ATT])
        ops.append(('att_pv6', 64, max(24, (H * ((T + 1) // 2) * (dh // 4) + 63) // 64), 1, [T, d, H, dh, QKV, P, SC, lw['pbu'], lw['pbv'], ATT] + [0] * 22))
        gemm(ATT, T, lw['out'], HB); add(X, HB, 1.0)
        ln(X, HB, lw['n_conv']); gemm(HB, T, lw['pw1'], H2)
        ew('glu', T * d, [T, d, H2, GL], local=128)
        ops.append(('dw4', 64, max(24, (d // 4 + 63) // 64), T, [T, d, k, GL, CV, lw['dw_w'], lw['dw_b']] + [0] * 25))
        ew('silu', T * d, [T * d, CV, CV], local=128)
        gemm(CV, T, lw['pw2'], HB); add(X, HB, 1.0)
        ln(X, HB, lw['n_ff2']); gemm(HB, T, lw['ff2_1'], H2, epi=2)
        gemm(H2, T, lw['ff2_2'], HB); add(X, HB, 0.5)
        ln(X, X, lw['n_out'])
    gemm(X, T, L['head'], LOG)
    return ops, cur[0], R, Tm, T

class GpuEncoder2:
    def __init__(self, cache=CACHE):
        if not os.path.exists(os.path.join(cache, 'layout.json')):
            prepare(cache=cache)
        self.cache = cache
        self.L = json.load(open(os.path.join(cache, 'layout.json')))
        self.tmp = os.path.join(cache, 'req'); os.makedirs(self.tmp, exist_ok=True)
        self.proc = None; self.lock = threading.Lock(); self.seq = 0; self.last_ms = None
    def start(self):
        if self.proc is None or self.proc.poll() is not None:
            env = dict(os.environ)   # Vulkan loader and driver: set LD_LIBRARY_PATH and VK_ICD_FILENAMES in your shell
            self.proc = subprocess.Popen([SERVER, os.path.join(self.cache, 'weights.bin')], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
                                         cwd=GPU_DIR)
        return self
    def logits(self, pcm):
        pcm = np.asarray(pcm, np.int16)
        n = len(pcm)
        L = self.L
        ops, size, R, Tm, T = build(n, L)
        arena_in = (pcm.astype(np.float32) / 32768.0)
        with self.lock:
            self.seq += 1
            base = os.path.join(self.tmp, f'f{self.seq % 4}')
            with open(base + '.plan', 'w') as f:
                for name, local, gx, gy, push in ops:
                    f.write(f"{name} {local} {gx} {gy} " + ' '.join(str(int(x)) for x in push[:32]) + '\n')
            arena_in.astype('<f4').tofile(base + '.arena')
            self.start()
            self.proc.stdin.write(f"{base}.plan {base}.arena {base}.out {R['LOG']} {T * (L['V'] + 1)}\n"); self.proc.stdin.flush()
            resp = self.proc.stdout.readline()
            if not resp.startswith('done'):
                raise RuntimeError('gpu server: ' + resp.strip())
            out = np.fromfile(base + '.out', dtype='<f4').reshape(T, L['V'] + 1)
        self.last_ms = float(resp.split()[1])
        return out
    def text(self, logits):
        V = self.L['V']; toks = self.L['tokens']
        am = logits.argmax(1); out, prev = [], -1
        for a in am:
            if a != prev and a != V: out.append(toks[int(a)])
            prev = a
        s = ''.join(out)
        return s[1:] if s.startswith(' ') else s
    def transcribe(self, pcm):
        return self.text(self.logits(pcm))
    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close(); self.proc.wait(timeout=10)

if __name__ == '__main__':
    import time
    enc = GpuEncoder2()
    pcm = np.fromfile(sys.argv[1], dtype='<i2')
    for i in range(2):
        t = time.time(); lg = enc.logits(pcm); print('total', round(time.time() - t, 2), 's, server', enc.last_ms, 'ms')
    print(enc.text(lg))
    enc.close()
