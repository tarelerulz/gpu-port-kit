# SPDX-License-Identifier: MIT
# Oido port scripts (nemo8 on the Pi GPU). Copyright (c) 2026 gpu-port-kit contributors. See LICENSE.
"""Float32 numpy reference for Oido's nemo8 CTC model (tasr_nemo.c). Output: raw head logits [T_enc][V+1].
Quantized layers are dequantized: W = q * scale[row] (4-bit: nibble - 8; blocked 4-bit when n % 16 == 0)."""
import sys, struct, numpy as np

NMEL, NBIN, WIN, HOP = 80, 257, 400, 160

class Blob:
    def __init__(self, path):
        self.b = open(path, 'rb').read()
        assert self.b[:4] == b'TNM1', 'not a TNM1 blob'
        self.hd = struct.unpack('<15I', self.b[4:64])
        self.p = 64
    def align(self):
        self.p = (self.p + 15) & ~15
    def f32(self, n):
        self.align()
        a = np.frombuffer(self.b, dtype='<f4', count=n, offset=self.p).copy()
        self.p += 4 * n
        return a
    def qlin(self, n, k, bits):
        self.align()
        blocked = bits == 4 and n % 16 == 0
        blk = 16 if (bits == 8 or blocked) else 32
        kp = (k + blk - 1) // blk * blk
        wb = n * kp if bits == 8 else n * kp // 2
        raw = np.frombuffer(self.b, dtype=np.uint8, count=wb, offset=self.p)
        self.p += wb
        if bits == 8:
            q = raw.view(np.int8).reshape(n, kp).astype(np.float32)
        elif blocked:
            # block of 16 rows = 8*kp bytes; byte [c][j] (c = k-pair index, j = row in block):
            # low nibble -> k = 2c, high nibble -> k = 2c+1
            r = raw.reshape(n // 16, kp // 2, 16).astype(np.int16)
            lo = (r & 15) - 8
            hi = (r >> 4) - 8
            q = np.empty((n // 16, 16, kp), dtype=np.float32)
            q[:, :, 0::2] = lo.transpose(0, 2, 1)
            q[:, :, 1::2] = hi.transpose(0, 2, 1)
            q = q.reshape(n, kp)
        else:
            # row-major, rows of kp/2 bytes; per 32-weight block: byte j -> k=j (low), k=j+16 (high)
            r = raw.reshape(n, kp // 2).astype(np.int16)
            nbk = kp // 32
            r = r.reshape(n, nbk, 16)
            lo = (r & 15) - 8
            hi = (r >> 4) - 8
            q = np.concatenate([lo, hi], axis=2).reshape(n, kp).astype(np.float32)
        s = self.f32(n)
        bias = self.f32(n)
        W = q[:, :k] * s[:, None]
        return W.astype(np.float32), bias

def layernorm(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -80, 80)))

def features(pcm, Bl):
    n = len(pcm)
    x = pcm.astype(np.float32) / 32768.0
    pre = np.empty(n, np.float32)
    pre[0] = x[0]
    pre[1:] = x[1:] - 0.97 * x[:-1]
    nvalid = n // HOP
    T = nvalid + 1
    window = Bl['window']
    fb = Bl['fb']  # [NBIN][NMEL]
    F = np.zeros((T, NMEL), np.float32)
    for t in range(T):
        idx = HOP * t - 200 + np.arange(WIN)
        ok = (idx >= 0) & (idx < n)
        seg = np.zeros(WIN, np.float32)
        seg[ok] = pre[idx[ok]]
        seg *= window
        X = np.fft.rfft(seg, 512)
        pw = (X.real ** 2 + X.imag ** 2).astype(np.float32)  # [257]
        mel = pw @ fb
        F[t] = np.log(mel + np.float32(5.9604644775390625e-08))
    # per-mel normalization over valid frames (ddof=1), last frame zero
    mu = F[:nvalid].mean(0)
    sd = F[:nvalid].std(0, ddof=1) + np.float32(1e-5)
    F[:nvalid] = (F[:nvalid] - mu) / sd
    F[nvalid] = 0.0
    return F, T

def conv3x3s2(inp, W, b=None):
    # inp [Tin][Fin][Cin]; W [Cout][Cin*9] with K index = ci*9 + dt*3 + df; pad 1, stride 2
    Tin, Fin, Cin = inp.shape
    Tout = (Tin + 2 - 3) // 2 + 1
    Fout = (Fin + 2 - 3) // 2 + 1
    pad = np.zeros((Tin + 2, Fin + 2, Cin), np.float32)
    pad[1:-1, 1:-1] = inp
    out = np.zeros((Tout, Fout, W.shape[0]), np.float32)
    Wr = W.reshape(W.shape[0], Cin, 3, 3)
    for dt in range(3):
        for df in range(3):
            sl = pad[dt:dt + 2 * Tout:2, df:df + 2 * Fout:2, :].reshape(Tout * Fout, Cin)  # [Tout*Fout][Cin]
            out += (sl @ Wr[:, :, dt, df].T).reshape(Tout, Fout, -1)
    if b is not None:
        out += b
    return out

def dwconv(x, w, b, k):
    # x [T][d]; w [d][k] (file layout); out[t][c] = b[c] + sum_j w[c][j] * x[t + j - k//2]
    T, d = x.shape
    h = k // 2
    xp = np.zeros((T + 2 * h, d), np.float32)
    xp[h:h + T] = x
    out = np.zeros((T, d), np.float32) + b
    for j in range(k):
        out += xp[j:j + T] * w[:, j][None, :]
    return out

def forward(path, pcm, stop_after=None):
    Bl = Blob(path)
    hd = Bl.hd
    d, H, ff, k, nl, sc, V = hd[1], hd[2], hd[3], hd[4], hd[5], hd[6], hd[7]
    bits = hd[8]
    assert hd[9] == 0, 'CTC model only'
    dh = d // H
    Bl.align()
    Bl_win = Bl.f32(WIN)
    fb = Bl.f32(NBIN * NMEL).reshape(NBIN, NMEL)
    F, Tmel = features(pcm, {'window': Bl_win, 'fb': fb})
    c0_w = Bl.f32(sc * 9).reshape(sc, 9)
    c0_b = Bl.f32(sc)
    f1 = (NMEL + 2 - 3) // 2 + 1
    f2 = (f1 + 2 - 3) // 2 + 1
    c2_W, c2_b = Bl.qlin(sc, sc * 9, 8)
    sub_W, sub_b = Bl.qlin(d, sc * f2, 8)
    layers = []
    for _ in range(nl):
        L = {}
        L['n_ff1'] = (Bl.f32(d), Bl.f32(d))
        L['ff1_1'] = Bl.qlin(ff, d, bits)
        L['ff1_2'] = Bl.qlin(d, ff, bits)
        L['n_att'] = (Bl.f32(d), Bl.f32(d))
        L['qkv'] = Bl.qlin(3 * d, d, bits)
        L['out'] = Bl.qlin(d, d, bits)
        L['pos'] = Bl.qlin(d, d, 8)
        L['pbu'] = Bl.f32(d)
        L['pbv'] = Bl.f32(d)
        L['n_conv'] = (Bl.f32(d), Bl.f32(d))
        L['pw1'] = Bl.qlin(2 * d, d, bits)
        L['dw_w'] = Bl.f32(d * k).reshape(d, k)
        L['dw_b'] = Bl.f32(d)
        L['pw2'] = Bl.qlin(d, d, bits)
        L['n_ff2'] = (Bl.f32(d), Bl.f32(d))
        L['ff2_1'] = Bl.qlin(ff, d, bits)
        L['ff2_2'] = Bl.qlin(d, ff, bits)
        L['n_out'] = (Bl.f32(d), Bl.f32(d))
        layers.append(L)
    head_W, head_b = Bl.qlin(V + 1, d, 8)

    # ---- subsampling: conv0 (ReLU), conv2 (ReLU), flatten channel-major, linear
    mel = F.reshape(Tmel, NMEL, 1)
    c0 = conv3x3s2(mel, c0_w.reshape(sc, 9).reshape(sc, 9)[:, :].reshape(sc, 9), c0_b)  # [T2][f1][sc]
    c0 = np.maximum(c0, 0)
    # conv0 weights are per-channel 3x3 with a single input channel: K index = dt*3 + df (ci = 0)
    c2 = conv3x3s2(c0, c2_W, c2_b)  # [T][f2][sc]
    c2 = np.maximum(c2, 0)
    T = c2.shape[0]
    flat = c2.transpose(0, 2, 1).reshape(T, sc * f2)  # index ch*f2 + f
    x = flat @ sub_W.T + sub_b
    x = (x * np.float32(np.sqrt(d))).astype(np.float32)

    # ---- relative positional encodings: row r holds position T-1-r, 2T-1 rows
    NP = 2 * T - 1
    pos_p = (T - 1 - np.arange(NP)).astype(np.float64)
    i = np.arange(d // 2)
    div = np.exp((2.0 * i) * -(np.log(10000.0) / d))
    pe = np.zeros((NP, d), np.float64)
    pe[:, 0::2] = np.sin(pos_p[:, None] * div[None, :])
    pe[:, 1::2] = np.cos(pos_p[:, None] * div[None, :])
    pe = pe.astype(np.float32)

    for li, L in enumerate(layers):
        if stop_after is not None and li == stop_after:
            return x, dict(T=T)
        # FF1 (half step)
        y = layernorm(x, *L['n_ff1'])
        h = silu(y @ L['ff1_1'][0].T + L['ff1_1'][1])
        x = x + 0.5 * (h @ L['ff1_2'][0].T + L['ff1_2'][1])
        # MHSA with relative positions
        y = layernorm(x, *L['n_att'])
        qkv = y @ L['qkv'][0].T + L['qkv'][1]
        q = qkv[:, :d].reshape(T, H, dh)
        kk = qkv[:, d:2 * d].reshape(T, H, dh)
        v = qkv[:, 2 * d:].reshape(T, H, dh)
        P = (pe @ L['pos'][0].T + L['pos'][1]).reshape(NP, H, dh)
        qu = q + L['pbu'].reshape(H, dh)[None]
        qv = q + L['pbv'].reshape(H, dh)[None]
        att = np.zeros((T, H, dh), np.float32)
        scale = np.float32(1.0 / np.sqrt(dh))
        for hh in range(H):
            ac = qu[:, hh, :] @ kk[:, hh, :].T  # [i][j]
            # position term: score(i,j) uses pos row (T-1-i) + j
            bd = qv[:, hh, :] @ P[:, hh, :].T  # [i][r]
            ii = np.arange(T)[:, None]
            jj = np.arange(T)[None, :]
            bpos = np.take_along_axis(bd, (T - 1 - ii) + jj, axis=1)
            s = (ac + bpos) * scale
            s = s - s.max(1, keepdims=True)
            w = np.exp(s)
            w = w / w.sum(1, keepdims=True)
            att[:, hh, :] = w @ v[:, hh, :]
        x = x + (att.reshape(T, d) @ L['out'][0].T + L['out'][1])
        # conv module
        y = layernorm(x, *L['n_conv'])
        z = y @ L['pw1'][0].T + L['pw1'][1]
        glu = z[:, :d] * (1.0 / (1.0 + np.exp(-np.clip(z[:, d:], -80, 80))))
        cv = dwconv(glu, L['dw_w'], L['dw_b'], k)
        cv = silu(cv)
        x = x + (cv @ L['pw2'][0].T + L['pw2'][1])
        # FF2 half step then LN
        y = layernorm(x, *L['n_ff2'])
        h = silu(y @ L['ff2_1'][0].T + L['ff2_1'][1])
        z = h @ L['ff2_2'][0].T + L['ff2_2'][1]
        x = layernorm(x + 0.5 * z, *L['n_out'])
    if stop_after is not None:
        return x, dict(T=T)
    logits = x @ head_W.T + head_b
    return logits.astype(np.float32), dict(d=d, H=H, ff=ff, k=k, nl=nl, sc=sc, V=V, bits=bits, T=T)

if __name__ == '__main__':
    model, pcm_path, ref_path = sys.argv[1], sys.argv[2], sys.argv[3]
    pcm = np.fromfile(pcm_path, dtype='<i2')
    logits, info = forward(model, pcm)
    print('info', info, 'logits', logits.shape)
    if ref_path != '-':
        ref = np.fromfile(ref_path, dtype='<f4').reshape(-1, info['V'] + 1)
        n = min(len(ref), len(logits))
        diff = np.abs(ref[:n] - logits[:n])
        am_ref = ref[:n].argmax(1)
        am_me = logits[:n].argmax(1)
        print('frames compared', n, 'max abs diff', float(diff.max()), 'mean abs diff', float(diff.mean()))
        print('argmax agreement', float((am_ref == am_me).mean()), 'blank ref/me', float((am_ref == info['V']).mean()), float((am_me == info['V']).mean()))
