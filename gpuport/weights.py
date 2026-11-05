"""Weight blob builder. Offsets are in floats. put_mm() stores a matmul weight the way gemm12 reads it:
W^T with N padded to a multiple of 12, row-major [K][NP]; bias padded to NP."""
import json
import os
import numpy as np
from .arena import roundup


class Weights:
    def __init__(self, align=16):
        self.parts, self.size, self.align = [], 0, align

    def put(self, arr):
        arr = np.asarray(arr, np.float32).ravel()
        pad = roundup(self.size, self.align) - self.size
        if pad:
            self.parts.append(np.zeros(pad, np.float32)); self.size += pad
        off = self.size
        self.parts.append(arr); self.size += arr.size
        return off

    def put_mm(self, Wm, bias=None, pad=12):
        """Wm: [N][K] (torch/nn layout). Returns dict(wo, bo, n, k, np) for gemm12 (pad=12) or gemm4 (pad=4)."""
        Wm = np.asarray(Wm, np.float32)
        n, k = Wm.shape
        NP = roundup(n, pad)
        Wt = np.zeros((k, NP), np.float32); Wt[:, :n] = Wm.T
        bp = np.zeros(NP, np.float32)
        if bias is not None:
            bp[:n] = np.asarray(bias, np.float32)
        return dict(wo=self.put(Wt), bo=self.put(bp), n=n, k=k, np=NP)

    def put_mm4(self, Wm, bias=None):
        """Weights for gemm_k5 (oido stream/gemm_k5.comp): vec4 per (output group cg, k) holding outputs cg*4..cg*4+3.
        Stored as [NP/4][K][4]. Bias padded to NP (float). Returns dict(wo4, bo, n, k, np)."""
        Wm = np.asarray(Wm, np.float32)
        n, k = Wm.shape
        NP = roundup(n, 4)
        Wp = np.zeros((NP, k), np.float32); Wp[:n] = Wm
        w4 = Wp.reshape(NP // 4, 4, k).transpose(0, 2, 1).copy()      # [cg][k][j]
        bp = np.zeros(NP, np.float32)
        if bias is not None:
            bp[:n] = np.asarray(bias, np.float32)
        return dict(wo4=self.put(w4), bo=self.put(bp), n=n, k=k, np=NP)

    def save(self, prefix, layout=None):
        """Writes <prefix>.bin (float32 LE) and <prefix>.json (layout dict, optional)."""
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        np.concatenate(self.parts).astype('<f4').tofile(prefix + '.bin')
        if layout is not None:
            json.dump(layout, open(prefix + '.json', 'w'))
        return prefix + '.bin'
