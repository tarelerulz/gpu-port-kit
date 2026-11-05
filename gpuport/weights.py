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

    def put_mm(self, Wm, bias=None):
        """Wm: [N][K] (torch/nn layout). Returns dict(wo, bo, n, k, np) for gemm12 pushes."""
        Wm = np.asarray(Wm, np.float32)
        n, k = Wm.shape
        NP = roundup(n, 12)
        Wt = np.zeros((k, NP), np.float32); Wt[:, :n] = Wm.T
        bp = np.zeros(NP, np.float32)
        if bias is not None:
            bp[:n] = np.asarray(bias, np.float32)
        return dict(wo=self.put(Wt), bo=self.put(bp), n=n, k=k, np=NP)

    def save(self, prefix, layout=None):
        """Writes <prefix>.bin (float32 LE) and <prefix>.json (layout dict, optional)."""
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        np.concatenate(self.parts).astype('<f4').tofile(prefix + '.bin')
        if layout is not None:
            json.dump(layout, open(prefix + '.json', 'w'))
        return prefix + '.bin'
