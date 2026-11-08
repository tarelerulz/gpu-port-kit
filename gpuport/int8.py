"""int8 linear layers on the GPU: activations quantized per row to int8, weights stored as packed int8 with one
scale per output row, int32 accumulate with packed dot products, then rescale + bias + activation.
Shaders: shaders/quant_rows.comp, shaders/gemm_i8.comp. Numbers match a numpy reference exactly before rounding."""
import numpy as np
from .arena import roundup


def read_q8(Bl, n, k):
    """Read one 8-bit linear from an speech-app blob (same layout as nemo_ref.Blob.qlin, bits=8).
    Returns (q int8 [n][k], scale f32 [n], bias f32 [n])."""
    Bl.align()
    kp = (k + 15) // 16 * 16
    raw = np.frombuffer(Bl.b, dtype=np.uint8, count=n * kp, offset=Bl.p)
    Bl.p += n * kp
    q = raw.view(np.int8).reshape(n, kp)[:, :k].copy()
    s = Bl.f32(n)
    b = Bl.f32(n)
    return q, s, b


def pack_linear(w, q, s, b):
    """Store an int8 linear in a Weights builder. Packed along K: Wp[kq][n] holds q[n][4kq..4kq+3] in one uint.
    Returns dict(kind='i8', wo, sw, bo, n, k, np, kq)."""
    q = np.asarray(q, np.int8)
    n, k = q.shape
    assert k % 4 == 0
    NP = roundup(n, 12); KQ = k // 4
    u = q.view(np.uint8).astype(np.uint32).reshape(n, KQ, 4)
    pw = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16) | (u[..., 3] << 24)     # [n][KQ]
    Wp = np.zeros((KQ, NP), np.uint32); Wp[:, :n] = pw.T
    sw = np.zeros(NP, np.float32); sw[:n] = s
    bp = np.zeros(NP, np.float32); bp[:n] = b
    return dict(kind='i8', wo=w.put(Wp.ravel().view(np.float32)), sw=w.put(sw), bo=w.put(bp),
                n=n, k=k, np=NP, kq=KQ)


def quant_op(plan, in_off, M, K, sIn, q_off, s_off):
    """Quantize M rows of K floats (row stride sIn) into packed int8 at q_off and row scales at s_off."""
    return plan.dispatch('quant_rows', 64, M, [M, K, in_off, sIn, q_off, s_off])


def lin_i8(plan, q_off, s_off, M, mm, out_off, epi=0):
    """int8 linear: needs the packed activations (q_off) and row scales (s_off) from quant_op."""
    NP = mm['np']; RG = (M + 3) // 4
    total = (NP // 12) * RG
    return plan.dispatch('gemm_i8', 256, total,
                         [M, mm['n'], mm['kq'], NP, q_off, s_off, 0, mm['wo'], mm['bo'], out_off, mm['n'], 0, epi, mm['sw']])
