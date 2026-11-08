"""Push-constant layouts for shaders already used in the speech-app ports (~/speech-app/port/gpu_encoder2.py and
~/src/asr-gpu/shaders). Add a helper here each time a new shader is verified.

Verification status:
  add    numpy-tested in selftest.py
  silu   numpy-tested in selftest.py
  gemm12 numpy-tested in selftest.py (epi 0 and 2); asr-gpu + speech-app use it
  ln, glu  push layout copied from gpu_encoder2.py; NOT separately tested here
"""
from .plan import Plan, split_groups
from .config import GEMM12_MAX_ROWS, MIN_WORKGROUPS
import numpy as np


def fbits(x):
    return int(np.float32(x).view(np.uint32))


def gemm12(plan, a_off, M, mm, out_off, K=None, N=None, sAm=None, epi=0, a_row_stride=None):
    """out[m][n] = epi(bias[n] + sum_k A[m][k] * W[k][n]) for m < M.
    mm is the dict from Weights.put_mm(). A row stride defaults to K; out row stride is N.
    epi: 0 none, 1 hardswish, 2 silu, 3 relu, 4 sigmoid (common.glsl).
    Row blocks of GEMM12_MAX_ROWS are split (the shader is only correct up to that)."""
    N = mm['n'] if N is None else N
    K = mm['k'] if K is None else K
    sAm = K if sAm is None else sAm
    NP = mm['np']
    for r0 in range(0, M, GEMM12_MAX_ROWS):
        mb = min(GEMM12_MAX_ROWS, M - r0)
        threads = (NP // 12) * ((mb + 3) // 4)
        groups = max(MIN_WORKGROUPS, (threads + 255) // 256)
        gx, gy = split_groups(groups)
        # gemm12o4: same math, vec4 stores along N (2026-11-08). Needs N % 4 == 0.
        name = 'gemm12o4' if N % 4 == 0 else 'gemm12'
        plan.add(name, 256, gx, gy,
                 [mb, N, K, NP, a_off + r0 * sAm, sAm, 1, mm['wo'], mm['bo'],
                  out_off + r0 * N, N, 1, epi])
    return plan


def add(plan, n, dst_off, src_off, scale):
    """a[dst+i] += scale * a[src+i]."""
    return plan.dispatch('add', 128, n, [n, dst_off, src_off, fbits(scale)])


def silu(plan, n, in_off, out_off):
    return plan.dispatch('silu', 128, n, [n, in_off, out_off])


def ln(plan, T, d, in_off, out_off, gamma_off):
    """One thread per row. gamma at gamma_off, beta at gamma_off + d (from gpu_encoder2.py)."""
    return plan.dispatch('ln', 64, T, [T, d, in_off, out_off, gamma_off, gamma_off + d])


def glu(plan, T, d, in_off, out_off):
    """From gpu_encoder2.py: push [T, d, H2, GL], one thread per output element."""
    return plan.dispatch('glu', 128, T * d, [T, d, in_off, out_off])


def gemm4(plan, a_off, M, mm, out_off, K=None, N=None, sAm=None, epi=0):
    """gemm4 (speech-app gpu/gemm4.comp): 4 outputs x 4 rows per thread, weights [K][NP] with NP = roundup(N,4).
    Build the weights with Weights.put_mm(W, b, pad=4). Output scalar stores, out row stride N."""
    N = mm['n'] if N is None else N
    K = mm['k'] if K is None else K
    sAm = K if sAm is None else sAm
    NP = mm['np']
    threads = (NP // 4) * ((M + 3) // 4)
    groups = max(MIN_WORKGROUPS, (threads + 255) // 256)
    gx, gy = split_groups(groups)
    plan.add('gemm4', 256, gx, gy, [M, N, K, NP, a_off, sAm, 1, mm['wo'], mm['bo'], out_off, N, 0, epi])
    return plan


def gemm_k5(plan, a_off, M, mm4, out_off, K=None, N=None, sAm=None, epi=0):
    """gemm_k5 (speech-app stream/gemm_k5.comp): one workgroup per (4 outputs, 4 rows), 64 threads split K, shared-memory
    reduction. Weights from Weights.put_mm4(). Needs the stream shader folder in the merged shader dir."""
    N = mm4['n'] if N is None else N
    K = mm4['k'] if K is None else K
    sAm = K if sAm is None else sAm
    gx = (N + 3) // 4; gy = (M + 3) // 4
    plan.add('gemm_k5', 64, gx, gy, [M, N, K, mm4['np'], a_off, sAm, 1, mm4['wo4'], mm4['bo'], out_off, N, 0, epi])
    return plan


def gemm12v(plan, a_off, M, mm, out_off, K=None, N=None, sAm=None, epi=0):
    """gemm12v (kit shaders/gemm12v.comp): gemm12 tile with A read as vec4 along K. Same weights (put_mm).
    Needs K % 4 == 0, sAm % 4 == 0, a_off % 4 == 0 (checked here)."""
    N = mm['n'] if N is None else N
    K = mm['k'] if K is None else K
    sAm = K if sAm is None else sAm
    if K % 4 or sAm % 4 or a_off % 4:
        raise ValueError('gemm12v needs K, sAm and a_off divisible by 4')
    NP = mm['np']
    RG = (M + 3) // 4
    total = (NP // 12) * RG
    groups = max(MIN_WORKGROUPS, (total + 63) // 64)
    gx, gy = split_groups(groups)
    plan.add('gemm12v', 64, gx, gy, [M, N, K, NP, a_off, sAm, 1, mm['wo'], mm['bo'], out_off, N, 1, epi])
    return plan
