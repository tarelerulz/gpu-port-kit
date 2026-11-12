"""Push-constant layouts for shaders already used in the speech-app ports (~/speech-app/port/gpu_encoder2.py and
~/src/asr-gpu/shaders). Add a helper here each time a new shader is verified.

Verification status:
  add    numpy-tested in selftest.py
  silu   numpy-tested in selftest.py
  gemm12 numpy-tested in selftest.py (epi 0 and 2); asr-gpu + speech-app use it
  ln, glu  push layout copied from gpu_encoder2.py; NOT separately tested here
"""
import os
from .plan import Plan, split_groups
from .config import GEMM12_MAX_ROWS, MIN_WORKGROUPS
import numpy as np


def fbits(x):
    return int(np.float32(x).view(np.uint32))


def gemm12(plan, a_off, M, mm, out_off, K=None, N=None, sAm=None, epi=0, a_row_stride=None, kscratch=None):
    """out[m][n] = epi(bias[n] + sum_k A[m][k] * W[k][n]) for m < M.
    mm is the dict from Weights.put_mm(). A row stride defaults to K; out row stride is N.
    epi: 0 none, 1 hardswish, 2 silu, 3 relu, 4 sigmoid (common.glsl).
    Rows are split into GEMM12_MAX_ROWS blocks for the measured faster dispatch size.
    kscratch (2026-11-09): arena offset of a scratch region of at least K * 252 floats. When given and A is
    row-major (including a padded row stride), each row block is transposed to k-major by tr_km and read by the AV4 kernel
    (m contiguous): 2.6-20x faster matmul at the nemo8 shapes, same numbers. The region is reused by each op
    in turn; the executor puts a barrier after every dispatch."""
    N = mm['n'] if N is None else N
    K = mm['k'] if K is None else K
    sAm = K if sAm is None else sAm
    NP = mm['np']
    # AV4 vectorizes rows, not K: odd K and padded source rows are valid.
    av4 = kscratch is not None and sAm >= K and K > 0
    for r0 in range(0, M, GEMM12_MAX_ROWS):
        mb = min(GEMM12_MAX_ROWS, M - r0)
        threads = (NP // 12) * ((mb + 3) // 4)
        groups = max(MIN_WORKGROUPS, (threads + 255) // 256)
        gx, gy = split_groups(groups)
        # gemm12o4: same math, vec4 stores along N (2026-11-08). Needs N % 4 == 0.
        if av4:
            Mp = (mb + 3) // 4 * 4
            tg = max(MIN_WORKGROUPS, (Mp * K + LOCAL['tr_km'] - 1) // LOCAL['tr_km'])
            plan.add('tr_km', LOCAL['tr_km'], tg, 1, [mb, K, Mp, a_off + r0 * sAm, sAm, kscratch])
            name = 'gemm12av4o4' if N % 4 == 0 else 'gemm12av4'
            src, sa_m, sa_k = kscratch, 1, Mp
        else:
            name = 'gemm12o4' if N % 4 == 0 else 'gemm12'
            src, sa_m, sa_k = a_off + r0 * sAm, sAm, 1
        plan.add(name, 256, gx, gy,
                 [mb, N, K, NP, src, sa_m, sa_k, mm['wo'], mm['bo'],
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


# ---- attention on the matmul kernel (2026-11-12) ---------------------------------------------------------------
# Relative-position attention written as matmuls on the k-major kernel, in place of att_scores5jf / att_softmax /
# att_pv6. Scores are kept transposed: ST[h][j][i], rows of pitch TP, the query index i contiguous.
#   att_prep        QU, QV (bias and 1/sqrt(dh) folded in), KT: [c][i]
#   pos projection  PT[c][r] = P^T, written transposed by the matmul itself (tr_km + gemm12av4o1)
#   gemm12wa1       per head: ST = QU.K (content), GT[r][i] = QV.P[r] (position, all 2T-1 offsets)
#   att_comb        ST[j][i] += GT[T-1-i+j][i]
#   att_softmax_t   over j, in place
#   gemm12wa2       per head: ATT[i][h*dh..] = sum_j ST[j][i] * V[j]
# Threads per workgroup, tuned per shader by measurement (tools: sweep in GPU_MAP section 22). Must match the .comp.
BATCH = os.environ.get('GPUPORT_BATCH', '1') != '0'      # heads / column positions as workgroup-y batches (2026-11-12)
LOCAL = {'ln_km4': 256, 'dw_km4': 256, 'att_softmax_t4': 128, 'tr_km': 256, 'att_prep': 256, 'att_comb': 256, 'glu_kmh': 256}


def att_km_regions(T, d, H, pos=True):
    """(name, floats) of the extra arena regions attention_km needs. TK (transpose scratch) must hold d * 252."""
    TP = (T + 11) // 12 * 12; NPr = 2 * T - 1; PTP = (NPr + 11) // 12 * 12
    return [('QU', d * TP), ('QV', d * TP), ('KT', d * TP)] + ([('PT', d * PTP)] if pos else []) + \
        [('ST', H * T * TP), ('GT', H * (NPr + 3) * TP)]      # pos=False: position table comes from the weights


def _g(threads, local=256):
    return max(MIN_WORKGROUPS, (threads + local - 1) // local)


def pos_table_km(Wpos, bpos, Tmax):
    """Position projection for every offset a clip of up to Tmax frames can need, transposed for the matmul:
    [d][pitch], column r = position Tmax-1-r. Depends only on the layer, so it goes into the weight blob once
    (2026-11-12) and attention_km reads a window of it. Returns (float32 array, pitch)."""
    d = Wpos.shape[0]; NPm = 2 * Tmax - 1; pitch = (NPm + 11) // 12 * 12
    pos = (Tmax - 1 - np.arange(NPm)).astype(np.float64)
    div = np.exp((2.0 * np.arange(d // 2)) * -(np.log(10000.0) / d))
    pe = np.zeros((NPm, d), np.float32)
    pe[:, 0::2] = np.sin(pos[:, None] * div[None, :]); pe[:, 1::2] = np.cos(pos[:, None] * div[None, :])
    P = (pe @ np.asarray(Wpos, np.float32).T + np.asarray(bpos, np.float32)).astype(np.float32)
    out = np.zeros((d, pitch), np.float32); out[:, :NPm] = P.T
    return out, pitch


def attention_km(plan, T, d, H, qkv, pe, mm_pos, pbu, pbv, att, R, pos=True, pos_tab=None, att_pitch=None, fused=True):
    """qkv: rows [q | k | v] (stride 3d). pe: positional table [2T-1][d]. mm_pos: put_mm of the pos projection.
    pbu, pbv: weight offsets. att: output [T][d]. R: region offsets with the att_km_regions names and 'TK'.
    Needs d % 4 == 0 and (d / H) % 4 == 0."""
    dh = d // H
    assert d % 4 == 0 and dh % 4 == 0
    TP = (T + 11) // 12 * 12; NPr = 2 * T - 1; PTP = (NPr + 11) // 12 * 12
    QU, QV, KT, PT, ST, GT, TK = (R.get(x) for x in ['QU', 'QV', 'KT', 'PT', 'ST', 'GT', 'TK'])
    none = 0xffffffff
    e, GR, bd = 0, NPr, None
    if pos_tab is not None:      # dict(off, Tmax, pitch): table from pos_table_km in the weights; no projection here
        assert T <= pos_tab['Tmax']
        s = pos_tab['Tmax'] - T; s4 = s // 4 * 4; e = s - s4; GR = NPr + e
        bd = ('gemm12wp1', pos_tab['off'] + s4, pos_tab['pitch'])
        pos = False
    if pos:      # P^T: 248-row blocks so every block starts on a vec4 boundary of the transposed output
        for r0 in range(0, NPr, 248):
            mb = min(248, NPr - r0); Mp = (mb + 3) // 4 * 4
            plan.add('tr_km', LOCAL['tr_km'], _g(Mp * d, LOCAL['tr_km']), 1, [mb, d, Mp, pe + r0 * d, d, TK])
            plan.add('gemm12av4o1', 256, _g((mm_pos['np'] // 12) * (Mp // 4)), 1,
                     [mb, d, d, mm_pos['np'], TK, 1, Mp, mm_pos['wo'], mm_pos['bo'], PT + r0, 1, PTP, 0])
    plan.add('att_prep', LOCAL['att_prep'], _g(d * TP, LOCAL['att_prep']), 1, [T, d, dh, TP, qkv, QU, QV, KT, pbu, pbv])
    q4 = (T + 3) // 4
    NPg = (GR + 11) // 12 * 12
    name, wbase, wpitch = bd or ('gemm12wa1', PT, PTP)
    if BATCH:      # heads in one dispatch (workgroup y); u[22] A stride, u[23] out stride, u[24] W stride (2026-11-12)
        plan.add('gemm12wa1b', 256, _g((TP // 12) * q4), H, [T, T, dh, TP, QU, 1, TP, KT, none, ST, 1, TP, 0] + [0] * 8 + [TP, dh * TP, T * TP, dh * TP])
        plan.add(name + 'b', 256, _g((NPg // 12) * q4), H, [T, GR, dh, NPg, QV, 1, TP, wbase, none, GT, 1, TP, 0] + [0] * 8 + [wpitch, dh * TP, GR * TP, dh * wpitch])
    else:
        for h in range(H):
            hb = h * dh * TP
            push = [T, T, dh, TP, QU + hb, 1, TP, KT + hb, none, ST + h * T * TP, 1, TP, 0] + [0] * 8 + [TP]
            plan.add('gemm12wa1', 256, _g((TP // 12) * q4), 1, push)
            push = [T, GR, dh, NPg, QV + hb, 1, TP, wbase + h * dh * wpitch, none, GT + h * GR * TP, 1, TP, 0] + [0] * 8 + [wpitch]
            plan.add(name, 256, _g((NPg // 12) * q4), 1, push)
    if fused:       # shift + add + softmax in one pass, 4 queries per thread
        plan.add('att_softmax_t4', LOCAL['att_softmax_t4'], _g(H * q4, LOCAL['att_softmax_t4']), 1, [T, H, TP, ST, GT, e, GR])
    else:
        plan.add('att_comb', LOCAL['att_comb'], _g(H * T * TP, LOCAL['att_comb']), 1, [T, H, TP, ST, GT, e, GR])
        plan.add('att_softmax_t', 64, _g(H * T, 64), 1, [T, H, TP, ST])
    NPv = (dh + 11) // 12 * 12
    if BATCH and att_pitch:
        plan.add('gemm12wa1b', 256, _g((NPv // 12) * q4), H,
                 [T, dh, T, NPv, ST, 1, TP, qkv + 2 * d, none, att, 1, att_pitch, 0] + [0] * 8 + [3 * d, T * TP, dh * att_pitch, dh])
        return plan
    for h in range(H):
        if att_pitch:     # output k-major: att[c][i], rows of pitch att_pitch (feeds the next matmul with no transpose)
            push = [T, dh, T, NPv, ST + h * T * TP, 1, TP, qkv + 2 * d + h * dh, none, att + h * dh * att_pitch, 1, att_pitch, 0] + [0] * 8 + [3 * d]
            plan.add('gemm12wa1', 256, _g((NPv // 12) * q4), 1, push)
            continue
        push = [T, dh, T, NPv, ST + h * T * TP, 1, TP, qkv + 2 * d + h * dh, none, att + h * dh, d, 1, 0] + [0] * 8 + [3 * d]
        plan.add('gemm12wa2', 256, _g((NPv // 12) * q4), 1, push)
    return plan


# ---- k-major layer helpers (2026-11-12): activations kept as [c][t], rows of pitch TP = roundup(T, 4) -------------
def km_block(M):
    """Row block for k-major matmuls: every block must start on a vec4 boundary, so 248 instead of 251 when split."""
    return M if M <= GEMM12_MAX_ROWS else 248


def gemm_km(plan, src, M, pitch, mm, out, out_pitch=None, epi=0, resid=None):
    """Matmul reading k-major input src[k][m] (pitch floats per row). out_pitch set: output k-major out[n][m] with that
    pitch (gemm12av4o1); else row-major [M][N] (gemm12av4o4 / gemm12av4). No transpose dispatch."""
    N, K, NP = mm['n'], mm['k'], mm['np']
    B = km_block(M)
    for r0 in range(0, M, B):
        mb = min(B, M - r0)
        g = _g((NP // 12) * ((mb + 3) // 4))
        if out_pitch and resid is not None:     # out += resid * result (the residual add folded into the matmul)
            plan.add('gemm12av4rs', 256, g, 1, [mb, N, K, NP, src + r0, 1, pitch, mm['wo'], mm['bo'], out + r0, 1, out_pitch, epi, fbits(resid)])
        elif out_pitch:
            plan.add('gemm12av4o1', 256, g, 1, [mb, N, K, NP, src + r0, 1, pitch, mm['wo'], mm['bo'], out + r0, 1, out_pitch, epi])
        else:
            plan.add('gemm12av4o4' if N % 4 == 0 else 'gemm12av4', 256, g, 1,
                     [mb, N, K, NP, src + r0, 1, pitch, mm['wo'], mm['bo'], out + r0 * N, N, 1, epi])
    return plan


def gemm_km1(plan, src, M, pitch, mm, out, out_pitch, epi=0):
    """gemm_km in ONE dispatch for any M (no 248-row blocks): for tall, short-K matmuls such as the DFT stages (M % 4 == 0)."""
    N, K, NP = mm['n'], mm['k'], mm['np']
    plan.add('gemm12av4o1', 256, _g((NP // 12) * ((M + 3) // 4)), 1, [M, N, K, NP, src, 1, pitch, mm['wo'], mm['bo'], out, 1, out_pitch, epi])
    return plan


def gemm_km_batch(plan, src, M, pitch, mm, out, out_pitch, nb, src_stride, out_stride, epi=0):
    """nb independent k-major matmuls with the same weights in ONE dispatch (workgroup y = batch): A moves by src_stride,
    out by out_stride floats per batch (2026-11-12). M <= 251 per batch."""
    N, K, NP = mm['n'], mm['k'], mm['np']
    g = _g((NP // 12) * ((M + 3) // 4))
    plan.add('gemm12av4o1b', 256, g, nb, [M, N, K, NP, src, 1, pitch, mm['wo'], mm['bo'], out, 1, out_pitch, epi] + [0] * 9 + [src_stride, out_stride])
    return plan


def asm_mm(plan, a_off, M, a_pitch, asm, out_off, out_pitch, relu=False):
    """Hand-written QPU matmul (exec_hybrid runner, work/asm/qpu_mm.py): out[n][m] = bias[n] + sum_k W[k][n] * A[k][m] for m < M <= 256,
    A / out k-major with the given row pitches (floats). asm = dict(wo, k, ntiles, tile_words) from nemo8.put_asm. Not a shader: the
    runner submits it as a raw CSD job. No activation (add a silu / relu op after it)."""
    assert M <= 256
    plan.add('qpu_mmrelu' if relu else 'qpu_mm', 256, 1, 1, [asm['k'], asm['ntiles'], a_off, a_pitch, out_off, out_pitch, asm['wo'], asm['tile_words'], M])
    return plan


def asm_mm_rs(plan, a_off, M, a_pitch, asm, out_off, out_pitch, scale):
    """Like asm_mm but out += scale * (acc + bias) (the residual add folded in, qpu_mm.kernel_rs)."""
    assert M <= 256
    plan.add('qpu_mmrs', 256, 1, 1, [asm['k'], asm['ntiles'], a_off, a_pitch, out_off, out_pitch, asm['wo'], asm['tile_words'], M, fbits(scale)])
    return plan


def gemm_to_km(plan, a_off, M, mm, out, out_pitch, kscratch, sAm=None, epi=0):
    """Row-major input (transposed through kscratch) -> k-major output. For the step that enters the k-major layers."""
    N, K, NP = mm['n'], mm['k'], mm['np']
    sAm = K if sAm is None else sAm
    B = km_block(M)
    for r0 in range(0, M, B):
        mb = min(B, M - r0); Mp = (mb + 3) // 4 * 4
        plan.add('tr_km', LOCAL['tr_km'], _g(Mp * K, LOCAL['tr_km']), 1, [mb, K, Mp, a_off + r0 * sAm, sAm, kscratch])
        plan.add('gemm12av4o1', 256, _g((NP // 12) * (Mp // 4)), 1,
                 [mb, N, K, NP, kscratch, 1, Mp, mm['wo'], mm['bo'], out + r0, 1, out_pitch, epi])
    return plan


def ln_km(plan, T, d, TP, in_off, out_off, gamma_off, v4=True):
    if v4:          # 4 frames per thread: 0.93 -> see GPU_MAP section 21
        return plan.dispatch('ln_km4', LOCAL['ln_km4'], (T + 3) // 4, [T, d, TP, in_off, out_off, gamma_off, gamma_off + d])
    return plan.dispatch('ln_km', 64, T, [T, d, TP, in_off, out_off, gamma_off, gamma_off + d])


def glu_dw_km(plan, T, d, k, TP, in_off, halo_off, out_off, w_off, b_off, act=1):
    """GLU then depthwise conv, k-major. halo_off: region of d * (TP + 32) floats (GLU output with a zero halo)."""
    TPH = TP + 32
    plan.dispatch('glu_kmh', LOCAL['glu_kmh'], d * TPH, [T, d, TP, TPH, in_off, halo_off])
    return plan.dispatch('dw_km4', LOCAL['dw_km4'], d * ((T + 3) // 4), [T, d, k, TP, TPH, halo_off, out_off, w_off, b_off, act])


def glu_km(plan, d, TP, in_off, out_off):
    return plan.dispatch('glu_km', 256, d * TP, [d * TP, in_off, out_off])


def dw_km(plan, T, d, k, TP, in_off, out_off, w_off, b_off, act=1):
    return plan.dispatch('dw_km', 256, d * TP, [T, d, k, TP, in_off, out_off, w_off, b_off, act])
