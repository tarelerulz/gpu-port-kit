"""Relative-position attention on the GPU, checked against numpy (the math from
~/speech-app/port/nemo_ref.py forward, one conformer layer). Chain:
  pe_gen (sin/cos table, 2T-1 rows)  ->  gemm12 pos projection (P = PE @ Wpos^T + b)
  ->  att_scores5 (S = scale*(qu.k + qv.P[T-1-i+j]))  ->  att_softmax (in place)  ->  att_pv6 (out = S @ V)
qkv rows are [q(d) | k(d) | v(d)]; head h uses columns h*dh..(h+1)*dh of each third.
Configs: real nemo8 dims (d=176, H=4, dh=44) with odd T, and a small one with T not a multiple of 4.
Run: python3 ~/gpu-port-kit/tests/test_attention_layout.py"""
import os
import sys
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport import Arena, Weights, Plan, ops, check
from gpuport.run import run_prefix
from gpuport.config import WORK_DIR, MIN_WORKGROUPS

rng = np.random.default_rng(3)


def pos_table(T, d):
    """Same as nemo_ref: row r holds position T-1-r; sin on even columns, cos on odd."""
    NP = 2 * T - 1
    pos_p = (T - 1 - np.arange(NP)).astype(np.float64)
    i = np.arange(d // 2)
    div = np.exp((2.0 * i) * -(np.log(10000.0) / d))
    pe = np.zeros((NP, d), np.float64)
    pe[:, 0::2] = np.sin(pos_p[:, None] * div[None, :])
    pe[:, 1::2] = np.cos(pos_p[:, None] * div[None, :])
    return pe.astype(np.float32)


def att_ref(qkv, P, pbu, pbv, T, d, H, dh):
    """numpy attention as in nemo_ref: scores per head, softmax over j, weighted values."""
    q = qkv[:, :d].reshape(T, H, dh); k = qkv[:, d:2 * d].reshape(T, H, dh); v = qkv[:, 2 * d:].reshape(T, H, dh)
    Pr = P.reshape(2 * T - 1, H, dh)
    qu = q + pbu.reshape(H, dh)[None]; qv = q + pbv.reshape(H, dh)[None]
    att = np.zeros((T, H, dh), np.float32)
    scale = np.float32(1.0 / np.sqrt(dh))
    ii = np.arange(T)[:, None]; jj = np.arange(T)[None, :]
    for hh in range(H):
        ac = qu[:, hh, :] @ k[:, hh, :].T
        bd = qv[:, hh, :] @ Pr[:, hh, :].T
        bpos = np.take_along_axis(bd, (T - 1 - ii) + jj, axis=1)
        s = (ac + bpos) * scale
        s = s - s.max(1, keepdims=True)
        w = np.exp(s); w = w / w.sum(1, keepdims=True)
        att[:, hh, :] = w @ v[:, hh, :]
    return att.reshape(T, d)


def run_config(T, d, H, scores='att_scores5'):
    dh = d // H
    NP = 2 * T - 1
    print(f'--- T={T} d={d} H={H} dh={dh} NP={NP} scores={scores}')
    qkv = rng.standard_normal((T, 3 * d)).astype(np.float32)
    Wpos = (rng.standard_normal((d, d)) * 0.05).astype(np.float32)
    bpos = (rng.standard_normal(d) * 0.1).astype(np.float32)
    pbu = (rng.standard_normal(d) * 0.1).astype(np.float32)
    pbv = (rng.standard_normal(d) * 0.1).astype(np.float32)

    w = Weights()
    mm_pos = w.put_mm(Wpos, bpos)
    pbuO = w.put(pbu); pbvO = w.put(pbv)
    path = w.save(os.path.join(WORK_DIR, f'att_{T}_{d}_{H}'))

    ar = Arena()
    iQKV = ar.alloc('QKV', T * 3 * d); iPE = ar.alloc('PE', NP * d); iP = ar.alloc('P', NP * d)
    iSC = ar.alloc('SC', H * T * T); iATT = ar.alloc('ATT', T * d)
    arena = np.zeros(ar.used(), np.float32)
    arena[iQKV:iQKV + T * 3 * d] = qkv.ravel()

    p = Plan()
    p.dispatch('pe_gen', 64, NP * d, [T, d, iPE])                       # 1 positional table
    ops.gemm12(p, iPE, NP, mm_pos, iP)                                   # 2 P = PE Wpos^T + b
    n_after_P = len(p.ops)
    push = [T, d, H, dh, iQKV, iP, iSC, pbuO, pbvO]
    JB = (T + 3) // 4
    if scores == 'att_scores6':                                          # 2 rows x 4 keys per invocation
        IB = (T + 1) // 2
        p.add('att_scores6', 64, max(MIN_WORKGROUPS, (H * IB * JB + 63) // 64), 1, push + [0] * 23)
    else:
        p.dispatch(scores, 64, H * T * JB, push + [0] * 23)              # 3 scores (scaled)
    p.dispatch('att_softmax', 64, H * T, push + [iATT] + [0] * 22)      # 4 softmax rows
    IB = (T + 1) // 2; nv = dh // 4
    groups = max(MIN_WORKGROUPS, (H * IB * nv + 63) // 64)
    p.add('att_pv6', 64, groups, 1, push + [iATT] + [0] * 22)           # 5 S @ V

    P_ref = (pos_table(T, d) @ Wpos.T + bpos).astype(np.float32)
    ok_P = check.bisect(f'T={T}', p, arena,
                        [(n_after_P - 1, 'P (pos projection)', iP, P_ref, 1e-4)], weights=path) is None
    want = att_ref(qkv, P_ref, pbu, pbv, T, d, H, dh)
    ok = check.op_test(f'attention output T={T} d={d} H={H}', p, arena, iATT, want.ravel(), 2e-4, weights=path)[0]
    return ok_P and ok


if __name__ == '__main__':
    scores = sys.argv[1] if len(sys.argv) > 1 else 'att_scores5'
    results = [run_config(T=8, d=64, H=2, scores=scores), run_config(T=13, d=176, H=4, scores=scores),
               run_config(T=21, d=176, H=4, scores=scores)]
    print('ALL ATTENTION TESTS PASS' if all(results) else 'ATTENTION TESTS FAILED')
    raise SystemExit(0 if all(results) else 1)
