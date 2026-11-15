"""Packed-weight matmul kernels vs the float32 k-major kernel (2026-11-12). One 251-row block per shape, k-major in and out.
gemm12av4o1 = float32 weights; gemm12k16 = half-float pairs; gemm12k8 = signed bytes + per-output scale.
usage: python3 tools/bench_packed.py"""
import os, sys
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport import Arena, Weights, Plan, check
from gpuport.timing import time_op
from gpuport.config import WORK_DIR, merged_shader_dir
from gpuport.run import run
SH = merged_shader_dir()
rng = np.random.default_rng(6)
for name, M, K, N in [('FF1 K=176 N=704', 251, 176, 704), ('FF2 K=704 N=176', 251, 704, 176), ('conv2 K=1584 N=176', 251, 1584, 176)]:
    Mp = 252; NP = (N + 11) // 12 * 12
    q = rng.integers(-127, 128, (N, K)).astype(np.int8); sc = (rng.random(N) * 0.002 + 0.0005).astype(np.float32)
    Wf = q.astype(np.float32) * sc[:, None]; b = (rng.standard_normal(N) * 0.1).astype(np.float32)
    A = (rng.standard_normal((M, K)) * 0.3).astype(np.float32)
    want = (A @ Wf.T + b).T                                    # [N][M]
    w = Weights(); mm = w.put_mm(Wf, b)
    W16 = np.zeros((K, NP), np.float16); W16[:, :N] = Wf.T; o16 = w.put(W16.view(np.uint16).astype(np.uint32).reshape(K, NP // 2, 2).dot([1, 65536]).astype(np.uint32).view(np.float32))
    Q = np.zeros((K, NP), np.int8); Q[:, :N] = q.T; o8 = w.put(Q.view(np.uint8).astype(np.uint32).reshape(K, NP // 4, 4).dot([1, 256, 65536, 16777216]).astype(np.uint32).view(np.float32))
    scp = np.zeros(NP, np.float32); scp[:N] = sc; osc = w.put(scp)
    path = w.save(os.path.join(WORK_DIR, 'bench_packed'))
    ar = Arena(); ia = ar.alloc('A', K * Mp); io = ar.alloc('O', N * Mp)
    arena = np.zeros(ar.used(), np.float32); Ak = np.zeros((K, Mp), np.float32); Ak[:, :M] = A.T; arena[ia:ia + K * Mp] = Ak.ravel()
    g = max(24, ((NP // 12) * 63 + 255) // 256)
    for kern, wo, extra in [('gemm12av4o1', mm['wo'], []), ('gemm12k16', o16, []), ('gemm12k8', o8, [0, osc])]:
        p = Plan().add(kern, 256, g, 1, [M, N, K, NP, ia, 1, Mp, wo, mm['bo'], io, 1, Mp, 0] + extra)
        got = run(p, arena, io, N * Mp, weights=path, shader_dir=SH)[0].reshape(N, Mp)[:, :M]
        err = np.abs(got - want).max()
        ms = time_op(p, arena, io, 8, repeats=20, weights=path, shader_dir=SH)
        print(f'{name:20s} {kern:12s} max err {err:.2e}  {ms:7.2f} ms  {M * K * N / ms / 1e6:.2f} GMAC/s')
