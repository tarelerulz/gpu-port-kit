"""Weight layout tests: each test packs weights the way the blob or shader expects, runs the op on the GPU
(or checks the layout directly) and compares with numpy.
Covers: matmul layout (put_mm), 8-bit quantized weights read by speech-app's nemo_ref.Blob.qlin and fed to the
GPU matmul, LayerNorm gamma/beta, GLU, depthwise conv [k][d] layout.
Run: python3 ~/gpu-port-kit/tests/test_weight_layouts.py   (needs the GPU; the quantized test needs ~/speech-app/port)"""
import importlib.util
import os
import sys
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport import Arena, Weights, Plan, ops, check
from gpuport.arena import roundup
from gpuport.config import WORK_DIR, MIN_WORKGROUPS

rng = np.random.default_rng(7)
results = []


def record(name, ok):
    results.append((name, ok))
    print(('PASS ' if ok else 'FAIL ') + name)


def save_blob(w, name):
    return w.save(os.path.join(WORK_DIR, 'layout_' + name))


def blob_floats(w):
    return np.concatenate(w.parts).astype(np.float32)


# ---- 1. put / put_mm layout (no GPU) -------------------------------------------------------------
def test_put_alignment_and_mm_layout():
    w = Weights()
    w.put(np.ones(5, np.float32))                 # misalign on purpose
    W = rng.standard_normal((25, 30)).astype(np.float32)
    b = rng.standard_normal(25).astype(np.float32)
    mm = w.put_mm(W, b)
    ok = all(mm[k] % 16 == 0 for k in ('wo', 'bo'))
    flat = blob_floats(w)
    NP, K, N = mm['np'], mm['k'], mm['n']
    wt = flat[mm['wo']:mm['wo'] + K * NP].reshape(K, NP)
    ok &= np.array_equal(wt[:, :N], W.T)          # W^T, row k = input index
    ok &= np.all(wt[:, N:] == 0)                  # padding columns are zero
    bias = flat[mm['bo']:mm['bo'] + NP]
    ok &= np.array_equal(bias[:N], b) and np.all(bias[N:] == 0)
    ok &= NP % 12 == 0 and NP >= N
    record('put/put_mm: 16-float alignment, W^T [K][NP], zero padding, bias', bool(ok))


# ---- 2. speech-app 8-bit quantized reader -> dequantized weights -> GPU matmul ----------------------------
def load_nemo_ref():
    path = os.path.expanduser('~/speech-app/port/nemo_ref.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location('nemo_ref', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pack_q8(q, scale, bias):
    """Bytes in the layout nemo_ref.Blob.qlin(bits=8) reads: rows of kp=roundup(k,16) int8, then
    scale[n] f32 (16-aligned), then bias[n] f32 (16-aligned)."""
    n, k = q.shape
    kp = roundup(k, 16)
    buf = np.zeros((n, kp), np.int8); buf[:, :k] = q
    out = bytearray(buf.tobytes())
    out += b'\0' * (roundup(len(out), 16) - len(out)); out += scale.astype('<f4').tobytes()
    out += b'\0' * (roundup(len(out), 16) - len(out)); out += bias.astype('<f4').tobytes()
    return bytes(out)


def test_q8_reader_and_gpu():
    nr = load_nemo_ref()
    if nr is None:
        print('SKIP quantized test: ~/speech-app/port/nemo_ref.py not found')
        return
    n, k, M = 40, 50, 37
    q = rng.integers(-127, 128, size=(n, k)).astype(np.int8)
    scale = rng.uniform(0.005, 0.03, n).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    Bl = nr.Blob.__new__(nr.Blob)
    Bl.b = pack_q8(q, scale, bias); Bl.p = 0
    W, bread = Bl.qlin(n, k, 8)                  # the speech-app reader
    ref_W = q.astype(np.float32) * scale[:, None]
    ok_reader = np.allclose(W, ref_W, rtol=0, atol=1e-7) and np.array_equal(bread, bias)
    record('8-bit reader: qlin returns q*scale[row] and bias', bool(ok_reader))

    A = rng.standard_normal((M, k)).astype(np.float32)
    w = Weights(); mm = w.put_mm(W, bread)
    path = save_blob(w, 'q8')
    ar = Arena(); ia = ar.alloc('A', M * k); io = ar.alloc('O', M * n)
    arena = np.zeros(ar.used(), np.float32); arena[ia:ia + M * k] = A.ravel()
    p = Plan(); ops.gemm12(p, ia, M, mm, io)
    want = A @ W.T + bread
    ok = check.op_test('8-bit weights -> GPU matmul (n=40, k=50, M=37)', p, arena, io, want.ravel(), 2e-4, weights=path)[0]
    record('8-bit weights -> GPU matmul matches numpy', ok)


# ---- 3. LayerNorm: gamma and beta stored back to back (beta at gamma + d) --------------------------
def test_layernorm():
    T, d = 10, 48
    x = rng.standard_normal((T, d)).astype(np.float32) * 3 + 1
    g = rng.standard_normal(d).astype(np.float32)
    bt = rng.standard_normal(d).astype(np.float32)
    w = Weights()
    gO = w.put(np.concatenate([g, bt]))
    path = save_blob(w, 'ln')
    ar = Arena(); ix = ar.alloc('X', T * d); oy = ar.alloc('Y', T * d)
    arena = np.zeros(ar.used(), np.float32); arena[ix:ix + T * d] = x.ravel()
    p = Plan(); ops.ln(p, T, d, ix, oy, gO)
    mu = x.mean(1, keepdims=True); var = x.var(1, keepdims=True)
    want = (x - mu) / np.sqrt(var + 1e-5) * g + bt
    ok = check.op_test('LayerNorm (gamma/beta layout)', p, arena, oy, want.ravel(), 1e-4, weights=path)[0]
    record('LayerNorm: gamma at O, beta at O+d, eps 1e-5', ok)


# ---- 4. GLU: input rows are 2d wide, [a | b] -> a * sigmoid(b) ------------------------------------
def test_glu():
    T, d = 12, 40
    x = rng.standard_normal((T, 2 * d)).astype(np.float32)
    ar = Arena(); ix = ar.alloc('X', T * 2 * d); oy = ar.alloc('Y', T * d)
    arena = np.zeros(ar.used(), np.float32); arena[ix:ix + T * 2 * d] = x.ravel()
    p = Plan(); ops.glu(p, T, d, ix, oy)
    a, b = x[:, :d], x[:, d:]
    want = a / (1 + np.exp(-b))
    ok = check.op_test('GLU', p, arena, oy, want.ravel(), 1e-5)[0]
    record('GLU: a in cols [0,d), b in [d,2d), out = a*sigmoid(b)', ok)


# ---- 5. Depthwise conv: weights stored transposed [k][d], bias [d], centered zero padding -----------
def test_depthwise_conv():
    T, d, k = 20, 64, 5
    h = k // 2
    x = rng.standard_normal((T, d)).astype(np.float32)
    Wdw = rng.standard_normal((d, k)).astype(np.float32) * 0.5    # nn layout [d][k]
    bdw = rng.standard_normal(d).astype(np.float32)
    w = Weights()
    wO = w.put(Wdw.T.copy())                                       # stored [k][d]
    bO = w.put(bdw)
    path = save_blob(w, 'dw')
    ar = Arena(); ix = ar.alloc('X', T * d); oy = ar.alloc('Y', T * d)
    arena = np.zeros(ar.used(), np.float32); arena[ix:ix + T * d] = x.ravel()
    p = Plan()
    groups = max(MIN_WORKGROUPS, (d // 4 + 63) // 64)
    p.add('dw4', 64, groups, T, [T, d, k, ix, oy, wO, bO])
    want = np.zeros((T, d), np.float32)
    for t in range(T):
        acc = bdw.copy()
        for j in range(k):
            s = t + j - h
            if 0 <= s < T:
                acc += Wdw[:, j] * x[s]
        want[t] = acc
    ok = check.op_test('depthwise conv k=5', p, arena, oy, want.ravel(), 1e-4, weights=path)[0]
    record('depthwise conv: weights [k][d], bias [d], centered, zero padded', ok)


if __name__ == '__main__':
    test_put_alignment_and_mm_layout()
    test_q8_reader_and_gpu()
    test_layernorm()
    test_glu()
    test_depthwise_conv()
    failed = [n for n, ok in results if not ok]
    print('ALL LAYOUT TESTS PASS' if not failed else f'FAILED: {failed}')
    raise SystemExit(1 if failed else 0)
