"""Create the folder for a new model port, with the steps from gpu-porting-guide.md as templates.
usage: python3 ~/gpu-port-kit/scaffold.py <dest_dir> <model_name>
Writes: ref.py (numpy reference, step 1), inventory.md (step 2), test_ops.py (step 3, one op at a time),
build_plan.py (steps 4-5), NOTES.md (step 7, dated log)."""
import os
import sys
from datetime import date

KIT = os.path.dirname(os.path.abspath(__file__))

TEMPLATES = {
'ref.py': '''"""{name}: numpy float reference (step 1). Must match the CPU output on 2 clips (2 s phrase, 10 s clip).
Compare collapsed tokens and argmax, and every stage (front end, one layer, head), not only the final text."""
import numpy as np

def forward(x):
    """x: model input. Return the logits / output per frame. Fill in from the model format."""
    raise NotImplementedError
''',
'inventory.md': '''# {name}: inventory (step 2)

Date: {today}

| op type | count | shape (in -> out) | weight dtype | weight layout in blob |
|---|---|---|---|---|

Notes on the blob: where the token list ends, read position advanced past it, normalisation constants after it.
Fit rule: f32 weights and large matmuls are worth porting; 8/4-bit weights and tiny per-token steps are not.
''',
'test_ops.py': '''"""{name}: one test per op type, each against numpy (step 3). Run: python3 test_ops.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/gpu-port-kit'))
from gpuport import Arena, Weights, Plan, ops, check

rng = np.random.default_rng(1)

def test_matmul():
    M, K, N = 37, 64, 40
    A = rng.standard_normal((M, K)).astype(np.float32)
    Wm = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    b = rng.standard_normal(N).astype(np.float32)
    w = Weights(); mm = w.put_mm(Wm, b)
    wpath = w.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work', 'matmul'))
    ar = Arena(); ia = ar.alloc('A', M * K); io = ar.alloc('O', M * N)
    arena = np.zeros(ar.used(), np.float32); arena[ia:ia + M * K] = A.ravel()
    p = Plan(); ops.gemm12(p, ia, M, mm, io)
    check.op_test('matmul', p, arena, io, (A @ Wm.T + b).ravel(), 1e-3, weights=wpath)

if __name__ == '__main__':
    test_matmul()
    # add one test per new op type here; add the shader's own .comp/.spv to the shader dir first
''',
'build_plan.py': '''"""{name}: plan builder (steps 4-5). Regions and weights come from the kit; check with the linter:
  python3 ~/gpu-port-kit/gpuport/plan.py work/{name}.plan ~/speech-app/port/gpu   (or your shader dir)"""
import os, sys
sys.path.insert(0, os.path.expanduser('~/gpu-port-kit'))
from gpuport import Arena, Weights, Plan, ops, check
from gpuport.run import run

def build(n_samples):
    ar = Arena()
    # ar.alloc('X', T * d)  -> one region per tensor; the kit adds the 64-float alignment and the 131072-float gap
    p = Plan()
    # ops.gemm12(p, ar['X'], M, mm, ar['Y'])
    # p.dispatch('my_shader', 128, n_threads, [n, ar['X'], ar['Y']])
    return p, ar

if __name__ == '__main__':
    p, ar = build(16000)
    print(len(p), 'ops; arena', ar.used(), 'floats')
    # bisect: check.bisect(label, p, arena_init, [(op_index, name, out_off, numpy_value, tol), ...])
''',
'NOTES.md': '''# {name} GPU port: log

Started {today}. Method: <notes>/gpu-porting-guide.md and the kit at ~/gpu-port-kit (README).

## Status
- [ ] 1 numpy reference matches CPU (2 clips, tokens identical)
- [ ] 2 inventory
- [ ] 3 per-op tests vs numpy
- [ ] 4 plan builder + linter clean
- [ ] 5 prefix checks: first wrong stage fixed
- [ ] 6 timing per op type, one change at a time
- [ ] 7 accuracy re-checked after every change

## Rounds
(date, change, numbers, verdict)
''',
}

def main():
    if len(sys.argv) != 3:
        print(__doc__); return 2
    dest, name = sys.argv[1], sys.argv[2]
    if os.path.exists(dest) and os.listdir(dest):
        print(f'{dest} exists and is not empty; pick a new folder'); return 1
    os.makedirs(os.path.join(dest, 'work'), exist_ok=True)
    for fn, body in TEMPLATES.items():
        with open(os.path.join(dest, fn), 'w') as f:
            f.write(body.format(name=name, today=date.today().isoformat()))
    print('created', dest, 'with', ', '.join(TEMPLATES))
    return 0

if __name__ == '__main__':
    sys.exit(main())
