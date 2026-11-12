"""Milestone test: FF1 matmuls (ff1_1, ff2_1 of every layer) on the hand-written QPU kernel inside the real nemo8 plan. 2026-11-12.
Reference = the normal plan (Mesa kernels) on the normal server; hybrid = same weights file, GPUPORT_ASM=1 plan on exec_hybrid."""
import os, sys, time
import numpy as np
KIT = '/home/tarel/gpu-port-kit'
sys.path.insert(0, KIT)
os.environ['GPUPORT_ASM'] = '1'
from gpuport.nemo8 import load_nemo_ref, prepare, build, collapse, SPEECH_APP, SHADERS
from gpuport.server import ExecServer
from gpuport.config import WORK_DIR

nr = load_nemo_ref(); w, L = prepare(nr)
wpath = w.save(os.path.join(WORK_DIR, 'nemo8_asm'))
V = L['V']
clip = sys.argv[1] if len(sys.argv) > 1 else 'clip10s'
x = np.fromfile(os.path.join(SPEECH_APP, clip + '.pcm'), '<i2').astype(np.float32) / 32768.0
n = len(x)
os.environ['GPUPORT_ASM'] = '0'
pr, ar, Rr, T = build(n, L)
ref = ExecServer(wpath, SHADERS).start()
os.environ['GPUPORT_ASM'] = '1'
ph, ah, Rh, _ = build(n, L)
print('asm ops in plan:', sum(o.shader == 'qpu_mm' for o in ph.ops), 'of', len(ph.ops), 'ops (reference', len(pr.ops), ')')
hyb = ExecServer(wpath, SHADERS).start()
lr, _ = ref.run(pr, x, Rr['LOG'], T * (V + 1)); lr = lr.reshape(T, V + 1)
lh, _ = hyb.run(ph, x, Rh['LOG'], T * (V + 1)); lh = lh.reshape(T, V + 1)
print(f'T={T}: max logit diff {np.abs(lr - lh).max():.3g}, argmax agree {100 * (lr.argmax(1) == lh.argmax(1)).mean():.2f}%, finite {bool(np.isfinite(lh).all())}')
print('ref text   :', collapse(lr, L['tokens'], V)); print('hybrid text:', collapse(lh, L['tokens'], V))
tr, th = [], []
for i in range(6):
    tr.append(ref.run(pr, x, Rr['LOG'], T * (V + 1))[1]); th.append(hyb.run(ph, x, Rh['LOG'], T * (V + 1))[1])
print(f'server ms, median of 6: reference {np.median(tr):.0f}, hybrid {np.median(th):.0f}  (all: ref {[round(t) for t in tr]}, hyb {[round(t) for t in th]})')
ref.stop(); hyb.stop()
