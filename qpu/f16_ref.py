"""Hybrid vs the Mesa plan on f16-ROUNDED weights (same rounding the QPU kernels use): if these agree, the QPU path is exact and larger differences to the f32 plan come from weight rounding."""
import os, sys
import numpy as np
KIT = '/home/tarel/gpu-port-kit'; sys.path.insert(0, KIT)
os.environ['GPUPORT_ASM'] = '1'; os.environ['GPUPORT_ASM_GATE'] = 'off'
from gpuport import weights as wmod, ops
from gpuport.nemo8 import load_nemo_ref, prepare, build, SPEECH_APP, SHADERS
from gpuport.server import ExecServer
from gpuport.config import WORK_DIR
nr = load_nemo_ref(); w, L = prepare(nr); wpath = w.save(os.path.join(WORK_DIR, 'nemo8_asm')); V = L['V']
f16 = lambda a: np.asarray(a, np.float32).astype(np.float16).astype(np.float32)
o_put, o_ptk = wmod.Weights.put_mm, ops.pos_table_km
wmod.Weights.put_mm = lambda self, Wm, bias=None, pad=12: o_put(self, f16(Wm), bias, pad)
ops.pos_table_km = lambda *a, **k: (lambda r: (f16(r[0]), r[1]))(o_ptk(*a, **k))
os.environ['GPUPORT_ASM'] = '0'
w16, L16 = prepare(nr); p16 = w16.save(os.path.join(WORK_DIR, 'nemo8_f16ref'))
base = np.fromfile(os.path.join(SPEECH_APP, 'clip10s.pcm'), '<i2').astype(np.float32) / 32768.0
s32 = ExecServer(wpath, SHADERS).start(); s16 = ExecServer(p16, SHADERS).start()
for n in [int(a) for a in sys.argv[1:]]:
    x = np.resize(base, n)
    os.environ['GPUPORT_ASM'] = '0'; pr, _, Rr, T = build(n, L); p6, _, R6, _ = build(n, L16)
    os.environ['GPUPORT_ASM'] = '1'; ph, _, Rh, _ = build(n, L)
    lr = s32.run(pr, x, Rr['LOG'], T * (V + 1))[0].reshape(T, V + 1)
    l6 = s16.run(p6, x, R6['LOG'], T * (V + 1))[0].reshape(T, V + 1)
    os.environ['GPUPORT_ASM'] = '1'; lh = ExecServer(wpath, SHADERS).start().run(ph, x, Rh['LOG'], T * (V + 1))[0].reshape(T, V + 1)
    d = lambda a, b: f'max {np.abs(a - b).max():.3g}, argmax {100 * (a.argmax(1) == b.argmax(1)).mean():.1f}%'
    print(f'T={T}: hybrid vs f32 plan: {d(lh, lr)} | f16-rounded Mesa plan vs f32 plan: {d(l6, lr)} | hybrid vs f16-rounded Mesa plan: {d(lh, l6)}', flush=True)
s32.stop(); s16.stop()
