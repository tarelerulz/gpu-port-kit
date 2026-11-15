import os, sys, numpy as np
KIT = '/home/tarel/gpu-port-kit'; sys.path.insert(0, KIT)
os.environ['GPUPORT_ASM'] = '1'
from gpuport.nemo8 import load_nemo_ref, prepare, build, SPEECH_APP, SHADERS
from gpuport.server import ExecServer
from gpuport.config import WORK_DIR
nr = load_nemo_ref(); w, L = prepare(nr); wpath = w.save(os.path.join(WORK_DIR, 'nemo8_asm'))
x = np.fromfile(os.path.join(SPEECH_APP, 'clip10s.pcm'), '<i2').astype(np.float32) / 32768.0
srv = ExecServer(wpath, SHADERS).start(); res = {}
for exp in ['none', 'tiny', 'full']:
    os.environ['GPUPORT_ASM_EXP'] = '' if exp == 'full' else exp
    p, a, R, T = build(len(x), L)
    t = [srv.run(p, x, R['LOG'], T * (L['V'] + 1))[1] for _ in range(6)]
    res[exp] = float(np.median(t)); print(f'{exp:5s}: {sum(o.shader == "qpu_mm" for o in p.ops):2d} asm ops, median {res[exp]:.0f} ms', flush=True)
print(f"hand-off overhead per asm op ~ (tiny - none) / 32 = {(res['tiny'] - res['none']) / 32:.2f} ms")
srv.stop()
