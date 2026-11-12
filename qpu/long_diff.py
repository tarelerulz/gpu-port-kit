import os, sys
import numpy as np
KIT = '/home/tarel/gpu-port-kit'; sys.path.insert(0, KIT)
os.environ['GPUPORT_ASM'] = '1'; os.environ['GPUPORT_ASM_GATE'] = 'off'
from gpuport.nemo8 import load_nemo_ref, prepare, build, SPEECH_APP
from gpuport.server import ExecServer
from gpuport.config import WORK_DIR
nr = load_nemo_ref(); w, L = prepare(nr); wpath = w.save(os.path.join(WORK_DIR, 'nemo8_asm')); V = L['V']
base = np.fromfile(os.path.join(SPEECH_APP, 'clip10s.pcm'), '<i2').astype(np.float32) / 32768.0
os.environ['GPUPORT_ASM'] = '0'; srv0 = ExecServer(wpath, os.path.join(KIT, 'shaders') if False else __import__('gpuport.nemo8', fromlist=['SHADERS']).SHADERS).start()
os.environ['GPUPORT_ASM'] = '1'; srv1 = ExecServer(wpath, __import__('gpuport.nemo8', fromlist=['SHADERS']).SHADERS).start()
for n in [int(a) for a in sys.argv[1:]]:
    x = np.resize(base, n)
    os.environ['GPUPORT_ASM'] = '0'
    pr, _, Rr, T = build(n, L)
    os.environ['GPUPORT_ASM'] = '1'; ph, _, Rh, _ = build(n, L)
    lr = srv0.run(pr, x, Rr['LOG'], T * (V + 1))[0].reshape(T, V + 1)
    lh = srv1.run(ph, x, Rh['LOG'], T * (V + 1))[0].reshape(T, V + 1)
    df = np.abs(lh - lr); fm = df.max(1)
    print(f'T={T}: max {df.max():.3g} at frame {int(fm.argmax())}, mean {df.mean():.4f}, p99.9 {np.percentile(df, 99.9):.3g}; frames with max>0.5: {np.where(fm > 0.5)[0].tolist()[:20]}; argmax agree {100 * (lh.argmax(1) == lr.argmax(1)).mean():.1f}%', flush=True)
srv0.stop(); srv1.stop()
