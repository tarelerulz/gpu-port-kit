import os, sys, numpy as np, subprocess
KIT = '/home/tarel/gpu-port-kit'; sys.path.insert(0, KIT)
os.environ['GPUPORT_ASM'] = '1'
from gpuport.nemo8 import load_nemo_ref, prepare, build, SPEECH_APP, SHADERS
from gpuport.server import ExecServer
from gpuport.config import WORK_DIR
nr = load_nemo_ref(); w, L = prepare(nr); wpath = w.save(os.path.join(WORK_DIR, 'nemo8_asm'))
x = np.fromfile(os.path.join(SPEECH_APP, 'clip10s.pcm'), '<i2').astype(np.float32) / 32768.0
srv = ExecServer(wpath, SHADERS).start()
p, a, R, T = build(len(x), L)
first = None; bad = 0; ts = []
for i in range(int(sys.argv[1])):
    out, ms = srv.run(p, x, R['LOG'], T * (L['V'] + 1)); ts.append(ms)
    if first is None: first = out.copy()
    elif not np.array_equal(out, first): bad += 1
fds = len(os.listdir(f'/proc/{srv.proc.pid}/fd'))
print(f'{len(ts)} runs: outputs identical to run 1 in {len(ts) - 1 - bad}/{len(ts) - 1}, median {np.median(ts):.0f} ms, max {max(ts):.0f}, server open fds {fds}')
srv.stop()
