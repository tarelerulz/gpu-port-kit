"""Fused silu (GPUPORT_ASM_SILU=1) vs separate silu op (=0): logits, transcript, time. usage: python3 tools/silu_fuse_check.py [clip] [reps]"""
import os, sys
import numpy as np
os.environ['GPUPORT_ASM'] = '1'
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, 'tests'))
import nemo8_rebuild as nb
from gpuport import server
from gpuport.config import WORK_DIR
clip = sys.argv[1] if len(sys.argv) > 1 else 'clip10s'; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
nr = nb.load_nemo_ref(); wt, L = nb.prepare(nr)
wpath = wt.save(os.path.join(WORK_DIR, 'nemo8_kit'))
pcm = np.fromfile(os.path.join(nb.SPEECH_APP, clip + '.pcm'), '<i2'); x = pcm.astype(np.float32) / 32768.0
res = {}
for mode in ('0', '1'):
    os.environ['GPUPORT_ASM_SILU'] = mode
    plan, ar, R, T = nb.build(len(pcm), L, optimized=True)
    srv = server.ExecServer(wpath, nb.SHADERS).start()
    srv.run(plan, x, R['LOG'], T * (L['V'] + 1))
    ms = []
    for _ in range(reps):
        out, m = srv.run(plan, x, R['LOG'], T * (L['V'] + 1)); ms.append(m)
    srv.proc.stdin.close(); srv.proc.wait(timeout=20)
    lg = out.reshape(T, L['V'] + 1)
    res[mode] = (lg.copy(), nb.collapse(lg, L['tokens'], L['V']), sorted(ms)[len(ms) // 2], len(plan.ops) if hasattr(plan, 'ops') else -1)
a, b = res['0'], res['1']
print(f'separate silu: median {a[2]:.0f} ms | fused: median {b[2]:.0f} ms')
print('max abs logit diff', float(np.abs(a[0] - b[0]).max()), ' NaN in fused:', bool(np.isnan(b[0]).any()))
print('argmax agreement', float((a[0].argmax(1) == b[0].argmax(1)).mean()))
print('same text:', a[1] == b[1]); print('fused:', b[1])
