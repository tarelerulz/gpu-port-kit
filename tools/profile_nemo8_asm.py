"""Per-op-type profile of the nemo8 plan WITH the hand-written QPU matmuls (tools/exec_hybrid_prof: every op waited on,
so times include sync cost and the sum exceeds the normal run). Reports the normal run time and the serialized one.
usage: GPUPORT_ASM=1 python3 tools/profile_nemo8_asm.py [clip] [repeats]"""
import collections, os, sys, time
import numpy as np
os.environ['GPUPORT_ASM'] = '1'
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, 'tests'))
import nemo8_rebuild as nb
from gpuport import server
from gpuport.config import WORK_DIR
clip = sys.argv[1] if len(sys.argv) > 1 else 'clip10s'
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = os.path.join(WORK_DIR, 'nemo8_asm_prof.txt')
nr = nb.load_nemo_ref(); wt, L = nb.prepare(nr)
wpath = wt.save(os.path.join(WORK_DIR, 'nemo8_kit'))
pcm = np.fromfile(os.path.join(nb.SPEECH_APP, clip + '.pcm'), '<i2')
plan, ar, R, T = nb.build(len(pcm), L, optimized=True)
x = pcm.astype(np.float32) / 32768.0
def go(profile):
    if profile:
        os.environ['EXEC_HPROF'] = OUT; server.EXEC_HYBRID = os.path.join(KIT, 'tools', 'exec_hybrid_prof')
    else:
        os.environ.pop('EXEC_HPROF', None); server.EXEC_HYBRID = os.path.join(KIT, 'tools', 'exec_hybrid')
    srv = server.ExecServer(wpath, nb.SHADERS).start()
    srv.run(plan, x, R['LOG'], T * (L['V'] + 1))                       # warm-up
    ms = []
    for _ in range(reps):
        out, m = srv.run(plan, x, R['LOG'], T * (L['V'] + 1)); ms.append(m)
    srv.proc.stdin.close(); srv.proc.wait(timeout=20)
    return sorted(ms)[len(ms) // 2]
normal = go(False)
ser = go(True)
rows, tot, req = [], 0.0, 0.0
for l in open(OUT):
    if l.startswith('#TOTAL_OPS'): tot = float(l.split()[1])
    elif l.startswith('#REQUEST'): req = float(l.split()[1])
    else:
        p = l.rsplit(' ', 2); rows.append((p[0], int(p[1]), float(p[2])))
print(f'{clip}: normal run median {normal:.0f} ms; serialized run {req:.0f} ms (sum of op times {tot:.0f} ms)')
grp = collections.defaultdict(lambda: [0, 0.0])
for k, n, ms in rows:
    g = k.split()[0]; grp[g][0] += n; grp[g][1] += ms
print('\nby op type (serialized, one run):')
for g, (n, ms) in sorted(grp.items(), key=lambda a: -a[1][1]):
    print(f'  {g:18s} {n:4d} ops {ms:8.1f} ms {100*ms/tot:5.1f}%  {ms/n:6.2f} ms/op')
print('\nqpu ops by shape:')
for k, n, ms in sorted(rows, key=lambda a: -a[2]):
    if k.startswith('qpu'): print(f'  {k:32s} {n:4d} x {ms/n:6.2f} ms = {ms:7.1f}')
