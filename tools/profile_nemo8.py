"""Per-dispatch GPU profile of the nemo8 full-GPU plan (tools/exec_prof, Vulkan timestamps).
Prints time by op type, by stage (front end, each layer, head), and the top dispatches.
usage: python3 tools/profile_nemo8.py [clip]   (clip = clip10s or clip5s; default clip10s)"""
import collections
import os
import sys
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT); sys.path.insert(0, os.path.join(KIT, 'tests'))
import nemo8_rebuild as nb
from gpuport.run import run
from gpuport.config import WORK_DIR

clip = sys.argv[1] if len(sys.argv) > 1 else 'clip10s'
PROF = os.path.join(KIT, 'tools', 'exec_prof')
nr = nb.load_nemo_ref(); w, L = nb.prepare(nr)
wpath = w.save(os.path.join(WORK_DIR, 'nemo8_kit'))
pcm = np.fromfile(os.path.join(nb.SPEECH_APP, clip + '.pcm'), '<i2')
plan, ar, R, T = nb.build(len(pcm), L, optimized=True)
prof_file = os.path.join(WORK_DIR, 'nemo8_prof.txt')
run(plan, pcm.astype(np.float32) / 32768.0, R['LOG'], T * (L['V'] + 1), weights=wpath, shader_dir=nb.SHADERS,
    exec_bin=PROF, arena_floats=nb.ARENA_FLOATS, prof=prof_file)
times = [float(l.split()[2]) for l in open(prof_file)]
names = [l.split()[1] for l in open(prof_file)]
total = sum(times)
print(f'{clip}: {len(times)} dispatches, sum of GPU times {total:.0f} ms')
by_op = collections.defaultdict(lambda: [0, 0.0])
for n, t in zip(names, times):
    by_op[n][0] += 1; by_op[n][1] += t
print('\nby shader:')
for n, (c, t) in sorted(by_op.items(), key=lambda x: -x[1][1]):
    print(f'  {n:14s} {c:4d} dispatches {t:8.1f} ms  {100*t/total:5.1f}%')
marks = plan.marks + [('end', len(times))]
print('\nby stage:')
for (name, a), (_, b) in zip(marks[:-1], marks[1:]):
    t = sum(times[a:b])
    print(f'  {name:10s} {t:8.1f} ms  {100*t/total:5.1f}%')
print('\ntop 12 dispatches:')
for i in sorted(range(len(times)), key=lambda i: -times[i])[:12]:
    print(f'  op {i:4d} {names[i]:14s} {times[i]:7.2f} ms')
