"""Memory guard over the whole nemo8 plan (added 2026-11-12).
1. canary: no dispatch may write outside the plan's regions; reports regions that are not filled completely.
2. NaN poison: everything the plan does not write itself is NaN. The logits must be the same as in a clean
   (zero-filled) run, i.e. the result must not depend on what was in the arena before. That matters on the
   resident server, where the arena keeps the previous request's data.
usage: python3 tests/nemo8_guard.py [clip] [int8]     (clip5s default, or clip10s; int8 = int8 linears and head)
GPUPORT_I8KM=0 with int8 checks the older row-major int8 kernels."""
import os
import sys
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport import check
from gpuport.run import run
from gpuport.config import WORK_DIR
from gpuport.nemo8 import load_nemo_ref, prepare, build, SPEECH_APP, SHADERS, ARENA_FLOATS

args = sys.argv[1:]
int8 = 'int8' in args
args = [a for a in args if a != 'int8']
clip = args[0] if args else 'clip5s'
EXEC_BIG = os.path.join(KIT, 'tools', 'exec_big')
nr = load_nemo_ref(); w, L = prepare(nr, int8=int8)
wpath = w.save(os.path.join(WORK_DIR, 'nemo8_int8' if int8 else 'nemo8_kit'))
pcm = np.fromfile(os.path.join(SPEECH_APP, clip + '.pcm'), '<i2').astype(np.float32) / 32768.0
plan, ar, R, T = build(len(pcm), L)
V1 = L['V'] + 1
kw = dict(weights=wpath, shader_dir=SHADERS, exec_bin=EXEC_BIG, arena_floats=ARENA_FLOATS)
clean, _ = run(plan, pcm, R['LOG'], T * V1, **kw)
# FR has a padded row stride (516 for 512 samples): 4 floats per row stay unwritten on purpose
# Q (int8 path): packed activations, sized for the widest matmul, so smaller ones fill it in part
# PT, ST, GT (matmul attention): rows padded to a multiple of 12, the padding is not always written
PART = ('TK', 'FR', 'Q', 'PT', 'ST', 'GT', 'X', 'HB', 'H2', 'ATT', 'CV', 'QU', 'QV', 'KT')   # k-major rows: padding frames
writes = [n for n in ar.regions if n != 'XIN' and n not in PART]
scratch = [n for n in ar.regions if n in PART]
ok1, out, probs = check.guard_run(f'nemo8{" int8" if int8 else ""} {clip} canary', plan, ar, {'XIN': pcm}, writes, scratch, **kw)
stray = [p for p in probs if 'outside' in p or 'stray' in p]
same1 = np.array_equal(out['LOG'], clean)
print(f'   stray writes: {len(stray)}; logits identical to the clean run: {same1}')
ok2, out2, probs2 = check.guard_run(f'nemo8{" int8" if int8 else ""} {clip} NaN poison', plan, ar, {'XIN': pcm}, writes, scratch, poison=True, **kw)
same2 = np.array_equal(out2['LOG'], clean)
print(f'   logits identical to the clean run with NaN around and inside unwritten memory: {same2}')
good = ok1 and ok2 and same1 and same2
print('NEMO8 GUARD PASS' if good else 'NEMO8 GUARD FAIL')
raise SystemExit(0 if good else 1)
