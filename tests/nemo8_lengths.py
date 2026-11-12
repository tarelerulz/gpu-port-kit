"""Default nemo8 plan vs the exact plan (old shaders, bit-identical to production) at many clip lengths (2026-11-12).
Covers frame counts of every remainder mod 4, odd and even conv sizes, and clips over 10 s (split row blocks).
Pass = same tokens, same argmax in every frame, small logit difference.
usage: python3 tests/nemo8_lengths.py"""
import os
import sys
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport.nemo8 import load_nemo_ref, prepare, build, collapse, SPEECH_APP, SHADERS
from gpuport.server import ExecServer
from gpuport.config import WORK_DIR

nr = load_nemo_ref(); w, L = prepare(nr)
wpath = w.save(os.path.join(WORK_DIR, 'nemo8_kit'))
V = L['V']
base = np.fromfile(os.path.join(SPEECH_APP, 'clip10s.pcm'), '<i2').astype(np.float32) / 32768.0
srv = ExecServer(wpath, SHADERS).start()
ok_all = True
for n in [1600, 4000, 16000, 16000 + 640, 16000 + 1280, 16000 + 1920, 36800, 80000, 124321, 160000, 163840, 176000, 192000, 195000, 205000]:
    x = np.resize(base, n)
    pe, _, Re, T = build(n, L, att_km=False)
    pd, ard, Rd, _ = build(n, L)
    le, ms_e = srv.run(pe, x, Re['LOG'], T * (V + 1)); ld, ms_d = srv.run(pd, x, Rd['LOG'], T * (V + 1))
    le = le.reshape(T, V + 1); ld = ld.reshape(T, V + 1)
    diff = np.abs(le - ld).max(); agree = (le.argmax(1) == ld.argmax(1)).mean()
    same = collapse(le, L['tokens'], V) == collapse(ld, L['tokens'], V)
    kind = ('k-major' if 'FLK' in Rd else 'matmul attention' if 'ST' in Rd else 'old shaders')
    ok = bool(same and agree == 1.0 and diff < 1e-2 and np.isfinite(ld).all())
    ok_all &= ok
    print(f'{"ok  " if ok else "FAIL"} {n / 16000:6.2f} s T={T:3d}: {ms_e:5.0f} -> {ms_d:5.0f} ms ({kind}), max diff {diff:.2g}, argmax {agree * 100:.1f}%, '
          f'same tokens {same}, arena {ard.used() / 1e6:.1f}M')
srv.stop()
print('NEMO8 LENGTHS PASS' if ok_all else 'NEMO8 LENGTHS FAIL')
raise SystemExit(0 if ok_all else 1)
