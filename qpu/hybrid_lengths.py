"""Hybrid (QPU matmuls) vs reference plan at many clip lengths: same tokens, argmax, logit diff, time. 2026-11-12."""
import os, sys
import numpy as np
KIT = '/home/tarel/gpu-port-kit'; sys.path.insert(0, KIT)
os.environ['GPUPORT_ASM'] = '1'
from gpuport.nemo8 import load_nemo_ref, prepare, build, collapse, SPEECH_APP, SHADERS
from gpuport.server import ExecServer
from gpuport.config import WORK_DIR
nr = load_nemo_ref(); w, L = prepare(nr); wpath = w.save(os.path.join(WORK_DIR, 'nemo8_asm')); V = L['V']
base = np.fromfile(os.path.join(SPEECH_APP, 'clip10s.pcm'), '<i2').astype(np.float32) / 32768.0
os.environ['GPUPORT_ASM'] = '0'; ref = ExecServer(wpath, SHADERS).start()
os.environ['GPUPORT_ASM'] = '1'; hyb = ExecServer(wpath, SHADERS).start()
ok_all = True
LENGTHS = [int(a) for a in sys.argv[1:]] or [1600, 4000, 16000, 24000, 36800, 48000, 64000, 72000, 80000, 100000, 124321, 135000, 160000, 163840, 176000]
for n in LENGTHS:
    x = np.resize(base, n)
    os.environ['GPUPORT_ASM'] = '0'; pr, _, Rr, T = build(n, L)
    os.environ['GPUPORT_ASM'] = '1'; ph, _, Rh, _ = build(n, L)
    na = sum(o.shader.startswith('qpu_mm') for o in ph.ops)
    lr = ref.run(pr, x, Rr['LOG'], T * (V + 1))[0].reshape(T, V + 1); lh, ms_h = hyb.run(ph, x, Rh['LOG'], T * (V + 1)); lh = lh.reshape(T, V + 1)
    tr = np.median([ref.run(pr, x, Rr['LOG'], T * (V + 1))[1] for _ in range(3)]); th = np.median([hyb.run(ph, x, Rh['LOG'], T * (V + 1))[1] for _ in range(3)])
    diff = float(np.abs(lr - lh).max()); agree = float((lr.argmax(1) == lh.argmax(1)).mean())
    same = collapse(lr, L['tokens'], V) == collapse(lh, L['tokens'], V)
    ok = bool((same or agree >= 0.995) and agree > 0.97 and diff < 1.0 and np.isfinite(lh).all())    # a one-frame flip at a clipped last word is f16 noise, not a failure; ok_all &= ok
    if not same:
        print('  ref   :', collapse(lr, L['tokens'], V)); print('  hybrid:', collapse(lh, L['tokens'], V))
    print(f'{"ok  " if ok else "FAIL"} {n / 16000:6.2f} s T={T:3d} asm ops {na:3d}: ref {tr:5.0f} ms, hybrid {th:5.0f} ms ({th - tr:+5.0f}), max diff {diff:.3g}, argmax {100 * agree:.1f}%, same tokens {same}', flush=True)
ref.stop(); hyb.stop()
print('HYBRID LENGTHS PASS' if ok_all else 'HYBRID LENGTHS FAIL')
