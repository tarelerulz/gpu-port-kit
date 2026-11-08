"""nemo8 with int8 linear layers on the GPU (encoder linears + head; front end and attention stay float32).
Compares with the C reference (~/speech-app/port/*_logits.bin) and with the float32 GPU path, and reports speed and
weight size. usage: python3 tests/nemo8_int8.py"""
import os, sys, time
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport import nemo8 as nb
from gpuport.server import ExecServer
from gpuport.config import WORK_DIR

nr = nb.load_nemo_ref()
w8, L8 = nb.prepare(nr, int8=True)
p8 = w8.save(os.path.join(WORK_DIR, 'nemo8_int8'))
wf, Lf = nb.prepare(nr, int8=False)
pf = wf.save(os.path.join(WORK_DIR, 'nemo8_kit'))
print(f'weight blob: int8 {os.path.getsize(p8)/1e6:.1f} MB, float32 {os.path.getsize(pf)/1e6:.1f} MB')
srv8 = ExecServer(p8, nb.SHADERS).start()
srvf = ExecServer(pf, nb.SHADERS).start()
V = L8['V']; toks = L8['tokens']
for clip, ref_file in [('clip5s', 'speak_logits.bin'), ('clip10s', 'clip10s_logits.bin')]:
    pcm = np.fromfile(os.path.join(nb.SPEECH_APP, clip + '.pcm'), '<i2')
    x = pcm.astype(np.float32) / 32768.0
    plan8, _, R8, T = nb.build(len(pcm), L8, optimized=True)
    planf, _, Rf, _ = nb.build(len(pcm), Lf, optimized=True)
    t0 = time.time(); out8, ms8 = srv8.run(plan8, x, R8['LOG'], T * (V + 1)); w8s = time.time() - t0
    outf, msf = srvf.run(planf, x, Rf['LOG'], T * (V + 1))
    l8 = out8.reshape(T, V + 1); lf = outf.reshape(T, V + 1)
    ref = np.fromfile(os.path.join(nb.SPEECH_APP, ref_file), '<f4').reshape(-1, V + 1)[:T]
    t8 = nb.collapse(l8, toks, V); tf = nb.collapse(lf, toks, V); tr = nb.collapse(ref, toks, V)
    print(f'{clip}: int8 {ms8:.0f} ms (wall {w8s:.2f} s) vs float32 {msf:.0f} ms')
    print(f'   argmax agree with float32 path: {(l8.argmax(1) == lf.argmax(1)).mean()*100:.1f}%, '
          f'with C reference: {(l8.argmax(1) == ref.argmax(1)).mean()*100:.1f}%, max logit diff vs float32 {np.abs(l8-lf).max():.2f}')
    print(f'   int8 text : {t8!r}')
    print(f'   float text: {tf!r}')
    print(f'   C text    : {tr!r}')
    print(f'   same as C reference: {t8 == tr}')
srv8.stop(); srvf.stop()
