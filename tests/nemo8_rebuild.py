"""Rebuild the nemo8 full-GPU encoder with the kit's helpers only (Weights, Arena, Plan, ops), then check it:
  1. weight blob: byte-identical to the production cache (~/speech-app/port/cache2/weights.bin)
  2. logits on clip10s (10 s) and clip5s (5 s) vs the C reference (~/speech-app/port/*_logits.bin)
  3. collapsed token text vs the same reference
  4. logits vs the production GPU encoder (~/speech-app/port/gpu_encoder2.py): must be identical
The C reference is int8; the float GPU path agrees with it on ~98.8% of argmax frames (earlier notes), so the
C check uses a 98% floor. The production comparison is the exact one.
The plan is run through tools/exec_big (exec with a 24M-float arena, 48M-float weights), since a 10 s clip
needs 18.7M arena floats.
usage: python3 ~/gpu-port-kit/tests/nemo8_rebuild.py"""
import importlib.util
import os
import sys
import time
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.expanduser('~/speech-app/port'))
from gpuport import Arena, Weights, Plan, ops
from gpuport.run import run
from gpuport.config import WORK_DIR



from gpuport.nemo8 import (load_nemo_ref, prepare, build, collapse, SPEECH_APP, MODEL, PROD_W, SHADERS,
                           ARENA_FLOATS)
EXEC_BIG = os.path.join(KIT, 'tools', 'exec_big')


def main():
    nr = load_nemo_ref()
    w, L = prepare(nr)
    os.makedirs(WORK_DIR, exist_ok=True)
    wpath = w.save(os.path.join(WORK_DIR, 'nemo8_kit'))
    mine = np.fromfile(wpath, '<f4'); prod = np.fromfile(PROD_W, '<f4')
    # the kit blob = the production blob + the cached position tables appended (2026-11-12)
    same = mine.size >= prod.size and np.array_equal(mine[:prod.size], prod)
    print(f'weights: kit {mine.size} floats, production {prod.size}; production blob is a byte-identical prefix: {same}')
    ok_all = same
    V = L['V']; toks = L['tokens']
    for clip, ref_file in [('clip10s', 'clip10s_logits.bin'), ('clip5s', 'speak_logits.bin')]:
        pcm = np.fromfile(os.path.join(SPEECH_APP, clip + '.pcm'), '<i2')
        plan, ar, R, T = build(len(pcm), L, att_km=False)    # old attention shaders: the bit-exact comparison
        arena_in = pcm.astype(np.float32) / 32768.0
        t0 = time.time()
        out, ms = run(plan, arena_in, R['LOG'], T * (V + 1), weights=wpath, shader_dir=SHADERS,
                      exec_bin=EXEC_BIG, arena_floats=ARENA_FLOATS)
        wall = time.time() - t0
        logits = out.reshape(T, V + 1)
        ref = np.fromfile(os.path.join(SPEECH_APP, ref_file), '<f4').reshape(-1, V + 1)
        ok_shape = ref.shape[0] == T
        if not ok_shape:
            print(f'{clip}: frames {T} vs reference {ref.shape[0]}')
        d = np.abs(logits - ref[:T]).max()
        agree = (logits.argmax(1) == ref[:T].argmax(1)).mean()
        text_k = collapse(logits, toks, V); text_r = collapse(ref[:T], toks, V)
        same_text = text_k == text_r
        print(f'{clip}: {len(pcm)/16000:.0f} s, T={T}, arena {ar.used()/1e6:.2f}M floats, gpu {ms:.0f} ms, wall {wall:.2f} s')
        print(f'   logits max diff {d:.3g}, argmax agree {agree*100:.2f}%, tokens identical: {same_text}')
        print(f'   text: {text_k!r}')
        if not same_text:
            print(f'   reference: {text_r!r}')
        ok_all &= ok_shape and same_text and agree > 0.98
        try:
            from gpu_encoder2 import GpuEncoder2
            if not hasattr(main, 'enc'):
                main.enc = GpuEncoder2()
            prod = main.enc.logits(pcm)
            pdiff = np.abs(logits - prod).max()
            same_prod = pdiff == 0.0
            print(f'   vs production GPU encoder: max diff {pdiff:.3g}, identical: {same_prod}')
            ok_all &= same_prod
            # default plan (attention on the matmul kernel, 2026-11-12): other summation order, so not bit-identical
            plan2, ar2, R2, _ = build(len(pcm), L)
            out2, ms2 = run(plan2, arena_in, R2['LOG'], T * (V + 1), weights=wpath, shader_dir=SHADERS,
                            exec_bin=EXEC_BIG, arena_floats=ARENA_FLOATS)
            l2 = out2.reshape(T, V + 1)
            d2 = np.abs(l2 - logits).max(); a2 = (l2.argmax(1) == logits.argmax(1)).mean()
            t2 = collapse(l2, toks, V) == text_k
            print(f'   default plan (matmul attention): gpu {ms2:.0f} ms (exact plan {ms:.0f}), max diff {d2:.3g}, '
                  f'argmax agree {a2*100:.2f}%, same tokens: {t2}')
            ok_all &= t2 and a2 == 1.0 and d2 < 1e-2
        except Exception as e:   # production path not available: say so, and count it as a failure
            print(f'   production comparison FAILED to run: {e}')
            ok_all = False
    print('NEMO8 REBUILD MATCHES' if ok_all else 'NEMO8 REBUILD DIFFERS (see above)')
    return 0 if ok_all else 1


if __name__ == '__main__':
    raise SystemExit(main())
