"""End to end: nemo8 on the Pi GPU transcribes a clip and the text must match the expected text.
Needs the nemo8 weights (GPUPORT_MODEL, see README) and the runner binaries from tools/.
usage: python3 tests/nemo8_e2e.py [clip.pcm [expected.txt]]   (raw 16 kHz mono 16-bit; default: tests/data/jfk.pcm)"""
import os
import sys
import time
import numpy as np
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport.nemo8 import Nemo8Encoder

clip = sys.argv[1] if len(sys.argv) > 1 else os.path.join(KIT, 'tests', 'data', 'jfk.pcm')
exp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(KIT, 'tests', 'data', 'jfk.txt')
pcm = np.fromfile(clip, '<i2')
enc = Nemo8Encoder().start()
t = time.time(); text = enc.transcribe(pcm); wall = time.time() - t
enc.close()
print(f'{len(pcm) / 16000:.1f} s of audio, GPU {enc.last_ms:.0f} ms, wall {wall:.2f} s')
print('text:', text)
want = open(exp).read().strip() if os.path.exists(exp) else None
ok = want is None or text.strip() == want
print('NEMO8 E2E PASS' if ok else f'NEMO8 E2E FAIL, expected: {want}')
raise SystemExit(0 if ok else 1)
