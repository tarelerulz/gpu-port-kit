"""Transcribe one or more clips with nemo8 on the Pi GPU.
usage: python3 tools/transcribe.py clip.wav [more clips]     (16 kHz mono 16-bit WAV, or raw 16 kHz int16 with a .pcm/.raw extension)
Set GPUPORT_MODEL to the nemo8 .tnm file. GPUPORT_ASM=1 uses the hand-written QPU matmul (needs tools/exec_hybrid, see README).
GPUPORT_LM + GPUPORT_LMDECODE add beam search with the language model (see \"Decoding\" in the README)."""
import os
import sys
import time
import wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpuport.nemo8 import Nemo8Encoder


def load(path):
    if path.endswith(('.pcm', '.raw')):
        return np.fromfile(path, '<i2')
    with wave.open(path) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise SystemExit(f'{path}: need 16 kHz mono 16-bit, got {w.getframerate()} Hz, {w.getnchannels()} ch, {8 * w.getsampwidth()} bit')
        return np.frombuffer(w.readframes(w.getnframes()), '<i2')


if len(sys.argv) < 2:
    raise SystemExit(__doc__)
enc = Nemo8Encoder().start()
for p in sys.argv[1:]:
    pcm = load(p)
    t = time.time(); text = enc.transcribe(pcm)
    print(f'{p}: {len(pcm) / 16000:.1f} s audio, GPU {enc.last_ms:.0f} ms, wall {time.time() - t:.2f} s')
    print(text)
enc.close()
