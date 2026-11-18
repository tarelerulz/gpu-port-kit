"""Beam search with a language model over the nemo8 scores the GPU produced.

The GPU path returns, per audio frame, a score for every token plus the CTC blank (the last column). Without this module the kit
decodes them greedily (best token per frame), which can spell unusual words with stray letters ("missssion"). This runs the same
beam search with the language model that the CPU build (tasr_cli --lm) uses, in a small helper program, tasr_lmdecode, built from
the oido repo (`make -C host tasr_lmdecode`).

Enable it with two environment variables (or the arguments of LMDecoder):
  GPUPORT_LM          the language model file, e.g. nemo_lm.tlm
  GPUPORT_LMDECODE    the tasr_lmdecode program
"""
import os
import struct
import subprocess
import numpy as np


class LMDecoder:
    def __init__(self, tool, lm, beam=4):
        if not os.path.isfile(tool):
            raise FileNotFoundError(f'tasr_lmdecode not found: {tool} (build it with: make -C host tasr_lmdecode in the oido repo)')
        if not os.path.isfile(lm):
            raise FileNotFoundError(f'language model not found: {lm}')
        self.p = subprocess.Popen([tool, lm, '--beam', str(beam)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def decode(self, logits, tokens):
        """logits: (T, V+1) float array of raw scores, blank last. tokens: the model's token strings (index = token id)."""
        lg = np.ascontiguousarray(logits, np.float32)
        T, V1 = lg.shape
        self.p.stdin.write(struct.pack('<ii', T, V1) + lg.tobytes())
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('tasr_lmdecode stopped (exit %s)' % self.p.poll())
        text = ''.join(tokens[int(i)] for i in line.split())
        return text[1:] if text.startswith(' ') else text

    def close(self):
        if self.p and self.p.poll() is None:
            try:
                self.p.stdin.write(struct.pack('<ii', 0, 0)); self.p.stdin.close()
                self.p.wait(timeout=2)
            except Exception:
                self.p.kill()


def from_env():
    """An LMDecoder if GPUPORT_LM and GPUPORT_LMDECODE are both set, else None."""
    lm, tool = os.environ.get('GPUPORT_LM'), os.environ.get('GPUPORT_LMDECODE')
    if lm and tool:
        return LMDecoder(tool, lm)
    if lm or tool:
        raise SystemExit('set both GPUPORT_LM (language model file) and GPUPORT_LMDECODE (tasr_lmdecode program), or neither')
    return None
