"""Run every kit check on the GPU: selftest (ops), memory guard, weight layouts, attention layout. Exit 1 on any failure.
usage: python3 ~/gpu-port-kit/tests/run_all.py"""
import os
import subprocess
import sys
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
steps = [['tests/check_runner.py'], ['selftest.py'], ['tests/test_memory_guard.py'], ['tests/test_weight_layouts.py'], ['tests/test_attention_layout.py'], ['tests/test_attention_km.py'], ['tests/nemo8_e2e.py'], ['tests/nemo8_guard.py'], ['tests/nemo8_lengths.py']]
# nemo8_rebuild.py compares against the author's private reference files (production weight cache, C reference logits); run it only where they exist
sys.path.insert(0, KIT)
from gpuport.nemo8 import PROD_W
if os.path.exists(PROD_W):
    steps.insert(-3, ['tests/nemo8_rebuild.py'])
else:
    print('SKIP tests/nemo8_rebuild.py: needs the author\'s private reference files (not published)')
bad = []
for s in steps:
    r = subprocess.run([sys.executable, os.path.join(KIT, s[0])], cwd=KIT, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ['(no output)'])[-1]
    print(f'{"OK  " if r.returncode == 0 else "FAIL"} {s[0]}: {last}')
    if r.returncode != 0:
        bad.append(s[0]); print(r.stdout[-2000:], r.stderr[-2000:])
print('ALL KIT CHECKS PASS' if not bad else f'FAILED: {bad}')
raise SystemExit(1 if bad else 0)
