"""Streaming front-end AV4 regression: exact output vs the row-major path.

Needs the local Oido streaming model and Pi Vulkan runner, like stream_fe.
Includes padded DFT input, odd mel K, split conv rows and final zero rows.
"""
import os
import sys
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
from gpuport.stream_fe import prepare, build, SHADERS, OIDO_PORT
from gpuport.server import ExecServer


def main():
    weights, fe, sc, d = prepare(os.path.join(OIDO_PORT, '..', 'models', 'oido_stream.tnm'))
    pcm = np.fromfile(os.path.join(OIDO_PORT, 'clip10s.pcm'), '<i2').astype(np.float32) / 32768
    srv = ExecServer(weights, SHADERS).start()
    try:
        for tm in (1, 21, 64, 76, 252):
            for final in (False, True):
                n = 160 * tm + 200
                x = np.resize(pcm, n)
                if final:
                    x[-280:] = 0
                args = (tm, n, tm - int(final), int(final), fe, sc, d)
                outputs = []
                times = []
                for av4 in (False, True):
                    plan, regions, t3, _ = build(*args, av4=av4)
                    errors = [msg for level, msg in plan.lint(SHADERS) if level == 'error']
                    assert not errors, errors
                    out, ms = srv.run(plan, x, regions['X'], t3 * d)
                    assert np.isfinite(out).all()
                    outputs.append(out)
                    times.append(ms)
                assert np.array_equal(*outputs), (tm, final, np.max(np.abs(outputs[0] - outputs[1])))
                print(f'EXACT Tm={tm} final={final}: {times[0]:.1f} -> {times[1]:.1f} ms', flush=True)
    finally:
        srv.stop()
    print('STREAM FRONT CHECKS PASS')


if __name__ == '__main__':
    main()
