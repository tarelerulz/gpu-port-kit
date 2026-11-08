"""Resident GPU server: speech-app's gpu/exec_server2 (24M-float arena and 48M-float weights, loaded once).
Each request sends a plan file and an arena file; the server runs it and writes the requested output region.
Avoids the per-run process start, Vulkan setup and weight-file load that the one-shot exec pays."""
import os
import subprocess
import threading
import numpy as np
from .config import VK_ENV, WORK_DIR, EXEC_ARENA_FLOATS
from .plan import Plan

EXEC_SERVER2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools', 'exec_server_fixed')  # built by tools/build_exec_server.sh
SERVER_ARENA_FLOATS = 24 << 20


class ExecServer:
    def __init__(self, weights, shader_dir, binary=EXEC_SERVER2, work=WORK_DIR):
        self.weights, self.shader_dir, self.binary, self.work = weights, shader_dir, binary, work
        self.proc = None
        self.lock = threading.Lock()
        self.seq = 0
        os.makedirs(work, exist_ok=True)

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return self
        env = dict(os.environ, **VK_ENV)
        self.proc = subprocess.Popen([self.binary, self.weights], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, env=env, cwd=self.shader_dir)
        line = self.proc.stderr.readline()          # 'ready' once pipelines and buffers exist
        if 'ready' not in line:
            raise RuntimeError('exec_server2 did not start: ' + line.strip())
        return self

    def run(self, plan, arena_init, out_off, out_n):
        """Returns (out float32[out_n], server_ms). Same contract as gpuport.run.run()."""
        if isinstance(plan, (list, tuple)):
            plan = Plan(plan)
        arena_init = np.asarray(arena_init, np.float32).ravel()
        if arena_init.size > SERVER_ARENA_FLOATS or out_off + out_n > SERVER_ARENA_FLOATS:
            raise MemoryError('arena beyond the server arena (24M floats)')
        with self.lock:
            self.start()
            self.seq += 1
            base = os.path.join(self.work, f'srv_{self.seq % 4}')
            plan.write(base + '.plan')
            arena_init.astype('<f4').tofile(base + '.arena')
            self.proc.stdin.write(f'{base}.plan {base}.arena 0 {base}.out {out_off} {out_n}\n')
            self.proc.stdin.flush()
            resp = self.proc.stdout.readline()
            if not resp.startswith('done'):
                raise RuntimeError('exec_server2: ' + resp.strip())
            out = np.fromfile(base + '.out', '<f4')
        return out, float(resp.split()[1])

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.stdin.close(); self.proc.wait(timeout=20)
        self.proc = None
