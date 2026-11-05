"""Plan = list of dispatches run in one command buffer (exec batch format):
    <shader> <local> <gx> <gy> <p0> ... <p31>      (shader is <name>.spv in the shader dir)
The lint() method checks the rules measured on this Pi (see config.py)."""
import math
import os
import re
from .config import MIN_WORKGROUPS, MAX_GROUP_DIM, MAX_PUSH, GEMM12_MAX_ROWS, EXEC_ARENA_FLOATS


class Op:
    __slots__ = ('shader', 'local', 'gx', 'gy', 'push')

    def __init__(self, shader, local, gx, gy, push):
        push = [int(x) for x in push]
        if len(push) > MAX_PUSH:
            raise ValueError(f'{shader}: {len(push)} push words, max {MAX_PUSH}')
        self.shader, self.local, self.gx, self.gy = shader, int(local), int(gx), int(gy)
        self.push = push + [0] * (MAX_PUSH - len(push))

    def line(self):
        return f'{self.shader} {self.local} {self.gx} {self.gy} ' + ' '.join(str(x) for x in self.push)


class Plan:
    def __init__(self, ops=None):
        self.ops = list(ops or [])

    def __len__(self):
        return len(self.ops)

    def add(self, shader, local, gx, gy, push):
        self.ops.append(Op(shader, local, gx, gy, push))
        return self

    def dispatch(self, shader, local, threads, push, min_groups=MIN_WORKGROUPS):
        """One thread per item: groups = ceil(threads/local), at least min_groups (the 24-group rule).
        Above 65535 groups the dispatch is split into gx x gy; the shader must then use
        flat = gid.x + gid.y * gx (lint reports these)."""
        groups = max(min_groups, math.ceil(threads / local))
        gx, gy = split_groups(groups)
        return self.add(shader, local, gx, gy, push)

    def prefix(self, n):
        return Plan(self.ops[:n])

    def write(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            for op in self.ops:
                f.write(op.line() + '\n')
        return path

    @classmethod
    def load(cls, path):
        ops = []
        for line in open(path):
            t = line.split()
            if not t or t[0].startswith('#'):
                continue
            ops.append(Op(t[0], t[1], t[2], t[3], [int(x) for x in t[4:4 + MAX_PUSH]]))
        return cls(ops)

    def lint(self, shader_dir):
        """Returns a list of (level, message). level is 'error' or 'warn'."""
        out = []
        for i, op in enumerate(self.ops):
            where = f'op {i} {op.shader}'
            groups = op.gx * op.gy
            if groups < MIN_WORKGROUPS:
                out.append(('error', f'{where}: {groups} workgroups < {MIN_WORKGROUPS} (wrong output seen for gemm12)'))
            if op.gx > MAX_GROUP_DIM or op.gy > MAX_GROUP_DIM:
                out.append(('error', f'{where}: dispatch {op.gx}x{op.gy} exceeds {MAX_GROUP_DIM} per dimension'))
            if op.gy > 1:
                out.append(('warn', f'{where}: 2-D dispatch {op.gx}x{op.gy}; check the shader uses gid.x + gid.y*gx'))
            spv = os.path.join(shader_dir, op.shader + '.spv')
            if not os.path.exists(spv):
                out.append(('error', f'{where}: {spv} not found'))
            comp = os.path.join(shader_dir, op.shader + '.comp')
            if os.path.exists(comp):
                m = re.search(r'local_size_x\s*=\s*(\d+)', open(comp).read())
                if m is None:
                    out.append(('warn', f'{where}: no local_size_x literal in {op.shader}.comp'))
                elif int(m.group(1)) != op.local:
                    out.append(('error', f'{where}: plan local {op.local} != local_size_x {m.group(1)} in .comp '
                                         '(the literal in the shader is what runs)'))
            else:
                out.append(('warn', f'{where}: no {op.shader}.comp source, local size not checked'))
            if op.shader.startswith('gemm12') and op.push[0] > GEMM12_MAX_ROWS:
                out.append(('error', f'{where}: gemm12 row block {op.push[0]} > {GEMM12_MAX_ROWS}'))
            if op.shader == 'gemm12' and op.push[0] == 0:
                out.append(('error', f'{where}: gemm12 with 0 rows'))
        return out


def split_groups(groups, max_dim=MAX_GROUP_DIM):
    if groups <= max_dim:
        return groups, 1
    gx = max_dim
    return gx, math.ceil(groups / gx)


def check_arena_limit(used_floats):
    if used_floats > EXEC_ARENA_FLOATS:
        return [('error', f'arena {used_floats} floats > exec batch arena {EXEC_ARENA_FLOATS}')]
    return []


if __name__ == '__main__':
    import sys
    plan_path = sys.argv[1]
    sdir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/oido/port/gpu')
    p = Plan.load(plan_path)
    msgs = p.lint(sdir)
    for lvl, m in msgs:
        print(lvl.upper(), m)
    errs = sum(1 for lvl, _ in msgs if lvl == 'error')
    print(f'{len(p)} ops, {errs} errors, {len(msgs) - errs} warnings')
    raise SystemExit(1 if errs else 0)
