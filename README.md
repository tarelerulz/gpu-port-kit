# gpu-port-kit: porting models to the Pi 400 GPU

Helpers for the workflow in `~/doc/agent-memory/gpu-porting-guide.md` (measured on the oido and asr4all ports).
The kit wraps the `exec` batch runner in `~/oido/port/gpu/` (one command buffer, barrier after each dispatch).
It does not replace the method; it makes each step quick and enforces the rules.

## Map of the GPU
- `GPU_MAP.md`: how the device, plans, memory and layouts work, with measured speed and known gaps. Read this first.
- `GPU_SHADERS.md`: generated catalog of every shader (local size, bindings, push words). Regenerate: `python3 tools/catalog_shaders.py`.
- `GPU_TIMINGS_T251.md`: per-op timings at nemo8 shapes. Regenerate: `python3 tools/bench_ops.py 251`.
- `tools/devinfo.c`: device limits (build line in the file).
- `python3 tests/run_all.py`: all checks (ops, weight layouts, attention).

## Start a new port
```
python3 ~/gpu-port-kit/scaffold.py ~/<model>-gpu <model>
```
This creates `ref.py`, `inventory.md`, `test_ops.py`, `build_plan.py`, `NOTES.md` and `work/`.
Work through them in the order of the checklist (`NOTES.md` has the boxes).

## What is in the kit
| file | use |
|---|---|
| `gpuport/arena.py` | `Arena`: named regions, 64-float alignment, 131072-float gap after each region, limit check |
| `gpuport/weights.py` | `Weights`: `put()` (16-float aligned) and `put_mm()` (gemm12 layout: W^T, N padded to 12) |
| `gpuport/plan.py` | `Plan`: `add()`, `dispatch()` (>= 24 groups, splits above 65535), `prefix(n)`, `write()`, `lint()` |
| `gpuport/plan.py` (CLI) | `python3 -m gpuport.plan <plan.txt> <shader_dir>` (run from the kit folder): lint an existing plan file |
| `gpuport/ops.py` | push layouts for shaders in use: `gemm12` (verified), `add`, `silu` (verified), `ln`, `glu` (verified by tests/test_weight_layouts.py) |
| `gpuport/run.py` | `run(plan, arena, out_off, out_n, weights=...)`: runs `exec`, returns output and submit ms |
| `gpuport/check.py` | `compare`, `op_test` (one op vs numpy), `bisect` (first wrong stage) |
| `gpuport/timing.py` | `time_op`: per-op ms from repeated dispatches |
| `tests/test_weight_layouts.py` | weight layout checks: put/put_mm, 8-bit oido reader -> GPU matmul, LayerNorm gamma/beta, GLU, depthwise [k][d] |
| `tests/test_attention_layout.py` | attention chain (pe_gen, pos projection, scores, softmax, S@V) vs numpy at 3 shapes |
| `tests/run_all.py` | runs all checks |
| `selftest.py` | checks the kit on the real GPU: add, silu, gemm12 (incl. row-block split), bisect, lint. `--timing`, `--oido` |

Config (`gpuport/config.py`) can be overridden with env vars: `GPUPORT_SHADERS` (dir with `exec` and `*.spv`), `GPUPORT_EXEC`, `GPUPORT_WORK`.

## Loop for one op (step 3 of the guide)
1. Write the shader `<name>.comp`, compile to `<name>.spv` (`glslc -O -o <name>.spv <name>.comp`). Keep the `layout(local_size_x = N)` literal; the linter checks it against the plan.
2. In `test_ops.py`: build an `Arena`, fill the input regions, `ops.<helper>` or `Plan().dispatch(...)`, then `check.op_test(label, plan, arena, out_off, numpy_result, tol)`.
3. Expect max diff ~1e-6 for f32 matmul/elementwise on this GPU (kit tests: 1.7e-6 to 2.9e-6 for gemm12).

## Loop for a whole plan (steps 4-5)
1. Build the plan with `Plan.dispatch` / `ops.*`. Lint runs automatically inside `run()`.
2. Pick checkpoints: `(op_index, name, out_off, numpy_value, tol)` in plan order.
3. `check.bisect(label, plan, arena_in, checkpoints, weights=path)` prints each stage; it stops at the first wrong one.
   Each checkpoint runs a plan prefix (one `exec` process each), so keep the list short.

## Timing (step 6)
`time_op(Plan([...]), arena, out_off, out_n, repeats=20)`. Change one thing at a time, re-run the per-op test after each change.

## Rules built into the kit (source: `~/doc/agent-memory/2026-11-03-gpu-porting-rules.md`)
| rule | where enforced |
|---|---|
| >= 24 workgroups per dispatch (gemm12 wrong below that) | `dispatch()` clamps; `lint()` error |
| gemm12 row block <= 251 rows | `ops.gemm12` splits; `lint()` error |
| 65535 groups per dimension | `split_groups()`; `lint()` error; 2-D dispatch warned (shader needs `gid.x + gid.y*gx`) |
| 131072-float gap after each arena region | `Arena.alloc()` |
| 64-float region alignment | `Arena.alloc()` |
| push constants <= 32 words | `Op` raises; `lint()` |
| plan local size must equal the shader's `local_size_x` literal | `lint()` error (reads `<name>.comp`) |
| arena fits exec batch buffer (8M floats) and weights (32M floats) | `Arena` limit, `run()` check |
| signed positions, vec4 alignment, no 64-way shared reductions | not checkable by the kit; see the rules note |

## nemo8 proof and speed
- `python3 tests/nemo8_rebuild.py`: rebuild check (weights, logits vs production, tokens vs C reference).
- `python3 tools/nemo8_bench.py 3`: GPU vs CPU speed, writes GPU_NEMO8_BENCH.md.
- `python3 tools/profile_nemo8.py tarel40`: per-dispatch GPU profile (needs tools/exec_prof).
- `python3 tools/bench_gemm.py`: matmul kernels at nemo8 shapes (GPU_GEMM_BENCH.md).
- `shaders/`: kit shaders (fe_frame_s, fe_conv0_v2, fe_im2col_v2, gemm12v), merged with the oido and stream shaders by `merged_shader_dir()`.

## Resident server
- `gpuport/server.py`: `ExecServer(weights, shader_dir)` drives oido's `exec_server2` (weights loaded once). Used by `tools/nemo8_bench.py`.

## Build helpers
- `tools/build_exec_prof.py`: builds `tools/exec_prof` (per-dispatch timestamps, 24M arena).
- `tools/build_exec_big.sh`: builds `tools/exec_big` (exec with a 24M arena) from ~/oido/port/gpu/exec.c. Needed by `tests/nemo8_rebuild.py`.

## Limits
- `exec` (batch) has a fixed 8M-float arena and 32M-float weights (`exec.c`). The nemo8 full-GPU plan needs ~4.4M floats for 1 s
  of audio; longer utterances need `exec_server2` (24M arena, 48M weights), which the kit does not drive yet.
- Persistent servers and streaming state (`stream_gpu2.py`) are not wrapped; use them as in `~/oido/port/`.
- The lint reads `.comp` files next to the `.spv`. Shaders in subfolders (e.g. `gpu/fe/`) give a warning, not a check.
- Attention (`att_*`), the front end and the streaming shaders have no layout test yet.

## Verified on the Pi (2026-11-04)
- `selftest.py --timing --oido`: all checks ok (add max diff 0, silu 2.4e-7, gemm12 1.7e-6 / 2.9e-6 incl. 2 row blocks); bisect ok;
  lint catches a 1-group dispatch and a local-size mismatch; `add` 16384 floats 0.105 ms per dispatch after the first.
- The real nemo8 plan (1 s audio, 399 ops) lints with 0 errors and 23 warnings (front-end shaders are in `gpu/fe/`, no `.comp` next to the `.spv`).
