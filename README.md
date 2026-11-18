# gpu-port-kit: porting models to the Pi 400 GPU

Helpers for porting models to the Pi's GPU, measured on the Oido speech model (nemo8) and the asr4all port. The method and the measured rules are in `GPU_MAP.md`, `V3D_FINDINGS.md` and `OIDO_GPU_STORY.md`.
The kit wraps the `exec` batch runner in `runner/exec.c` (one command buffer, barrier after each dispatch).

**Status:** research code from one Pi 400, not a packaged install. On 2026-11-14 the steps below were run from a fresh clone (with `HOME` pointed at an empty folder so that none of the author's files were used): all kit checks pass and the nemo8 model transcribes the public JFK sample correctly on the GPU. The Vulkan loader and driver folders were still the author's, passed by environment variable; a stock Raspberry Pi OS install was not tested.
It does not replace the method; it makes each step quick and enforces the rules.

## Quick start
Needs: a Raspberry Pi 4 or 400 (V3D 4.2) running Linux with Vulkan (loader and the v3dv driver; the patched Mesa in `patches/` adds hang reporting), `glslc`, Python 3 with numpy, and a C compiler with the Vulkan and libdrm headers.

1. Build the runners and the older shaders:
   `sh tools/build_exec_big.sh && sh tools/build_exec_server.sh && sh tools/build_exec_hybrid.sh && sh tools/build_speech_app_shaders.sh`
2. If your Vulkan loader or driver is not the system one, set `GPUPORT_LD_LIBRARY_PATH` and `GPUPORT_VK_ICD` (see "Paths you may need to set").
3. Get the nemo8 weights (see "Model weights" below) and set `GPUPORT_MODEL=/path/to/nemo8.tnm`.
4. Check the runner: `python3 tests/check_runner.py`.
5. Run everything: `python3 tests/run_all.py` (about 2 minutes). It ends with `ALL KIT CHECKS PASS`. One check, `tests/nemo8_rebuild.py`, needs the author's private reference files and is skipped.
6. Transcribe a clip: `python3 tools/transcribe.py clip.wav` (16 kHz mono 16-bit WAV, or raw `.pcm`). Add `GPUPORT_ASM=1` for the hand-written QPU matmul (needs the `exec_hybrid` build from step 1).

Measured on the author's Pi 400 (GPU at 700 MHz), 11 s JFK sample, same text in both modes: default 1.2 s, `GPUPORT_ASM=1` 0.7 s. The very first run after a start takes several seconds longer while pipelines compile.

## GPU setup: what the author's numbers assume
The default path (`python3 tools/transcribe.py clip.wav`) needs only Vulkan with the v3dv driver. Everything below is optional, and it is where your results may differ from the published ones.

- **GPU clock.** The published timings were measured with the GPU at 700 MHz (`v3d_freq=700` and `over_voltage=2` in `config.txt`). A stock Pi 4 or 400 runs the GPU slower (check yours with `vcgencmd measure_clock v3d`), so expect longer times. Do not raise the clock unless you want to: `v3d_freq=700` without `over_voltage=2` made the author's Pi fail to boot (black screen). If you try it, use `tryboot` (a one-time test boot) first and keep a way to edit the SD card from another computer. The gain is small, about 40 ms on a 10 s clip going from 650 to 700 MHz.
- **`GPUPORT_ASM=1` (the hand-written matmul).** The compiled kernels are in the repo (`shaders/qpu/*.qpu`), so you do not need py-videocore6 to use it. You do need the `exec_hybrid` build (step 1 of the quick start) and permission to open the GPU render device (`/dev/dri/renderD128`; on Raspberry Pi OS the default user is in the `render` group). If it fails, run without the flag: the default path gives the same text, only slower. py-videocore6 (with numpy 2.3.x) is only needed to rebuild the kernels or run the checks in `qpu/` (see `qpu/README.md`).
- **Driver.** The author ran a self-built Mesa 25.3.6 v3dv and a kernel with `patches/kernel/v3d-fence-error.patch`, which makes a failed GPU job report an error instead of passing silently. Neither is required for the default path. Whether the stock Raspberry Pi OS driver and kernel are enough for `GPUPORT_ASM=1` has not been tested.
- **Not tested here:** a stock Raspberry Pi OS install, and any board other than the author's Pi 400.

## Oido port scripts (`port/`)
`port/gpu_encoder2.py` is the original, standalone version of the nemo8 GPU encoder: one file that builds the plan, talks to the resident server and decodes the text, with the first (slower) set of shaders and no QPU kernel. It works with the current runner (fixed 2026-11-14; `python3 port/gpu_encoder2.py clip.pcm`, raw 16 kHz int16; 11 s JFK sample correct in about 5 s of GPU time) but is kept for reference. Use `tools/transcribe.py` for real work: it is 4 to 7 times faster. `port/nemo_ref.py` reads the model file and is used by the kit.
Settings (environment variables):
- `OIDO_MODEL`: the nemo8 weights file (download from the Hugging Face links in "Model weights" below).
- `OIDO_GPU_DIR`: folder with the shaders (default: `work/speech-app-shaders`, built by `tools/build_speech_app_shaders.sh`).
- `OIDO_GPU_SERVER`: the server binary (default: `tools/exec_server_fixed`, built by `tools/build_exec_server.sh`).
- `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`: the Vulkan loader and driver (see the quick start).

The resident server source is in `runner/exec_server2.c`; build it with `sh tools/build_exec_server.sh` (it creates `tools/exec_server_fixed`; point `OIDO_GPU_SERVER` at it). The weights are not included.

## Map of the GPU
- `OIDO_GPU_STORY.md`: the whole history of the Oido port in plain words (3.3 s -> 0.54 s for 10 s of speech), what worked and what did not.
- `V3D_FINDINGS.md`: the short technical write-up of the V3D findings.
- `GPU_MAP.md`: how the device, plans, memory and layouts work, with measured speed and known gaps. Read this first.
- `GPU_SHADERS.md`: generated catalog of every shader (local size, bindings, push words). Regenerate: `python3 tools/catalog_shaders.py`.
- `GPU_TIMINGS_T251.md`: per-op timings at nemo8 shapes. Regenerate: `python3 tools/bench_ops.py 251`.
- `tools/devinfo.c`: device limits (build line in the file).
- `python3 tests/run_all.py`: all checks (ops, weight layouts, attention).

## Check the runner first
- `python3 tests/check_runner.py`: runs unsplit matrix calls and fails if the runner is broken (it caught the row-count bug). Run it before trusting any GPU result. It is the first step of `tests/run_all.py`.

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
| `gpuport/check.py` `guard_run` | memory guard: runs a plan with guard values around every region and reports writes outside the plan's regions, outputs never written, and (with `poison=True`) results that depend on memory the plan did not write |
| `gpuport/profile.py` | `profile(plan, arena, weights=..., marks=...)`: per-dispatch GPU time of any plan, by shader (with GMAC/s for matmuls), by stage, top dispatches. Needs `tools/exec_prof` |
| `tests/test_memory_guard.py` | guard on every matmul path (row-major, k-major, padded stride, split rows) and the elementwise ops; also proves the guard catches an overrun and a short write |
| `tests/stream_guard.py` | guard on the streaming front end and the live streaming encoder chunk plans (`fe`, `enc` or both; not in `run_all`, about 2 minutes) |
| `tests/nemo8_guard.py` | guard over the whole nemo8 plan: no stray writes, logits do not depend on old arena contents |
| `tests/test_weight_layouts.py` | weight layout checks: put/put_mm, 8-bit model reader -> GPU matmul, LayerNorm gamma/beta, GLU, depthwise [k][d] |
| `gpuport/ops.py` `attention_km` | relative-position attention as matmuls on the k-major kernel (GPU_MAP section 19); `tests/test_attention_km.py [--time]` checks it against numpy with the memory guard and times it against the old shaders |
| `gpuport/ops.py` k-major helpers | `gemm_km`, `gemm_to_km`, `ln_km`, `glu_dw_km`: layers with activations kept as [channel][frame] (GPU_MAP section 21) |
| `tests/nemo8_lengths.py` | default nemo8 plan vs the exact plan at 15 clip lengths (0.1 s to 12.8 s) |
| `tests/test_attention_layout.py` | attention chain (pe_gen, pos projection, scores, softmax, S@V) vs numpy at 3 shapes |
| `tests/run_all.py` | runs all checks |
| `selftest.py` | checks the kit on the real GPU: add, silu, gemm12 (incl. row-block split), bisect, lint. `--timing`, `--speech-app` (needs the speech-app folder, see below) |

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

## Memory check for a new op or plan (do this with step 3 and again with step 5)
`check.guard_run(label, plan, arena, {'A': input}, writes=['O'], scratch=['TK'], weights=path)` returns `(ok, regions, problems)`.
Comparing values alone misses a shader that also writes somewhere else, or one that reads leftovers of an earlier run.
Run it once plain and once with `poison=True`; the values must be the same both times.

## Timing (step 6)
For a whole plan use `gpuport.profile.profile(plan, arena_init, weights=path)`: any `gemm12`/`gemm12o4` row near 2 GMAC/s is a
matmul still on the row-major path (pass `kscratch` to get about 5-6).
`time_op(Plan([...]), arena, out_off, out_n, repeats=20)`. Change one thing at a time, re-run the per-op test after each change.

## Rules built into the kit (measured on the Pi 400; background in `GPU_MAP.md`)
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
| power-of-two row stride (>= 512) on a row-major matmul: 7x slower | `lint()` warning |
| signed positions, vec4 alignment, no 64-way shared reductions | not checkable by the kit; see the rules note |

## nemo8 proof and speed
- `python3 tests/nemo8_rebuild.py`: rebuild check (weights, logits vs production, tokens vs C reference).
- `python3 tools/nemo8_bench.py 3`: GPU vs CPU speed, writes GPU_NEMO8_BENCH.md.
- `python3 tools/profile_nemo8.py clip10s`: per-dispatch GPU profile (needs tools/exec_prof).
- `python3 tools/bench_gemm.py`: matmul kernels at nemo8 shapes (GPU_GEMM_BENCH.md).
- `shaders/`: kit shaders (fe_frame_s, fe_conv0_v2, fe_im2col_v2, gemm12v), merged with the older speech-app and stream shaders (sources in `shaders/speech-app/`, see "Paths you may need to set") by `merged_shader_dir()`.

## Resident server
- `gpuport/server.py`: `ExecServer(weights, shader_dir)` drives `exec_server2` (`runner/exec_server2.c`) (weights loaded once). Used by `tools/nemo8_bench.py`.

## Build helpers
- `tools/build_exec_prof.py`: builds `tools/exec_prof` (per-dispatch timestamps, 24M arena).
- `tools/build_exec_big.sh`: builds `tools/exec_big` (exec with a 24M arena) from `runner/exec.c` (set `EXEC_SRC` to use another copy). Needed by `tests/nemo8_rebuild.py`.

## Limits
- `exec` (batch) has a fixed 8M-float arena and 32M-float weights (`exec.c`). The nemo8 full-GPU plan needs ~4.4M floats for 1 s
  of audio; longer utterances need `exec_server2` (24M arena, 48M weights), which the kit does not drive yet.
- Persistent servers and streaming state (`stream_gpu2.py`) are not wrapped; use `stream_gpu2.py` from the [oido-streaming](https://github.com/tarelerulz/oido-streaming) repo as the example.
- The lint reads `.comp` files next to the `.spv`. Shaders in subfolders (e.g. `gpu/fe/`) give a warning, not a check.
- Attention (`att_*`), the front end and the streaming shaders have no layout test yet.

## Verified on the Pi (2026-11-04)
- `selftest.py --timing --speech-app`: all checks ok (add max diff 0, silu 2.4e-7, gemm12 1.7e-6 / 2.9e-6 incl. 2 row blocks); bisect ok;
  lint catches a 1-group dispatch and a local-size mismatch; `add` 16384 floats 0.105 ms per dispatch after the first.
- The real nemo8 plan (1 s audio, 399 ops) lints with 0 errors and 23 warnings (front-end shaders are in `gpu/fe/`, no `.comp` next to the `.spv`).

## Paths you may need to set
Some code still defaults to the author's folder layout (`~/speech-app/...`, `~/opt/pios-vulkan`, `~/opt/mesa-v3dv`). On another machine:
- `GPUPORT_SHADERS`: folder with the older speech-app `*.spv` shaders. Their sources are in `shaders/speech-app/`; build them with `sh tools/build_speech_app_shaders.sh` (needs `glslc`). If `~/speech-app/port/gpu` does not exist, the kit uses that build output (`work/speech-app-shaders`) by default. The kit's own shaders are in `shaders/`.
- `GPUPORT_SPEECH_APP`: folder with the old speech-app `port/` files (`nemo_ref.py` is also in this repo under `port/`).
- `GPUPORT_MODEL`: the nemo8 `.tnm` weights file.
- `GPUPORT_LD_LIBRARY_PATH` and `GPUPORT_VK_ICD`: a Vulkan loader and v3dv driver folder, if the system ones are not enough. Without them the system Vulkan is used.
- `GPUPORT_EXEC`: the runner binary (default `tools/exec_big`).

## Model weights (not included)
The nemo8 model used by the nemo8 tests is published on Hugging Face (the same weights as the oido
project's `models/nemo8.tnm`):
- English int8 (used by the tests): https://huggingface.co/lokutor-ai/oido-ctc-small-int8 (CC-BY-4.0, as listed in the oido README; check the page)
- English int4: https://huggingface.co/lokutor-ai/oido-ctc-small-int4 (CC-BY-SA-4.0 per the oido README)
- English streaming: https://huggingface.co/lokutor-ai/oido-ctc-small-stream-int8 (CC-BY-SA-4.0 per the oido README)

The tests look for the weights and reference logits in a folder that is set with the environment
variables in `gpuport/config.py`. The tests that need them say so when the files are missing.

## Licenses
- The kit's own code and documentation: MIT (see `LICENSE`).
- `patches/kernel/`: GPL-2.0+, because it changes `drivers/gpu/drm/v3d` (SPDX `GPL-2.0+`).
- `patches/mesa/`: MIT, because it changes a Mesa file that is MIT-licensed.
- `runner/exec.c` (the batch runner): MIT.
- `shaders/common.glsl`, `shaders/specialization.glsl`, `shaders/gemm12.comp`: MIT (see `shaders/LICENSE-shaders`).
- `shaders/speech-app/*.comp` (the older speech-app shaders, written by the author): MIT, same terms as `shaders/LICENSE-shaders`.
- Not included: the model weights (linked above).
