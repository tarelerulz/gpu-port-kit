# How the Pi 400 GPU works, for porting models

Written 2026-11-04 from the kit's tests and benchmarks. **(T)** = run and checked in this session.
**(N)** = from the earlier oido/asr4all notes (~/doc/agent-memory/2026-11-03-gpu-porting-rules.md etc.), not re-run here.
Companion files: GPU_SHADERS.md (generated shader catalog), GPU_TIMINGS_T251.md (generated timings).

## 1. Device (T: tools/devinfo.c)
- V3D 4.2.14.0 through the PiOS Vulkan loader and broadcom ICD (~/opt/pios-vulkan), Vulkan API 1.3.
- One queue family with compute, graphics and transfer. One memory type: DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT,
  heap 1895 MB. Data is written and read through a plain mapped pointer; there is no staging copy.
- Workgroup: max 256 invocations; max size 256x256x256. Subgroup size 16.
- Shared memory 16 KB per workgroup.
- Push constants 128 bytes = 32 uint words.
- Up to 8 storage buffers per stage. Max storage buffer range 1 GB. Storage offset alignment 32 bytes.
- Dispatch count 65535 per dimension.

## 2. How a plan runs (T: gpuport/run.py, tools/bench_ops.py)
- A plan is a list of dispatches, recorded into one command buffer. A compute barrier follows every dispatch, so each op
  sees the previous op's writes. You do not sync by hand.
- Every shader gets the same two buffers: binding 0 = the arena (floats; some shaders also view it as vec4),
  binding 1 = the weights (read only). Every address is a float offset passed in push constants.
- Plan line: `<shader> <local> <gx> <gy> <push 0..31>`. The shader file is `<shader>.spv` in the directory exec runs from.
- Fixed sizes of the `exec` batch runner (exec.c): arena 8M floats (32 MB), weights 32M floats (128 MB).
  Larger plans use exec_server2 (arena 24M, weights 48M), which the kit does not wrap yet. Arena need: 2 s audio = 5.95M floats, 10 s = 18.7M (computed).
- Fixed cost: about 3.8 ms per run for submit and wait (T). The per-op timings below subtract it.
- Output: one region of the arena is read back after the plan (run() takes out_off and out_n).

## 3. Memory rules
- Arena regions start on 64-float boundaries and are followed by a 131072-float gap (T in Arena; N: a matmul output placed just
  below its input read back wrong).
- vec4 views need offsets divisible by 4. All row strides in the oido port are multiples of 4 (N).
- Weight offsets are 16-float aligned (T: Weights.put).
- In-place is fine for: softmax (T), silu (shader comment), LayerNorm X -> X (N, used at the end of the encoder).
- Use signed ints for offsets that can go negative (positional index T-1-i+j). Unsigned wrap gave wrong positions (N).

## 4. Work sizes (T: Plan.dispatch / lint)
- groups = max(24, ceil(threads / local)). Fewer than 24 workgroups gave wrong gemm12 output at some shapes (N).
- The `layout(local_size_x = N)` literal in the .comp is what runs. The plan's local must match it (T: lint catches a mismatch).
- Above 65535 groups the dispatch is split gx x gy; the shader must compute gid.x + gid.y*gx. **Not tested in the kit yet.**

## 5. Layout conventions (T: tests/test_weight_layouts.py, tests/test_attention_layout.py)
| op | inputs | weights | output | tested |
|---|---|---|---|---|
| gemm12 (matmul + epilogue) | A row-major [M][K], row stride sAm | `put_mm`: W^T as [K][NP], NP = roundup(N,12), zero padding; bias [NP] | [M][N] | selftest; layouts test (M=37, 300 split into 2 blocks) |
| gemm12 epilogue codes | | | | 0 none, 1 hardswish, 2 silu, 3 relu, 4 sigmoid (common.glsl) |
| 8-bit weights (oido format) | int8 rows of kp = roundup(K,16) | the reader gives W = q * scale[row]; the GPU gets W in f32 via put_mm | | layouts test: reader exact, GPU max diff 1.1e-5 |
| add | a[dst+i] += scale*a[src+i]; scale as float bits in push | none | in place | selftest |
| silu | in -> out | none | | selftest (2.4e-7) |
| ln | rows [T][d] | gamma at O, beta at O+d (one put of concat) | [T][d], eps 1e-5 | layouts test (4.8e-7) |
| glu | rows [T][2d]: a = cols [0,d), b = [d,2d) | none | [T][d] = a*sigmoid(b) | layouts test (1.2e-7) |
| dw4 (depthwise conv) | [T][d] | weights [k][d] (transposed from nn [d][k]), bias [d] | [T][d], centered, zero padded | layouts test (exact) |
| pe_gen | none | none | [2T-1][d]: row r = position T-1-r; even col sin, odd col cos; div = exp(-2i ln(10000)/d) | attention test (via P) |
| attention (att_scores5 or att_scores6) | qkv rows [q|k|v], each d wide; head h = columns h*dh..(h+1)*dh; P [2T-1][d] | pbu, pbv [d] | ATT [T][d] | attention test: T=8,13,21; d=64/176; max diff 7.8e-7 |

Attention chain, in order: pe_gen -> gemm12 (P = PE Wpos^T + bpos) -> att_scores5 (S = scale*(qu.k + qv.P[T-1-i+j]),
qu = q+pbu, qv = q+pbv, scale = 1/sqrt(dh)) -> att_softmax (in place over j) -> att_pv6 (ATT = S V).

Not covered by a test yet: the front end (fe_*), the streaming shaders (ch_*, gemm_ns*), att_pv/att_scores variants
other than the ones above, and gemm4 / gemm12u.

## 6. Measured speed (T: GPU_TIMINGS_T251.md, 10 s clip = 251 frames)
| op | ms |
|---|---|
| gemm12 FF1 (251x176 @ 176x704) | 16.2 |
| gemm12 FF2 (251x704 @ 704x176) | 16.2 |
| gemm12 QKV (251x176 @ 176x528) | 12.1 |
| att_pv6 | 11.5 |
| gemm12 pos (501 positions, 2 blocks) | 8.7 |
| att_scores5 | **37.6** |
| att_scores6 (tiled) | 44.3 (slower, correct) |
| att_softmax | 5.2 |
| dw4 (k=31) | 6.8 |
| pe_gen | 1.4 |
| ln / glu / silu / add (251x176) | 0.3 - 0.6 each |

- Sum of the attention and FF matmul ops for one conformer layer is about 107 ms; 16 layers give about 1.7 s. The rest is the
  conv matmuls and small ops. Full-GPU nemo8 with its front end on the GPU takes 5.2 s per 10 s clip (measured here, both the kit
  rebuild and the production encoder). The 2.85 s in the older notes is the hybrid setup, with the front end on the CPU (N).
- **att_scores5 is the largest single cost**, more than a whole FF matmul. It is the first place to look for speed
  Tried: att_scores6 (2 query rows x 4 keys per invocation, key vectors shared) is correct (attention test passes at 3 shapes) but
  SLOWER: 44.3 ms vs 37.6 ms at T=251. REJECTED for now; the bottleneck is not the key loads. Next idea to test: fewer live
  values per thread (the loop over dh/4 and the 8 accumulators), or fewer dispatch threads with more work each.
- gemm12 at M=251 runs at about 1.9 GMAC/s (251*176*704 MACs in 16.2 ms). The CPU NEON int8 path is faster for this model (N).
- Element-wise ops are cheap; a dispatch costs about 40 us (N).
- Arena size did not change speed (N). Shared-memory reductions with one thread per sum hung the GPU (N): avoid them.

## 7. Full-model check
- tests/nemo8_rebuild.py rebuilds nemo8 with kit helpers. Weights are byte-identical to the production cache; logits are
  identical to the production GPU encoder on tarel40 and speak; tokens match the C reference on both. Needs tools/exec_big
  (built from exec.c with a 24M-float arena: `sed` the two defines and compile with -lvulkan).

## 8. How to design a model for this GPU
1. Check the fit first (N): f32 weights, big M (many frames per matmul), few steps per block. 8/4-bit weights and per-token decoding do not fit.
2. Write the numpy reference, then one test per op type with the layout from section 5. Use `Weights.put_mm`, `ops.*`, `check.op_test`.
3. Build the plan and run `check.bisect` with checkpoints after each stage. Fix the first wrong stage before tuning.
4. Time each op at the real shapes (`tools/bench_ops.py T`), change one thing, and re-check accuracy.
5. Log each round in a dated note.

## 9. Known gaps and warnings
- Shaders in ~/src/asr4all-gpu/shaders: `ew` and `gemm12` have .spv older than .comp (stale flag in GPU_SHADERS.md). The oido
  gemm12 in ~/oido/port/gpu is not flagged. The asr4all gemm12.comp needs `-I ~/src/asr4all-gpu/shaders` (specialization.glsl).
- The 65535-group split path is untested. The streaming and front-end layouts are only checked end to end (N).
- att_scores5's header comment says "11 vec4 chunks", but the code loops over dh/4 (44/4 = 11 here), so it is general.
- Arena need for the nemo8 full-GPU plan (computed from gpu_encoder2.build): 1 s = 4.37M floats, 2 s = 5.95M, 5 s = 10.7M, 10 s = 18.7M.
  The exec batch runner (8M) therefore fits about 2 s of audio; exec_server2 (24M) fits 10 s. The kit's `run()` only drives exec, so check `Arena.used()` first.

## 10. Rules measured 2026-11-05 (nemo8 optimization round)
- **Power-of-two row stride is a 7x penalty for gemm12 (T).** A DFT input with row stride 512 ran at 115 ms per 251-row block
  (0.29 GMAC/s); the same data with stride 513, 516, 520, 528 or 544 ran at 16-17 ms (1.9-2.0 GMAC/s). Pad activation rows
  to a non-power-of-two stride before a matmul reads them (tools/bench_gemm.py, the stride test in the notes).
- **gemm4 (oido gpu/gemm4.comp) is slower than gemm12 at all four nemo8 shapes (T, tools/bench_gemm.py):**
  0.09-1.05 GMAC/s against 1.9-2.0. Keep gemm12.
- **Per-thread-per-output front-end shaders are slow (T).** fe_conv0 (one thread per output, 9 taps with re-reads) took 434 ms;
  the rewritten version (one thread per (t2, f) producing all 176 channels, same arithmetic order) takes 70 ms, bit-identical.
  fe_im2col (scattered gather) took 223 ms; one thread per (row, channel) takes 77 ms, bit-identical.
- Front end runs on 1001 frames (10 ms hop), not the 251 encoder frames. Its matmuls are the largest share of the front end.
- Profile tool: tools/build_exec_prof.py builds tools/exec_prof (Vulkan timestamps per dispatch); tools/profile_nemo8.py
  prints time by shader and by stage.

## 11. nemo8 after this round (T, tools/nemo8_bench.py -> GPU_NEMO8_BENCH.md)
| | before | after |
|---|---|---|
| GPU kernel time, 10 s clip | 5.17 s | 3.89 s |
| logits vs production GPU encoder | max diff 0 | max diff 0 |
| CPU reference (host tasr_cli_a72, greedy) | 0.46 s wall, 1.62 CPU-s | |
| GPU process CPU | | 0.68 CPU-s per run |

Where the 3.9 s goes now: layers 2.7 s (70%; matmuls at the 1.9 GMAC/s ceiling plus att_scores5 at 0.6 GMAC/s, 0.6 s total),
front end 1.1 s (conv2 dense matmul about 0.7 s: 1.4 GMAC of inherent work at 1.96 GMAC/s). Wall time is about 0.8 s more than
the kernel time: one-shot exec process setup (device, pipelines, plan file, lint). A resident server would remove most of that.

Honest verdict: the GPU path is about 8x slower than the CPU path in wall time. It uses about 40% of the CPU time, so it is an
offload option, not a speed win. Next levels need a faster matmul kernel (the layer and conv2 time is there) or a faster
attention score kernel.

## 12. Round 2 (2026-11-05): what was tried for speed
Matmul kernels at the nemo8 shapes, one 251-row block (tools/bench_gemm.py, GPU_GEMM_BENCH.md):
| kernel | DFT (K=512, stride 512) | conv2 (K=1584) | subsample (K=3520) | FF1 (K=176, N=704) | verdict |
|---|---|---|---|---|---|
| gemm12 (oido) | 115 ms | 36 ms | 80 ms | 16 ms | keep |
| gemm4 | 387 ms | 66 ms | 152 ms | 30 ms | REJECTED |
| gemm_k5 (stream) | 92 ms | 103 ms | 174 ms | 204 ms | REJECTED |
| gemm12v (kit, vec4 A along K) | 112 ms | 147 ms | 322 ms | 70 ms | REJECTED (4x slower at the tile, cause not isolated) |
The DFT row uses stride 512; with a padded stride gemm12 gives 16-17 ms (section 10).

Run path: the resident server (gpuport/server.py over oido's exec_server2) runs nemo8 in 3.93 s wall for the 10 s clip,
kernel time 3.91 s, and the server process uses 0.03 CPU-s per run (the one-shot exec used 0.68 CPU-s, mostly setup).
A correction to section 11: the GPU CPU cost is 0.03 s with the resident server, not 0.68 s.

Still open: layers (2.7 s) need a faster matmul core; attention scores (0.6 s) need a different kernel design; the conv2 matmul
is inherent 1.4 GMAC; a real FFT would save about 0.1 s on the DFT part.

## 13. Dictation menu (2026-11-05)
- prefix+v > "Oido full GPU (front end too, kit: ~2 s per phrase)" now runs gpuport.nemo8.Nemo8Encoder (resident exec_server2)
  through oido-type.py --gpu-full. The old gpu_encoder2.py is no longer used by the menu; its file is kept.
- Rollback: cp ~/voice-chat/live-mic-stream/oido-type.py.bak-2026-11-05-gpufull-old ~/voice-chat/live-mic-stream/oido-type.py;
  cp ~/.tmux-voice-menu.conf.bak-2026-11-05 ~/.tmux-voice-menu.conf; then reload tmux (source-file ~/.tmux.conf).
- Same text as the old engine on speak, tarel40 and a 2 s slice (checked through OidoCLI.transcribe).
- Latency floor (MEASURED): gemm12 takes about 16 ms per matmul dispatch whether M is 51 or 251 (latency-bound per dispatch).
  The encoder has about 400 dispatches, so a 2 s phrase takes about 2.0 s, and a 10 s clip 3.9 s. The CPU path answers a short
  phrase in about 0.1 s. The menu option is for freeing the CPU, not for responsiveness.

