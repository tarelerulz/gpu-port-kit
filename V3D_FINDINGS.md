# What I learned running a speech model on the Raspberry Pi 4/400 GPU (V3D 4.2)

DRAFT, 2026-11-12. Numbers are from one Pi 400 (BCM2711, V3D 4.2 at 600 MHz, kernel 6.12, Mesa v3dv 25.x), measured with the scripts in this repo.
The Pi 4 has the same GPU generation but I have not tested it there.

**Setting.** A 13M-parameter speech recognition encoder (NeMo FastConformer "nemo8") ported to the V3D GPU with Vulkan compute shaders, then partly
re-written as hand-assembled QPU kernels. 10 s of audio went from 2.24 s (first working version) to 1.05 s (tuned Mesa shaders) to 0.77 s (hand-written
matmul kernels). The CPU path (3 NEON threads) takes 0.56 s, so the GPU is still slower than the CPU on this machine; the point of the GPU path is that it
leaves the CPU cores free (0.04 CPU-seconds per run vs 1.5).

## 1. The compiler, not the hardware, limits a Vulkan matmul on V3D

- A GLSL matmul tile of 12 outputs x 4 rows per invocation (48 accumulators) runs at **6.1 GMAC/s**. The theoretical peak at 600 MHz is about 19 GMAC/s
  (8 QPUs x 4 lanes x 0.6 GHz, one fused multiply-add per lane per cycle through the dual-issue add and mul pipes).
- The compiled K loop (dumped with `V3D_DEBUG=cs,qpu`, shader cache disabled) is **114 instructions for 48 multiply-adds**: the 48 fmul/fadd pairs dual-issue,
  but there are 16 separate `ldtmu` instructions, about 79 nops, per-lane `.ifa` masking caused by an early `return`, and a signed divide in the address
  arithmetic. Using unsigned addressing (`uint`) removes the divide: +3%.
- Things I tried that were **slower**: no early return (the 48 sums then spill to memory: 1.2 GMAC/s), software-pipelining the loads (0.8-2.3 GMAC/s, extra live
  registers spill), smaller tiles (more loads per result), a workgroup-uniform output-column layout (2.1 GMAC/s), `subgroupBroadcastFirst` (the driver fails to
  pack the instruction). Anything that adds live values to a 48-accumulator tile loses.
- Workgroup size: 256 invocations was best for every shader I swept; sizes below 256 were slower (roughly in proportion).
- Dispatch overhead is small (about 50 us per dispatch with a barrier); batching small matmuls into one dispatch (workgroup-y as batch index) still helped
  (120 conv matmuls -> 3 dispatches, 43 -> 30 ms).

## 2. Hand-written QPU code is feasible, and worth about 1.6x on matmul

[py-videocore6](https://github.com/Idein/py-videocore6) assembles and runs QPU programs through the DRM interface and works on a current kernel.

- Practical gotchas: it needs `numpy` 2.3.x (2.5 breaks its `NDArray` subclass); a program started while a Vulkan process is alive gets `EACCES` on submit if it
  opens `/dev/dri/card0` (v3dv holds card0 as DRM master) - open **`/dev/dri/renderD128`** instead; small immediates only cover -16..15 (16 must be built
  with a shift, and an instruction using a small immediate cannot also carry the `ldunifa` signal); in my kernel the `ldtmu` results go to r0, r1, r2 and r4.
- Kernel design (out[n][m] = bias[n] + sum_k W[k][n] * A[k][m], A and out "k-major", m contiguous): lanes run over m, each thread holds 2 vectors (32 rows) and a
  tile of 16 outputs = 32 accumulators in rf0-31; one MAC instruction is `fadd(prev) + fmul` chained through r3 (full dual issue); the activations come through the
  TMU requested two k steps ahead (two extra instructions per k step for the `tmua` writes); **the weights come from the uniform stream** (`unifa` write, then the
  `ldunifa` signal, which uses no ALU slot); 8 threads (one per QPU); the K loop is unrolled by two and branches with its three delay slots filled by MACs.
  Tails (rows beyond M) are lane-predicated with `ifa`/`ifb` flags so no store leaves the matrix.
- Speed on the real shapes (K=176..1584, N=176..704, M=251): **7.2 GMAC/s with f32 weights, 10.0 GMAC/s with f16 weights**, against 6.1 for the Mesa shader.

## 3. The bottleneck of the hand-written kernel is the uniform read rate

Timing-only experiments (wrong results, same instruction stream), K=2816, N=176:

| weight signal | GMAC/s |
|---|---|
| every 2 instructions (f32 weights, the real kernel) | 7.2 |
| every 4 instructions | 10 |
| every 8 instructions | 11.5 |
| none (pure ALU/TMU loop) | 13-15 |

Plain `ldunif` was no faster than `ldunifa`. Packing **two f16 weights per 32-bit word** and unpacking them with the multiply's input modifiers (`.l` / `.h`)
halves the uniform reads with no extra instructions: that is the 10 GMAC/s variant. Accuracy of f16 weights on this model: greedy word error rate 6.57% (44 of 670
words, clean synthetic speech) for both f32 and f16 weights; frame-level argmax agreement 99.96%; four real clips transcribe identically.

## 4. Mixing Vulkan and raw QPU jobs in one pipeline

The hand-written kernels run as raw V3D CSD jobs next to Vulkan dispatches working on the same buffers. What works:

1. Allocate the Vulkan buffers with `VkExportMemoryAllocateInfo` (dma-buf; the device exposes `VK_KHR_external_memory_fd` and `VK_EXT_external_memory_dma_buf`), export
   with `vkGetMemoryFdKHR`, import into a second DRM fd with `DRM_IOCTL_PRIME_FD_TO_HANDLE`, get the GPU address with `DRM_IOCTL_V3D_GET_BO_OFFSET`. The QPU job
   reads and writes the Vulkan data directly.
2. **Do not rely on submission order.** `vkQueueSubmit` can return before v3dv's submit thread has queued all kernel jobs, so a raw CSD job submitted right after
   it can run in the middle of the Vulkan work (half-updated inputs; it stayed wrong even with a 20 ms delay).
3. Use explicit sync: Vulkan segment signals a `SYNC_FD` semaphore -> `vkGetSemaphoreFdKHR` -> `DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE` (import sync file) -> CSD submit with
   `in_sync` / `out_sync` -> `SYNCOBJ_HANDLE_TO_FD` (export sync file) -> `vkImportSemaphoreFdKHR` into the next segment's wait semaphore. 98 of 98 stress rounds were
   correct with no CPU wait on the GPU path.
4. The hand-off costs nothing measurable: a plan with 32 one-tile QPU jobs ran 931 ms vs 937 ms for the same plan with the matmuls removed. Back-to-back raw CSD
   jobs cost about 0.08 ms each (limited by the Python submit loop); CSD jobs from one fd execute in submission order (the UAPI header says so).

## 5. What it bought on the real model (10 s clip, same text in every case)

| version | GPU time |
|---|---|
| first working GPU port | 2.24 s |
| tuned Mesa shaders (k-major layers, attention as matmuls, two-stage DFT, batching) | 1.05 s |
| + hand-written f16 matmul kernels for all large matmuls, hybrid runner | 0.77 s |

Smaller clips gain too (5 s: 0.51 -> 0.38 s). The Mesa path remains for what is left (front-end elementwise shaders, softmax, layer norm, head).

## 6. Caveats

- One model, one board. The f16-weight accuracy result is for this model; I did not test noisy speech or a beam-search decoder.
- Not a general solution: it needs a custom assembler, raw DRM submission and a hand-built weight format. It only pays off for compute-bound matmuls; memory-bound
  workloads such as LLM token generation are not helped (the GPU's memory reads are about the same as the CPU's).
- The GPU path is still slower than four NEON cores on this board.

## 7. Where the code is

`runner/exec_hybrid.c` (Vulkan + CSD runner), `qpu/qpu_mm.py` (the assembler kernels), `qpu/` tests (`hybrid_lengths.py`, `stress_test.py`, the dma-buf and
back-to-back-submit checks), `shaders/` (GLSL), notes in `GPU_MAP.md`. Reproduce the matmul numbers with `qpu/qpu_mm.py test` (needs py-videocore6 on `PYTHONPATH`).
