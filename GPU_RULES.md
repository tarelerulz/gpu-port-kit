# Rules for running compute on the Raspberry Pi 4/400 GPU (V3D 4.2, Vulkan)

DRAFT. Everything here was measured on one Raspberry Pi 400 (BCM2711, V3D 4.2) with Mesa v3dv 25.3.x and Linux 6.12.
Nothing was checked on a Pi 5 (different GPU), so the speed rules may not carry over. Each rule says where it comes from.
For the compiler and hand-written QPU findings see [V3D_FINDINGS.md](V3D_FINDINGS.md).

## 0. Is a port worth it?

- On this board the GPU is about 2-8x slower than the CPU's int8 NEON path for the same model. Port to free CPU cores, or to
  match CPU speed, not to beat it.
- Good fit: f32 or f16 weights, large matmuls (many frames per matmul), few separate steps per block.
- Bad fit: 8/4-bit weights (no fast int8/int4 on V3D), tiny models (overhead dominates), per-token decoding (one row per step).
- A hybrid (CPU and GPU share one matmul by rows) can beat either alone, but a layer-by-layer split did not help.

## 1. Method

1. Write a numpy reference and check it against the CPU output on two clips (a short phrase and a 10 s clip). Compare token
   sequences, not only the final text.
2. Write one shader per op type and test each alone against numpy on random data. Expect max diff around 1e-6.
3. Build the plan (arena offsets, dispatch sizes, weight layouts) and run plan prefixes: the first N ops, then read one arena
   region. The first region that differs from numpy is the first wrong stage.
4. Only after it is correct, time each op type alone, change one thing, and re-check accuracy.
5. Keep a dated log of every round (rules and numbers) and a backup of each file you change.

## 2. Correctness traps (each cost real debugging time)

- `local_size` is a literal in the `.comp` file. The pipeline's local-size argument does not change it. Keep your dispatch
  maths in sync with the literal.
- One dispatch dimension is limited to about 65535 groups. Split big launches with `gid.x + gid.y * stride`.
- Launches with fewer than about 24 workgroups gave wrong output for one matmul shader at some shapes. Use `max(24, needed)`
  groups and let the extra groups return at their guards.
- That same matmul shader was only correct for row blocks of 251 rows or fewer. Test shapes at and above your real sizes.
- A matmul output placed just below its input (closer than about 100k floats) read back wrong. Leave a gap (131072 floats here)
  after each arena region.
- Unsigned arithmetic wraps: `T - 1 - r` goes huge when `r > T - 1`. Use signed ints for positions.
- `vec4` loads need 4-aligned offsets, so keep every row stride a multiple of 4.
- Weight blobs: advance the read position past variable-length fields (for example a token list) before reading later fields.
- Buffer sizes must cover the largest offset, including transposed or padded weight copies.

## 3. Speed rules (measured)

- On V3D, removing whole ops or jobs wins. Adding threads or dispatches loses.
- Dispatch overhead is small (about 40 us each; 400 trivial dispatches took 16 ms). Do not optimise dispatch count first.
- Contiguous reads (adjacent threads, adjacent addresses) are fast. Strides of a row or weight stride were about 10x slower.
- Registers: 12 `vec4` accumulators per thread was slow, and 88 spilled. Loop over chunks and keep few values live.
- More blocking per thread (2x2, 4x4) was not always faster. Measure each shape.
- Split-K with a shared-memory reduction cost most of the time (about 18 of 22 ms per matmul step). For small M, one thread per
  4 outputs x 4 frames with the full K loop and direct stores was much faster (18 to 7.7 ms for one matmul).
- Variable-index shared arrays are slow. Fixed-index code is faster.
- Fusing residual adds and activations removes steps but gave no speed gain when the matmul dominates.
- Arena size (16 MB vs 96 MB) did not change speed.
- Tried and rejected here: splitting layer norm into two dispatches, deeper prefetch, K-unrolling, smaller workgroups, vec2
  softmax. Details are in the project notes.

## 4. Hangs: a hung job can report success (read this one)

On the Pi 4/400, if a compute job never finishes, the kernel resets the GPU and logs `Resetting GPU for hang`, but without
the patches below Vulkan returns `VK_SUCCESS` from `vkWaitForFences` and the output buffer holds garbage.

Why (found by reading the code, then confirmed by test):

- Kernel `drivers/gpu/drm/v3d`: a job only checks its own finished fence before it runs. A failed dependency is not checked,
  so the no-op job that carries your `VkFence` runs and signals success. Two dependencies matter: the previous job of the same
  submit, and a syncobj the job waits on from an earlier submit. Checking only the first one is not enough (tested: that
  variant still returned success).
- Mesa `vk_drm_syncobj.c`: after a successful wait the sync-file status is not read, so an error on a signalled fence is
  reported as success.

What to do:

- Test your own setup: dispatch a shader that loops forever, call `vkWaitForFences`, and expect `VK_ERROR_DEVICE_LOST` (-4).
  Then run a normal job to check the GPU recovered.
- Until both fixes are in your kernel and Mesa, check the kernel log for `Resetting GPU for hang` (or a reset counter) around
  every GPU run, and discard output from a run that reset.
- Do not write a single-thread loop over many shared partials. One such shader hung the GPU until the server was killed.
- Run a memory guard when developing: check for stray writes, unwritten output and NaN poison (the kit's `check.guard_run`).
- Status: Mesa issue mesa/mesa#16960 filed; kernel patch sent to dri-devel on 2026-11-20 (compile-tested on drm-misc-next,
  runtime-tested as an equivalent change on 6.12); loom.cpp 1.0.0 ships a reset-counter workaround. Patches in `patches/`.

## 5. System setup

- GPU clock: 700 MHz needs `over_voltage=2`. 700 without it did not boot. Try a new clock with `tryboot` first and never put
  an untested clock in `config.txt`. 750 (ov4) and 800 (ov6) ran the benchmark cleanly in short tests; long-run thermals were
  not tested.
- CPU governor: for short GPU jobs after an idle gap, `ondemand` added about 8-16 ms. `performance` removed it, `schedutil`
  was worse. Performance costs heat and power, so it is your call.
- A keep-awake thread (touching the GPU while idle) removes most of the first-job penalty.

## 6. Checklist for the next model

1. Check the fit rule (section 0).
2. Numpy reference matches CPU tokens on two clips.
3. A shader per op type, each tested alone.
4. A plan builder with offsets, gaps, dispatch minimums and weight layouts.
5. Prefix-check to find the first wrong stage before tuning.
6. Time per op type. Prefer no-split matmuls for small M.
7. Run the planted-hang test and watch the reset counter (section 4).
8. Re-check accuracy after every change.
