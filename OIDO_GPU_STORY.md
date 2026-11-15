# How the Oido speech model got fast on the Raspberry Pi 400 GPU

This is the whole story in plain words: what the project was, what was tried, what worked, what did not, and why.
It covers 2026-11-02 to 2026-11-13. Every number was measured on one Pi 400 with the scripts in this repo.
The short technical write-up for other Pi GPU programmers is `V3D_FINDINGS.md`; this file is the longer history.

## The one-paragraph version

Oido's speech model ("nemo8", 13 million weights) turns audio into text. On the Pi's CPU it needs about 0.55 s for
10 s of speech. The Pi's GPU (V3D 4.2) was expected to be much slower, and at first it was: 3.3 s. Over eleven days it
came down to 0.54 s of GPU time. The steps that mattered, in order: store the data so neighbouring GPU threads read
neighbouring memory; turn every heavy step into the same matrix multiply; replace the compiler's matrix multiply with
hand-written GPU assembly; raise the GPU clock; then remove wasted work, wasted jobs and idle GPU cores one at a time.
The GPU is now level with the CPU on a 10 s clip while leaving the CPU cores almost idle.

## 10 s of speech, GPU time, step by step

| date | what changed | GPU time |
|---|---|---|
| 11-03 | first version that gave the right text (encoder on GPU, front end on CPU) | 3.30 s |
| 11-05 | runner bug fixed (see "The bug that hid for two days"), whole model on GPU | 3.89 s |
| 11-09 | activations stored "k-major" so neighbouring threads share memory lines | 2.50 s |
| 11-12 | attention, convolutions and the DFT all rewritten as matrix multiplies; position tables computed once | 1.19 s |
| 11-12 | two-stage DFT, cheaper address arithmetic, batched launches | 1.06 s |
| 11-12 | hand-written assembly matrix multiply for all big matmuls | 0.77 s |
| 11-12 | GPU clock 600 -> 650 -> 700 MHz | 0.68 s |
| 11-12 | activation function folded into the matmul; second kernel shape for wide matmuls | 0.60 s |
| 11-13 | wasted attention work removed, tiny jobs merged, idle cores filled (details below) | 0.54 s |

CPU for comparison: 0.55 to 0.58 s, using three cores at full load. The GPU version uses about 0.05 s of CPU.
The side-by-side benchmark of 11-13 (`GPU_NEMO8_BENCH.md`, wall time, 9 runs each): 10 s clip GPU 0.55 s, CPU 0.58 s;
5 s clip GPU 0.30 s, CPU 0.28 s.

## Words used below

- **Matmul**: matrix multiply. About 85% of this model's work is matmuls. Speed is counted in GMAC/s (billions of
  multiply-adds per second). The model needs about 4.7 GMAC for 10 s of audio.
- **Frame**: the model works on one "frame" per 40 ms of audio. 10 s = 251 frames.
- **QPU**: one of the GPU's 8 processor cores. Each one works on 16 values at a time.
- **Mesa**: the open-source graphics driver. It compiles shader programs (GLSL) into QPU code.
- **Kernel**: a small program that runs on the GPU. "Hand-written kernel" = written directly in QPU assembly.
- **Job**: one launch of a kernel on the GPU. Launching costs about 0.09 ms even if the job does nothing.
- **Plan**: the list of about 400 operations the runner executes for one clip.

## Phase 1 (11-02 to 11-03): make it work at all

A desk review on 11-02 said the port was not worth doing: the GPU's best floating-point rate was about what one CPU
core does with 8-bit integers, and the CPU has four cores. That prediction was about right for the first version and
wrong for the last one.

The method that made the port possible:
1. Write the model in plain numpy first and prove it gives the same text as the C program.
2. Write one GPU shader per operation and test each alone against numpy.
3. Run the plan in prefixes (first N steps) and compare with numpy to find the first wrong step.

Result: right text, 3.3 s per 10 s clip, six times slower than the CPU. Putting the audio front end on the GPU too
made it 5.3 s.

## The bug that hid for two days (11-05)

Matmuls of more than 251 rows gave wrong answers, so everything was cut into 251-row pieces and a rule was written
down: "the shader is only correct up to 251 rows". The real cause was in the runner: it passed the wrong value for a
constant the shader read as its row count, so rows from 256 up were never computed. Fixing the runner removed the rule.
Lesson: a rule that says "it just breaks above N" usually means a bug, not a limit.

Also tested that day: 8-bit integer weights on the GPU. Same text, no speed gain (the GPU has no fast integer
multiply-add), so the GPU path stayed floating point.

## Phase 2 (11-09): put the data where the GPU wants it

The GPU runs 16 threads side by side. If thread 0 reads memory address X and thread 1 reads X+4, one memory fetch
serves both. If thread 1 reads X+700, every thread causes its own fetch. The activations were stored row by row
("row-major"), which gave the bad case. Storing them the other way round ("k-major": one row per channel, frames next
to each other) made the same matmul shader 2.6 to 20 times faster. Whole model: 3.89 -> 2.50 s.

This was the single biggest idea in the project: **on this GPU, memory layout matters more than arithmetic.**

## Phase 3 (11-12 morning): make everything a matmul

Once the matmul was fast, every slow step that was not a matmul became the problem. Each was rewritten as one:
- **Attention** (every frame looks at every other frame): scores and weighted sums became matmuls.
- **Position tables**: the same table was recomputed on every run. It only depends on clip length, so it is now
  computed once and stored with the weights.
- **Front-end convolutions**: rewritten as matmuls on regrouped data, with no copy step.
- **DFT** (audio -> frequencies): one big matmul became two small ones.
- All layer steps (layer norm, depthwise conv, softmax) rewritten to read and write k-major directly, so nothing
  had to be transposed in between.

Result: 2.24 -> 1.06 s. By now 85% of the time was inside one shader, the matmul, running at about 6 GMAC/s.

## Phase 4 (11-12 afternoon): hand-written assembly

The matmul shader could not be made faster, and reading the code the compiler produced showed why: 114
instructions for 48 multiply-adds. The hardware could do far better than the compiler.

So the matmul was written by hand in QPU assembly (`qpu/qpu_mm.py`, using the py-videocore6 assembler):
- One thread per QPU, 32 running sums kept in registers, multiply and add in the same instruction.
- Weights read as a stream. The stream turned out to be the limit, so weights are stored as 16-bit floats, two per
  word, which halves the reads. Accuracy check: same text, same error rate.
- The runner (`runner/exec_hybrid.c`) mixes these raw jobs with the normal Vulkan shaders in one pipeline, passing
  "this is finished" signals between them so nothing waits on the CPU.

Standalone: 10 GMAC/s against Mesa's 6. Whole model: 1.05 -> 0.77 s.

Then:
- **GPU clock**: 600 -> 700 MHz (with `over_voltage=2`; 700 without it did not boot). About 12% faster.
- **Activation folded in**: the "silu" function now runs at the end of the matmul job instead of as its own step.
- **Second kernel shape** (`qpu/qpu_mm4.py`): 64 rows by 8 columns per thread instead of 32 by 16. It halves the
  weight reads, which helps the wide matmuls (the big convolution and half of the feed-forward ones).

Result: 0.60 s.

## Phase 5 (11-13): stop wasting work

No single big idea was left, so the day went on finding where time was being wasted. What worked:

1. **Position term computed twice as much as needed.** For each block of 32 frames the job computed all 501
   position columns, then threw half away. Each core now computes only the columns it will keep.
2. **Too many tiny jobs.** Attention ran 192 jobs per clip (4 heads x 3 matmuls x 16 layers), each paying the
   0.09 ms launch cost. A kernel variant now loops over the 4 heads inside one job.
3. **Idle cores.** The kernel gives each core a block of rows. With 5, 6 or 7 blocks, only that many of the 8
   cores worked; with 9 blocks, a whole second pass ran for one block. Splitting such a matmul into two jobs
   (full blocks, then the leftover) keeps all cores busy. A 10.25 s clip went from 1040 ms to 648 ms.
4. **Leftover rows on the cheaper kernel.** A few rows past a 64-row block now run on the 32-row kernel.
5. **A whole step removed.** Attention needed the queries twice, with two different constants added. The
   difference only depends on the position, so it was added to the position table once, offline. The copy step
   ("att_prep") is gone and the matmul writes straight into the layout attention reads.
6. **Output layer and softmax.** The last matmul moved to the hand-written kernel; softmax went from three passes
   over its data to two.
7. **Short clips** were still using an older front end; the newer one now starts at 0.6 s.
8. **Time outside the GPU.** Each clip paid 33 ms for writing files to the SD card and reading the kernel log.
   Files now go to RAM and the log check is 5 times faster.

GPU time per clip after all of this (same text as the reference at every length):

| clip | morning of 11-13 | evening of 11-13 |
|---|---|---|
| 1.5 s | 190 ms | 152 ms |
| 2.3 s | 168 ms | 161 ms |
| 5 s | 296 ms | 282 ms |
| 6.25 s | 471 ms | 395 ms |
| 10 s | 598 ms | 541 ms |
| 10.25 s | 1040 ms | 648 ms |
| 13 s | 1394 ms | 853 ms |
| 17.9 s | 2060 ms | 1218 ms |

## What did not work (and what each one taught)

- **More threads or smaller work groups** (tried for softmax, layer norm, and others, several times): always
  slower. Splitting layer norm into two parallel passes: slower. On this GPU, *removing* a step or a job helps;
  adding parallelism does not.
- **Unrolling the inner loop, prefetching the next data early, batching the stores**: no change. The kernel is
  limited by memory reads, not by instruction count.
- **Reading activations further ahead**: wrong output. The GPU's load unit holds only about 8 pending reads.
- **16-bit float arithmetic, packed 8-bit weights in the Mesa shader**: slower or no gain.
- **The 64-row kernel for the narrow matmuls**: 5% slower there; the arithmetic showed no gain, so it was not wired.
- **Adding the position term into the scores inside the kernel**: would save about 3 ms and risks hanging the GPU.
- **One experiment hung the GPU** (a kernel asked for fewer loads than it read back). After that every result was
  wrong until a reboot. Every experiment since runs with a timeout and a check of the kernel log.

## What is still slow

- **Steps at block boundaries.** Cost goes up in jumps of 64 rows: 2.3 s costs 161 ms, 2.8 s costs 239 ms. The
  same happens just past 5.1 s and 10.2 s.
- **The big convolution** in the front end: 101 ms of a 10 s clip. It already runs near the kernel's limit.
- **Host overhead**: about 30 ms per clip outside the GPU, half of it building and writing the plan.
- The GPU still needs the CPU's 0.55 s beaten by a clear margin to be the better choice on speed alone. Its
  advantage today is that the CPU stays free.

## How to check any of this yourself

- `python3 tests/run_all.py`: all correctness checks (the model test needs the weights file).
- `GPUPORT_ASM=1 python3 qpu/hybrid_lengths.py`: every clip length against the reference: same tokens, time.
- `GPUPORT_ASM=1 python3 qpu/stress_test.py 20`: 20 runs of the 10 s clip must be bit-identical; prints the median.
- `python3 tools/exp/determinism.py 15 16000 44800 100000`: repeat runs at chosen lengths (sample counts).
- `python3 tools/exp/skip_by_op.py`: where the time goes, by removing one operation type at a time.
- `python3 tools/exp/launch_cost.py`, `host_cost.py`: the launch cost per job and the time outside the GPU.

Every change has an off switch (an environment variable) so it can be compared with the version before it:
`GPUPORT_ASM_SPLIT`, `GPUPORT_ASM_HEADS`, `GPUPORT_ASM_HEADS_ALL`, `GPUPORT_ASM_NOPREP`, `GPUPORT_ASM_TAIL`,
`GPUPORT_ASM_HEAD`, `GPUPORT_SM_ONLINE`, `GPUPORT_SK_FULL`, `GPUPORT_ASM4`, `GPUPORT_ASM_SILU`. Set to `0` (or `1`
for `GPUPORT_SK_FULL`) to get the older behaviour.

## Where things are

- `gpuport/nemo8.py`: builds the plan for a clip; `gpuport/ops.py`: one function per operation.
- `runner/exec_hybrid.c`: runs a plan (Vulkan shaders + raw QPU jobs).
- `qpu/qpu_mm.py`, `qpu/qpu_mm4.py`: the hand-written kernels. `shaders/qpu/*.qpu`: their machine code.
- `shaders/*.comp`: the GLSL shaders. `GPU_MAP.md`: how memory, plans and layouts work.
- The model weights are not in this repo.
