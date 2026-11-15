# Hand-written QPU matmul (experiment, 2026-11-12)

`qpu_mm.py` assembles the kernel with [py-videocore6](https://github.com/Idein/py-videocore6) (needs numpy 2.3.x; set PYTHONPATH to its `src`) and writes `shaders/qpu/mm16.qpu`.
`runner/exec_hybrid.c` (`tools/build_exec_hybrid.sh`) runs the nemo8 plan with `qpu_mm` ops as raw V3D CSD jobs, ordered with the Vulkan segments by sync files.
Enable with `GPUPORT_ASM=1`. `hybrid_test.py`, `overhead_test.py`, `stress_test.py` are the checks; `check1.py` + `vkshare.c` / `check2.py` are the dma-buf sharing and back-to-back-submit tests.
Results and notes: ~/doc/benchmarks/qpu-asm-matmul-2026-11-12/NOTE.md (10 s clip 1055 -> 1012 ms with the FF1 matmuls on this kernel, same text).
