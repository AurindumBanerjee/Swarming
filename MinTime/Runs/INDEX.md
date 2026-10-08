# ScratchBench runs

Each run has its own code files, outputs, logs and manifest; old runs are never edited. New runs: python new_run.py create ...

| run | what | code md5 (as run) | location |
|---|---|---|---|
| run00_original_sep2026 | Original Sep 2026 baseline: FIX 4 warm start (all 50 particles identical in carried genes), old port resolver. Code + outputs copied. | ScratchBench 9cd1e2b1, ScratchBench2 7fc8245a | Runs/run00_original_sep2026/ |
| run01_hybrid_jitter0.02_partial | Current code (FIX 11-13: 40% warm, elite + jitter 0.02), library part only, KILLED partway (numpy/solve 0.05 complete, sm 0.05 18/20). Pre-convention run; scripts named in its README. | ScratchBench_library 1b74422d | Runs/run01_hybrid_jitter0.02_partial/ |
| run02_naive_jitter0 | Current code with WARM_START_JITTER = 0.0 (40% identical warm block + 60% random; = the 'naive' arm of GPUSwarm/Comparisons). Library then pure_python. library part finished; its pure_python part became run04 (pre-convention run). | ScratchBench_library_naive b3efb87d, ScratchBench2_purepython_naive 2b6aa56a | Runs/run02_naive_jitter0/ |
| run03_original_code_repro | Reproduction of run00 with the ORIGINAL code (FIX 4: all 50 particles warm, old resolver); thresholds 0.05/0.045/0.04 (0.03 dropped), 20 runs; library then pure_python (full sweep). | library 4a3e65a7, pure_python 17270def | Runs/run03_original_code_repro/ |
| run03_original_code_library | Reproduction of run00 with the ORIGINAL code (FIX 4: all 50 particles warm, old resolver): library methods only; thresholds 0.05/0.045/0.04 (0.03 dropped), 20 runs. Replaces the never-started run03_original_code_repro. | library 4ed971bc | Runs/run03_original_code_library/ |
| run04_naive_purepython_5runs | pure_python part of the run02 code (current code, WARM_START_JITTER = 0.0, naive warm start), 5 runs per threshold (PURE_PYTHON_NUM_RUNS = 5), thresholds 0.05/0.045/0.04. | pure_python 30c7376b | Runs/run04_naive_purepython_5runs/ |
| run05_original_purepython_5runs | pure_python part with the ORIGINAL code (run00 ScratchBench2.py, FIX 4), 5 runs per threshold (PURE_PYTHON_NUM_RUNS = 5), thresholds 0.05/0.045/0.04. | pure_python 3b5ad44b | Runs/run05_original_purepython_5runs/ |
| run03_original_code_repro | SUPERSEDED before start (see run03_original_code_library, run05_original_purepython_5runs) | - | Runs/run03_original_code_repro/ (empty output) |
