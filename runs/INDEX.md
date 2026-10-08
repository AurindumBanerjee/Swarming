# Swarming runs

Every run has its own folder: `runs/<experiment>/<run_id>/` with code/ (exact copy), output/ and logs/ (raw, git-ignored),
summary/ and plots/ (tracked), run.sh, manifest.json, README.txt. New runs: `python reference/new_run.py` (see reference/README.md).
Runs up to run07 predate the layout: moved here from MinTime/Runs on 2026-10-08 (run07 follows when it finishes).

| run | status | start | what | code |
|---|---|---|---|---|
| scratchbench/run00_original_sep2026 | finished | 2026-09-03 | Original baseline: FIX 4 warm start (all 50 particles identical in carried genes), old port resolver; library + pure_python, 4 thresholds. Code + outputs + summary + plots. | ScratchBench 9cd1e2b1, ScratchBench2 7fc8245a |
| scratchbench/run01_hybrid_jitter0.02_partial | stopped | 2026-10-06 | FIX 11-13 code (40% warm, elite + jitter 0.02), library only; killed partway (0.05 numpy/solve complete, sm 18/20). | ScratchBench_library 1b74422d |
| scratchbench/run02_naive_jitter0 | finished (library) | 2026-10-06 | Same code, WARM_START_JITTER = 0.0 (naive warm block: 20 identical + 30 random); library; 0.04 never reached. | ScratchBench_library_naive b3efb87d |
| scratchbench/run03_original_code_library | stopped | 2026-10-08 | Original code (FIX 4), library, 20 runs; killed 2026-10-08 12:51 (0.05 complete; 0.045 numpy/solve complete, sm 16/20). | library__run03 4ed971bc |
| scratchbench/run03_original_code_repro | never started | - | Superseded by run03_original_code_library. | - |
| scratchbench/run04_naive_purepython_5runs | never executed | - | pure_python of the run02 code, 5 runs per threshold; cancelled. | pure_python__run04 30c7376b |
| scratchbench/run05_original_purepython_5runs | never executed | - | pure_python of the original code, 5 runs per threshold; cancelled. | pure_python__run05 3b5ad44b |
| scratchbench/run06_warm100_jitter0_library | never executed | - | 100% warm start, library, 20 runs; replaced by run08 (5 runs). | library__run06 0a6f7aed |
| MinTime/Runs/run07_gpubench2 | running | 2026-10-08 | GPUBench2.py on the GPU, pinned to CPUs 16-23; moves to runs/ when finished. | gpubench2__run07 f1c96e89 |
