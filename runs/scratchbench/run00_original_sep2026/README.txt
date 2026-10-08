RUN run00_original_sep2026 -- the ORIGINAL ScratchBench baseline (Sep 2026), copied here for reference.

code/ScratchBench.py   md5 9cd1e2b12626b5ed4fdbabd95b9c6167   (library methods: numpy, solve, sm, iterative)
code/ScratchBench2.py  md5 7fc8245a34307ccbcf5b046a3b988220   (pure_python)
output/                exact copy of MinTime/ScratchBenchBaseline (1213 files; verified identical with diff -rq)
The originals (/DATA/Aurindum/Swarming/ScratchBench*.py and MinTime/ScratchBenchBaseline) were NOT moved or changed.

How the code was identified: both scripts have mtime 2026-09-03 10:10; the first log in the outputs
(numpy 0.05 run 1) starts at 2026-09-03 10:11:46; neither file changed afterwards (the pure_python runs ended 2026-09-06);
Swarming.zip on the server holds byte-identical copies; the logs' RUN_START line has no warm_start/jitter fields,
which only the later script version writes.

Algorithm of this code (FIX 4 warm start): from stage 2 on, ALL 50 particles copy the previous stage's best in the
carried genes (only the two genes of the new capacitor are random) -> carried genes frozen, greedy forward selection.
Port-collision resolver has the pre-FIX-12 early break. Thresholds in the outputs: 0.05 0.045 0.04 0.03, 20 runs each
(pure_python 0.03 only 5 runs), timer time.perf_counter(). Thread settings of that run were not recorded.
NOTE 2026-10-08: moved from MinTime/Runs/run00_original_sep2026 to runs/scratchbench/run00_original_sep2026 (repo restructure to the GPUSwarm layout). Paths printed above in this README and baked into code/*.py (ROOT_OUT) refer to the old location; nothing else changed.
