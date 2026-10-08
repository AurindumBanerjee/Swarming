RUN scratchbench/run08_warm100_5runs
Current code with a 100% warm start (WARM_START_FRACTION = 1.0, WARM_START_JITTER = 0.0: all 50 particles carry the previous stage best, no random particles, no jitter), library methods numpy/solve/sm/iterative, thresholds 0.05/0.045/0.04, 5 runs per threshold. Same algorithm as the original FIX 4 code except the port-resolver (FIX 12) and stage-reset (FIX 13) fixes. Replaces the 20-run run06.

Created: 2026-10-08T12:53:56+0530  on eeiitj123.iitj.ac.in
Source: runs/scratchbench/run06_warm100_jitter0_library/code/library__run06_warm100_jitter0_library.py  (git None, md5 0a6f7aed77803a90f62c43f1e79498c0)
As run:  code/library__run06_warm100_jitter0_library__run08_warm100_5runs.py  (md5 e42ba235f2fdcb4b1f6d7b2e2bfadac0)

Changes vs source (only these lines differ):
--- library__run06_warm100_jitter0_library.py
+++ library__run06_warm100_jitter0_library__run08_warm100_5runs.py
@@ -120 +120 @@
-ROOT_OUT = "MinTime/Runs/run06_warm100_jitter0_library/output/library"
+ROOT_OUT = "runs/scratchbench/run08_warm100_5runs/output"
@@ -123 +123 @@
-NUM_RUNS = 20
+NUM_RUNS = 5

Execution: args ''; env none; taskset 0-7; conda env work

Launch: nohup setsid bash runs/scratchbench/run08_warm100_5runs/run.sh > runs/scratchbench/run08_warm100_5runs/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
NOTE 2026-10-08 12:53 IST: run07_gpubench2 (GPU job, pinned to CPUs 16-23) runs concurrently; this run is pinned to CPUs 0-7 (NUMA node 0) with BLAS/OMP threads = 1.
START: 2026-10-08 12:54:05 IST  host: eeiitj123.iitj.ac.in  load: 1.20 1.68 1.75
END:   2026-10-08 20:22:07 IST  exit code 0
NOTE 2026-10-08: finished 2026-10-08 20:22 IST (7.5 h, exit code 0). Ran concurrently with run07 (GPU job) -- separate cores (this run 0-7) and the GPU is not used here.
RESULT (summary/summary.txt; 5 runs per threshold, exact methods numpy/solve/sm agree):
  * 0.05: 5/5, 5.0 caps; 0.045: 5/5, 6.6 caps; 0.04: 0/5 (best minZ just above the target, e.g. 0.04002 in run 2; 0.04029 and 0.04164 in runs 1 and 3). iterative (approximation): 1/5 at 0.04.
  * Run-by-run vs run00 (original code, same seeds, runs 1-5): at 0.05 and 0.045 caps and minZ are IDENTICAL in 5/5 runs for every exact method
    (and 5/5, 4/5 for iterative), i.e. the 100% warm start reproduces the original search while the placements stay distinct.
  * At 0.04 they diverge: run00 succeeded in 18/20 runs, but in ALL 20 of its 0.04 runs the final placement had fewer distinct ports than
    capacitors (e.g. 15 capacitors on 12 pads); the old port resolver (FIX 12) left duplicate ports, which this run's resolver never does
    (0/5 here; 0/20 at 0.05, 1/20 at 0.045 in run00). The old 0.04 successes therefore relied on duplicate-port placements.
