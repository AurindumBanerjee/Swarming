RUN run06_warm100_jitter0_library
Current code (FIX 11-13 resolver/reset) with a 100% warm start: WARM_START_FRACTION = 1.0, WARM_START_JITTER = 0.0, i.e. all 50 particles carry the previous stage's best in the carried genes (no random particles, no jitter). Same algorithm as the original FIX 4 except for the port-resolver fix (compare run03). Library methods, thresholds 0.05/0.045/0.04, 20 runs.

Created:   2026-10-08 11:34:07 IST
Machine:   eeiitj123.iitj.ac.in  (Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.17)
CPU:       Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
CPUs:      64 logical; Thread(s) per core=2 Core(s) per socket=16 Socket(s)=2
Python:    3.10.18  NumPy 2.2.5  SciPy 1.15.3
BLAS:      name: mkl-sdl; version: '2023.1';
Threads:   OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 (exported by run.sh); particle loop is serial.
Timer:     time.perf_counter() (monotonic wall clock), per FIX 10 in the script

Parts / code actually run:
  library      code/library__run06_warm100_jitter0_library.py  md5 0a6f7aed77803a90f62c43f1e79498c0  (source /DATA/Aurindum/Swarming/MinTime/Runs/run02_naive_jitter0/code/ScratchBench_library_naive.py md5 b3efb87d84bdbf485b843912109ba2b7)

Changes vs source (only these lines differ):
--- library
--- ScratchBench_library_naive.py
+++ library__run06_warm100_jitter0_library.py
@@ -120 +120 @@
-ROOT_OUT = "MinTime/ScratchBenchRerun_20261006/library"
+ROOT_OUT = "MinTime/Runs/run06_warm100_jitter0_library/output/library"
@@ -134 +134 @@
-WARM_START_FRACTION = 0.4
+WARM_START_FRACTION = 1.0


Launch: cd /DATA/Aurindum/Swarming && nohup setsid bash /DATA/Aurindum/Swarming/MinTime/Runs/run06_warm100_jitter0_library/run.sh > /DATA/Aurindum/Swarming/MinTime/Runs/run06_warm100_jitter0_library/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
NOTE 2026-10-08: moved from MinTime/Runs/run06_warm100_jitter0_library to runs/scratchbench/run06_warm100_jitter0_library (repo restructure to the GPUSwarm layout). Paths printed above in this README and baked into code/*.py (ROOT_OUT) refer to the old location; nothing else changed.
STATUS 2026-10-08: NEVER EXECUTED (cancelled before start). The code copy bakes an output path under MinTime/Runs/; to run this configuration use reference/new_run.py (new run id).
