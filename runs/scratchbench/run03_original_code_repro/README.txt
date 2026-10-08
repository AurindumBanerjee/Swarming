RUN run03_original_code_repro
Reproduction of run00 with the ORIGINAL code (FIX 4: all 50 particles warm, old resolver); thresholds 0.05/0.045/0.04 (0.03 dropped), 20 runs; library then pure_python (full sweep).

Created:   2026-10-07 12:25:50 IST
Machine:   eeiitj123.iitj.ac.in  (Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.17)
CPU:       Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
CPUs:      64 logical; Thread(s) per core=2 Core(s) per socket=16 Socket(s)=2
Python:    3.10.18  NumPy 2.2.5  SciPy 1.15.3
BLAS:      name: mkl-sdl; version: '2023.1';
Threads:   OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 (exported by run.sh); particle loop is serial.
Timer:     time.perf_counter() (monotonic wall clock), per FIX 10 in the script

Parts / code actually run:
  library      code/library__run03_original_code_repro.py  md5 4a3e65a73631460f8ecf97db32eb3677  (source /DATA/Aurindum/Swarming/MinTime/Runs/run00_original_sep2026/code/ScratchBench.py md5 9cd1e2b12626b5ed4fdbabd95b9c6167)
  pure_python  code/pure_python__run03_original_code_repro.py  md5 17270def1aa8da2bda472eb693845e16  (source /DATA/Aurindum/Swarming/MinTime/Runs/run00_original_sep2026/code/ScratchBench2.py md5 7fc8245a34307ccbcf5b046a3b988220)

Changes vs source (only these lines differ):
--- library
--- ScratchBench.py
+++ library__run03_original_code_repro.py
@@ -114 +114 @@
-ROOT_OUT = "MinTime/ScratchBenchBaseline"
+ROOT_OUT = "MinTime/Runs/run03_original_code_repro/output/library"
@@ -120 +120 @@
-TARGETS = [0.05, 0.045, 0.04, 0.03]
+TARGETS = [0.05, 0.045, 0.04]

--- pure_python
--- ScratchBench2.py
+++ pure_python__run03_original_code_repro.py
@@ -114 +114 @@
-ROOT_OUT = "MinTime/ScratchBenchBaseline"
+ROOT_OUT = "MinTime/Runs/run03_original_code_repro/output/pure_python"
@@ -120 +120 @@
-TARGETS = [0.05, 0.045, 0.04, 0.03]
+TARGETS = [0.05, 0.045, 0.04]


Launch: cd /DATA/Aurindum/Swarming && nohup setsid bash /DATA/Aurindum/Swarming/MinTime/Runs/run03_original_code_repro/run.sh > /DATA/Aurindum/Swarming/MinTime/Runs/run03_original_code_repro/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
Queued 2026-10-07 12:26 IST: starts automatically after run02 chain (pid 141088) exits, via queue_after.sh.

SUPERSEDED 2026-10-07 12:49 IST BEFORE ANY RUN STARTED: its queue was cancelled; replaced by run03_original_code_library,
run04_naive_purepython_5runs and run05_original_purepython_5runs (pure_python is now 5 runs per threshold). Nothing was ever written to output/.
NOTE 2026-10-08: moved from MinTime/Runs/run03_original_code_repro to runs/scratchbench/run03_original_code_repro (repo restructure to the GPUSwarm layout). Paths printed above in this README and baked into code/*.py (ROOT_OUT) refer to the old location; nothing else changed.
STATUS 2026-10-08: NEVER EXECUTED (cancelled before start). The code copy bakes an output path under MinTime/Runs/; to run this configuration use reference/new_run.py (new run id).
