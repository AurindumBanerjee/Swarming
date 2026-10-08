RUN run05_original_purepython_5runs
pure_python part with the ORIGINAL code (run00 ScratchBench2.py, FIX 4), 5 runs per threshold (PURE_PYTHON_NUM_RUNS = 5), thresholds 0.05/0.045/0.04.

Created:   2026-10-07 12:49:03 IST
Machine:   eeiitj123.iitj.ac.in  (Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.17)
CPU:       Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
CPUs:      64 logical; Thread(s) per core=2 Core(s) per socket=16 Socket(s)=2
Python:    3.10.18  NumPy 2.2.5  SciPy 1.15.3
BLAS:      name: mkl-sdl; version: '2023.1';
Threads:   OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 (exported by run.sh); particle loop is serial.
Timer:     time.perf_counter() (monotonic wall clock), per FIX 10 in the script

Parts / code actually run:
  pure_python  code/pure_python__run05_original_purepython_5runs.py  md5 3b5ad44b20e7019137a09cd7d17ed243  (source /DATA/Aurindum/Swarming/MinTime/Runs/run00_original_sep2026/code/ScratchBench2.py md5 7fc8245a34307ccbcf5b046a3b988220)

Changes vs source (only these lines differ):
--- pure_python
--- ScratchBench2.py
+++ pure_python__run05_original_purepython_5runs.py
@@ -114 +114 @@
-ROOT_OUT = "MinTime/ScratchBenchBaseline"
+ROOT_OUT = "MinTime/Runs/run05_original_purepython_5runs/output/pure_python"
@@ -120 +120 @@
-TARGETS = [0.05, 0.045, 0.04, 0.03]
+TARGETS = [0.05, 0.045, 0.04]
@@ -155 +155 @@
-PURE_PYTHON_NUM_RUNS = None    # e.g. 3 to only run 3 of the NUM_RUNS reps
+PURE_PYTHON_NUM_RUNS = 5


Launch: cd /DATA/Aurindum/Swarming && nohup setsid bash /DATA/Aurindum/Swarming/MinTime/Runs/run05_original_purepython_5runs/run.sh > /DATA/Aurindum/Swarming/MinTime/Runs/run05_original_purepython_5runs/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
NOTE 2026-10-08: moved from MinTime/Runs/run05_original_purepython_5runs to runs/scratchbench/run05_original_purepython_5runs (repo restructure to the GPUSwarm layout). Paths printed above in this README and baked into code/*.py (ROOT_OUT) refer to the old location; nothing else changed.
STATUS 2026-10-08: NEVER EXECUTED (cancelled before start). The code copy bakes an output path under MinTime/Runs/; to run this configuration use reference/new_run.py (new run id).
