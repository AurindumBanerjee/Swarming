RUN run04_naive_purepython_5runs
pure_python part of the run02 code (current code, WARM_START_JITTER = 0.0, naive warm start), 5 runs per threshold (PURE_PYTHON_NUM_RUNS = 5), thresholds 0.05/0.045/0.04.

Created:   2026-10-07 12:49:03 IST
Machine:   eeiitj123.iitj.ac.in  (Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.17)
CPU:       Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
CPUs:      64 logical; Thread(s) per core=2 Core(s) per socket=16 Socket(s)=2
Python:    3.10.18  NumPy 2.2.5  SciPy 1.15.3
BLAS:      name: mkl-sdl; version: '2023.1';
Threads:   OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 (exported by run.sh); particle loop is serial.
Timer:     time.perf_counter() (monotonic wall clock), per FIX 10 in the script

Parts / code actually run:
  pure_python  code/pure_python__run04_naive_purepython_5runs.py  md5 30c7376bd4376cc1fa5e1bb33d9dd097  (source /DATA/Aurindum/Swarming/MinTime/ScratchBenchRerun_20261006/ScratchBench2_purepython_naive.py md5 2b6aa56ada456d60dad7e935bd20f0c0)

Changes vs source (only these lines differ):
--- pure_python
--- ScratchBench2_purepython_naive.py
+++ pure_python__run04_naive_purepython_5runs.py
@@ -120 +120 @@
-ROOT_OUT = "MinTime/ScratchBenchRerun_20261006/pure_python"
+ROOT_OUT = "MinTime/Runs/run04_naive_purepython_5runs/output/pure_python"
@@ -166 +166 @@
-PURE_PYTHON_NUM_RUNS = None    # e.g. 3 to only run 3 of the NUM_RUNS reps
+PURE_PYTHON_NUM_RUNS = 5


Launch: cd /DATA/Aurindum/Swarming && nohup setsid bash /DATA/Aurindum/Swarming/MinTime/Runs/run04_naive_purepython_5runs/run.sh > /DATA/Aurindum/Swarming/MinTime/Runs/run04_naive_purepython_5runs/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
