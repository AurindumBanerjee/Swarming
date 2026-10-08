RUN run03_original_code_library
Reproduction of run00 with the ORIGINAL code (FIX 4: all 50 particles warm, old resolver): library methods only; thresholds 0.05/0.045/0.04 (0.03 dropped), 20 runs. Replaces the never-started run03_original_code_repro.

Created:   2026-10-07 12:49:02 IST
Machine:   eeiitj123.iitj.ac.in  (Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.17)
CPU:       Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
CPUs:      64 logical; Thread(s) per core=2 Core(s) per socket=16 Socket(s)=2
Python:    3.10.18  NumPy 2.2.5  SciPy 1.15.3
BLAS:      name: mkl-sdl; version: '2023.1';
Threads:   OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 (exported by run.sh); particle loop is serial.
Timer:     time.perf_counter() (monotonic wall clock), per FIX 10 in the script

Parts / code actually run:
  library      code/library__run03_original_code_library.py  md5 4ed971bcdf721fae2bba6083c977f0c9  (source /DATA/Aurindum/Swarming/MinTime/Runs/run00_original_sep2026/code/ScratchBench.py md5 9cd1e2b12626b5ed4fdbabd95b9c6167)

Changes vs source (only these lines differ):
--- library
--- ScratchBench.py
+++ library__run03_original_code_library.py
@@ -114 +114 @@
-ROOT_OUT = "MinTime/ScratchBenchBaseline"
+ROOT_OUT = "MinTime/Runs/run03_original_code_library/output/library"
@@ -120 +120 @@
-TARGETS = [0.05, 0.045, 0.04, 0.03]
+TARGETS = [0.05, 0.045, 0.04]


Launch: cd /DATA/Aurindum/Swarming && nohup setsid bash /DATA/Aurindum/Swarming/MinTime/Runs/run03_original_code_library/run.sh > /DATA/Aurindum/Swarming/MinTime/Runs/run03_original_code_library/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
Part library START: 2026-10-08 07:23:40 IST  load: 0.36 0.82 0.98
Run plan 2026-10-08 11:34 IST: sequence_03_05_gated.sh cancelled (run03 unaffected); new order after run03: run06 (100% warm, library) -> run04 -> run05 via sequence_06_04_05.sh
NOTE 2026-10-08 12:28 IST: run07_gpubench2 (GPU job) runs concurrently with this run, pinned to CPUs 16-23 (NUMA node 1, the sweep uses node 0) with OMP/MKL/OPENBLAS threads = 1. Timings of this run between this time and run07's end (see run07 README) were taken with that job on the machine.
Part library END:   2026-10-08 12:51:33 IST  exit code 0
NOTE 2026-10-08: moved from MinTime/Runs/run03_original_code_library to runs/scratchbench/run03_original_code_library (repo restructure to the GPUSwarm layout). Paths printed above in this README and baked into code/*.py (ROOT_OUT) refer to the old location; nothing else changed.
NOTE 2026-10-08: KILLED by user request at 12:51 IST (SIGTERM to the python process); the 'exit code 0' on the Part library END line above is not trustworthy. Partial result: 0.05 complete for all four methods; at 0.045 numpy and solve complete, sm 16/20, iterative not started.
