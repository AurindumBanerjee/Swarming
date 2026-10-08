RUN run07_gpubench2
GPUBench2 (GPU / PyTorch, complex128) on the 21-port benchmark PDN: methods numpy, solve, sm, woodbury, woodbury_batch, fast; thresholds 0.05/0.045/0.04 (0.03 dropped), 20 runs, 50 particles x 40 iterations (script default), hybrid warm start as coded in the script (40% warm, jitter 0.02). Ran CONCURRENTLY with the CPU ScratchBench sweeps, pinned to CPUs 16-23 (NUMA node 1) with BLAS/OMP pinned to 1 thread.

Created:   2026-10-08 12:28:13 IST
Machine:   eeiitj123.iitj.ac.in  (Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.17)
CPU:       Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
CPUs:      64 logical; Thread(s) per core=2 Core(s) per socket=16 Socket(s)=2
Python:    3.10.18  NumPy 2.2.5  SciPy 1.15.3
BLAS:      name: mkl-sdl; version: '2023.1';
Threads:   OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 (exported by run.sh); particle loop is serial.
Timer:     time.perf_counter() (monotonic wall clock), per FIX 10 in the script

Parts / code actually run:
  gpubench2    code/gpubench2__run07_gpubench2.py  md5 f1c96e8999b67a2ed45fafb0f3e58434  (source /DATA/Aurindum/GPUSwarm/GPUBench2.py md5 af569a633a2e812dd84bd3693c244ce2)

Changes vs source (only these lines differ):
--- gpubench2
--- GPUBench2.py
+++ gpubench2__run07_gpubench2.py
@@ -94 +94 @@
-ROOT_OUT = args.out
+ROOT_OUT = "MinTime/Runs/run07_gpubench2/output/gpubench2"


Execution: conda env gpuwork; taskset 16-23; extra env none; arguments {'gpubench2': '--targets 0.05,0.045,0.04 --runs 20'}

Launch: cd /DATA/Aurindum/Swarming && nohup setsid bash /DATA/Aurindum/Swarming/MinTime/Runs/run07_gpubench2/run.sh > /DATA/Aurindum/Swarming/MinTime/Runs/run07_gpubench2/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
Part gpubench2 START: 2026-10-08 12:28:22 IST  load: 1.35 1.24 1.19
