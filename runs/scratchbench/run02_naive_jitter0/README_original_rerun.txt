ScratchBench independent re-run -- ScratchBenchRerun_20261006
Existing results (MinTime/ScratchBenchBaseline) are NOT touched; this run writes only below this directory.

Date written:   2026-10-06 10:04:10 IST
Machine:        eeiitj123.iitj.ac.in  (Linux 3.10.0-1160.el7.x86_64 x86_64)
CPU:            Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
CPU count:      64 logical CPUs; 2 sockets x 16 cores x 2 threads
Python:         3.10.18   (/DATA/Aurindum/conda/envs/work/bin/python)
NumPy:          2.2.5
SciPy:          1.15.3
BLAS (NumPy):    name: mkl-sdl; version: '2023.1';

Thread settings (exported by run_rerun.sh for BOTH parts; the scripts set no limits themselves):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  The PSO particle loop in both scripts is serial: each run uses one thread.
  Parts run strictly one after the other; no other jobs were started on the machine.
  Load at README creation: 0.50 0.15 0.09

Scripts actually executed (copies of the latest laptop versions of ScratchBench.py / ScratchBench2.py;
only METHODS, TARGETS and ROOT_OUT set; run with cwd=/DATA/Aurindum/Swarming):
  1b74422db9cfc8975276c91ad9b462bf  ScratchBench_library.py
  8cf4ff43847c48d390255053317a01b5  ScratchBench2_purepython.py
  2657775a96547a6f254e6d94843bfc4d  run_rerun.sh
Source scripts on the laptop (Swarming/ScratchBench.py, ScratchBench2.py) md5:
  dafba8e36e3b7a2e20f7632370135186  ScratchBench.py
  1d8d765e94811473b5173f7b37458750  ScratchBench2.py
Differences vs the sources: ROOT_OUT and TARGETS=[0.05, 0.045, 0.04] only (METHODS already equal).
NOTE: the copies of these two scripts that sit on the server in /DATA/Aurindum/Swarming are an OLDER version
      (md5 9cd1e2b1... / 7fc8245a...; no FIX 11-13). They were left untouched and were not used.

Part 1: ScratchBench_library.py   METHODS=[numpy, solve, sm, iterative]  TARGETS=[0.05,0.045,0.04]  NUM_RUNS=20
        ROOT_OUT=MinTime/ScratchBenchRerun_20261006/library       console: library_console.log
Part 2: ScratchBench2_purepython.py  METHODS=[pure_python]  TARGETS=[0.05,0.045,0.04]  NUM_RUNS=20
        PURE_PYTHON_NUM_RUNS=None PURE_PYTHON_TARGETS=None (full sweep)
        ROOT_OUT=MinTime/ScratchBenchRerun_20261006/pure_python   console: pure_python_console.log
Fixed: BASE_SEED=12345 P=50 T=15 MAX_CAPS=20 c1=c2=1.5 W 0.9->0.4 WARM_START_FRACTION=0.4 WARM_START_JITTER=0.02

---- timeline ----
Part 1 (library: numpy solve sm iterative) START: 2026-10-06 10:04:18 IST  load: 0.46 0.15 0.09
Note 2026-10-06 11:57 IST: run_rerun.sh was replaced by run_chain.sh (Part 1 python pid 14294 untouched, never interrupted). The chain inserts the GPUSwarm/Comparisons warm-start runs between Part 1 and Part 2 so Part 2 timings are not disturbed.
Note 2026-10-06 20:32 IST: Part 1 python pid 14294 PAUSED (SIGSTOP) at user request to run the warm-start comparison first; run_chain.sh stopped. Run in flight at pause time (see last run log) has inflated wall time. Resume with kill -CONT 14294.

---- 2026-10-06 ~21:00 IST: switch to NAIVE warm start (user request) ----
The jitter-0.02 Part 1 run was paused at 20:32 and killed (SIGKILL) at the switch; it had finished numpy, solve and sm at 0.05 only partly (sm at 0.05 incomplete).
Its partial outputs were moved UNCHANGED to previous_jitter0.02/ (library/, library_console.log, and copies of the exact scripts used).
New run: WARM_START_JITTER = 0.0 (nothing else changed; elite particle + 40% identical warm block + 60% random = the "naive" arm of GPUSwarm/Comparisons).
    b3efb87d84bdbf485b843912109ba2b7  ScratchBench_library_naive.py
  2b6aa56ada456d60dad7e935bd20f0c0  ScratchBench2_purepython_naive.py
  cea008049ed3b4ff2e93d0cc456b431f  run_chain2.sh
Same ROOT_OUT as before (library/ and pure_python/), Part 1 then Part 2, BLAS pinned to 1 thread, run by run_chain2.sh.
[naive] Part 1 (library, jitter 0.0) START: 2026-10-06 21:45:23 IST  load: 0.09 0.09 1.24
Run plan 2026-10-07 12:49 IST: run02's chain stopped after its library part; run02 pure_python replaced by run04 (5 runs/threshold). Sequencer: Runs/sequence_03_05.sh
Superseded 2026-10-07 12:51: sequence_03_05.sh cancelled (replaced by sequence_03_05_gated.sh: holds run03 until GO_RUN3 exists).
[naive/run02] Part 1 (library) END: 2026-10-08 07:22:40 IST  (launcher replaced by Runs/sequence_03_05_gated.sh; exit code not captured -- see library_console.log; run02 pure_python is now run04)
