RUN run02_naive_jitter0  (pre-convention run; folder organised 2026-10-08 after its library part ended)
Current code with WARM_START_JITTER = 0.0 (naive warm block: 20 identical warm particles + 30 random, no elite jitter).
Library part (numpy, solve, sm, iterative), thresholds 0.05/0.045/0.04, 20 runs, BLAS pinned to 1 thread, perf_counter timer.
Code: code/ScratchBench_library_naive.py (md5 b3efb87d84bdbf485b843912109ba2b7); code/ScratchBench2_purepython_naive.py
was NOT run here (its pure_python part became run04_naive_purepython_5runs). Output: output/library/, logs in logs/, launchers in launchers/.
Full original timeline, environment, md5 sums and notes: README_original_rerun.txt (written as the run went; paths in it refer to the
old location MinTime/ScratchBenchRerun_20261006/, which no longer exists).
NOTE 2026-10-08: moved from MinTime/Runs/run02_naive_jitter0 to runs/scratchbench/run02_naive_jitter0 (repo restructure to the GPUSwarm layout). Paths printed above in this README and baked into code/*.py (ROOT_OUT) refer to the old location; nothing else changed.
