RUN run01_hybrid_jitter0.02_partial  (pre-convention run; folder organised 2026-10-08 after run02 ended)
Current code (FIX 11-13): hybrid warm start = 40% warm block (particle 0 exact elite, others jittered by 0.02), 60% random.
Library part only; thresholds 0.05/0.045/0.04, 20 runs. Stopped (killed) partway on 2026-10-06 ~21:00 IST after numpy and solve
at 0.05 (20/20 each) and sm at 0.05 (18/20 results). Code: code/ScratchBench_library.py (md5 1b74422db9cfc8975276c91ad9b462bf),
code/ScratchBench2_purepython.py was never run. Output: output/library/, console: logs/library_console.log.
Original timeline and environment: README_original_rerun.txt in ../run02_naive_jitter0/.
Results: at 0.05 this code needed 11.55 caps on average (old code 5.25) -- see the warm-start comparison in GPUSwarm/Comparisons.
NOTE 2026-10-08: moved from MinTime/Runs/run01_hybrid_jitter0.02_partial to runs/scratchbench/run01_hybrid_jitter0.02_partial (repo restructure to the GPUSwarm layout). Paths printed above in this README and baked into code/*.py (ROOT_OUT) refer to the old location; nothing else changed.
