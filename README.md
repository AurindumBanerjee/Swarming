# Swarming

PSO-based decoupling-capacitor placement on a 21-port benchmark PDN, with several matrix-inversion methods
(`numpy`, `solve`, `sm` = Sherman-Morrison, `iterative` = approximate series, `pure_python` = hand-written
Gauss-Jordan) benchmarked against each other. The code runs on the EEDept server (`/DATA/Aurindum/Swarming`);
this repository is the record of the code, the run manifests and the documentation. Bulk outputs are not tracked.

## Layout

| path | what | tracked |
|---|---|---|
| `ScratchBench.py` | **latest reference** library-methods benchmark (numpy, solve, sm, iterative); FIX 11-13 (hybrid warm start, port resolver, stage reset) | yes |
| `ScratchBench2.py` | same, for the `pure_python` baseline | yes |
| `ScratchBenchOld.py`, `ScratchTest.py`, `PythonBench*.py` | earlier benchmark scripts (the `PythonBench*` ones are the older `ScratchBench_sim` simulations) | yes |
| `MinImp.py`, `MinImp3.py`, `MinImp*/` | earlier PSO experiments and their logs | scripts and old logs |
| `swarming.py`, `Refer.py`, `nocpu.py` | `swarming.py` is the CuPy (GPU) version; `Refer.py` and `nocpu.py` are reference variants (see the file headers) | yes |
| `plot.py`, `Plotting.py`, `plot_ieee_figures.py`, `plot_matrix_evaluation_speedup.py` | analysis and figure scripts; the last two build IEEE-style speedup figures from `summarize_baseline.py` output | yes |
| `Testing1.py`, `Testing2.py`, `TestAdjDet.py`, `TestGauss.py`, `TestIterative.py` | CPU PSO parameter sweeps (particles x iterations, c1/c2) and PSO runs with adjoint-determinant, Gaussian and iterative inversion | yes |
| `summarize_baseline.py` | summarises a results tree (last RUN_START block of each log only) | yes |
| `Data/` | benchmark PDN: `y2.mat`, `decaps.mat`, `freq2.mat` (21 ports, 1391 frequencies) | yes |
| `Dataset/` | MPHY / VDDQ DDR3 dataset (server only, large `.mat`) | on the server repo; **not** in the local checkout |
| `Outputs/` | older outputs and logs | old files tracked, new ones ignored |
| `MinTime/` | legacy timing-benchmark result folders (`ScratchBench2..5`, `ScratchTest*`, ...), ignored. `MinTime/Runs/` held the runs until 2026-10-08 (now `runs/`) | legacy |
| **`runs/`** | **one isolated folder per run: `runs/<experiment>/<run_id>/`** (experiment `scratchbench`); `runs/INDEX.md` lists them; `runs/_ops/` keeps the launch/formalise scripts used so far | code, manifests, READMEs, summaries, plots tracked; `output/` and `logs/` ignored |
| **`reference/`** | **starting point for new work, same convention as the GPUSwarm repo**: `template_experiment.py`, `new_run.py`, `summarize_run.sh`, `output_style/` (samples of every output type) | yes |

The scripts take their data from `/DATA/Aurindum/Swarming/Data` (absolute `BASE_DIR`) and write to a relative
`ROOT_OUT`, so run them with the repository root as the working directory.

## Runs convention (`runs/`, shared with the GPUSwarm repo)

Every run has its own folder with its own copy of the code, so an old run's files are never rewritten:

```
runs/<experiment>/<run_id>/
    code/<script>__<run_id>.py   the script exactly as run (source copy + only the --set lines listed in its README)
    output/                      ROOT_OUT of the script: raw results (ignored by git; copy it around separately)
    logs/                        console output (ignored)
    summary/  plots/             tables and figures (tracked)
    run.sh  manifest.json  README.txt   launcher (BLAS pinned to 1 thread), md5 of source and as-run code, diff, environment, timeline
runs/INDEX.md                    one row per run
reference/new_run.py             creates a run folder:  python reference/new_run.py --experiment scratchbench --id runNN_name --desc "..."                                      --source ScratchBench.py --set 'ROOT_OUT = "runs/scratchbench/runNN_name/output"' --set 'NUM_RUNS = 5' [--taskset 0-7] [--conda-env work]
reference/summarize_run.sh       runs summarize_baseline.py (+ the speedup plots when a pure_python partner is present) into <run>/summary and <run>/plots
```

Never edit a finished or running run's files; make a new run with `reference/new_run.py` instead. The ScratchBench scripts write to
a literal `ROOT_OUT`, so a run sets it to its own `output/` with `--set`; new code should read `RUN_DIR` instead (`reference/template_experiment.py`).

| run | what |
|---|---|
| `run00_original_sep2026` | the original Sep-2026 baseline: FIX 4 warm start (all 50 particles identical in the carried genes), old port resolver. Code, outputs, summary and plots copied unchanged. |
| `run01_hybrid_jitter0.02_partial` | FIX 11-13 code (40 % warm block, elite + jitter 0.02), library part only, stopped part-way. |
| `run02_naive_jitter0` | same code with `WARM_START_JITTER = 0.0` (naive warm block), library part, finished. |
| `run03_original_code_library` | original code (FIX 4), library part, 20 runs; **stopped** on 2026-10-08 (0.05 complete for all four methods; at 0.045 numpy and solve complete, sm 16/20, iterative not started). |
| `run03_original_code_repro`, `run04_naive_purepython_5runs`, `run05_original_purepython_5runs`, `run06_warm100_jitter0_library` | planned, **never executed** (cancelled 2026-10-08). Their code copies still point at `MinTime/Runs/...` output paths. |
| `gpubench2/run07_gpubench2` | `GPUBench2.py` on the GPU (PyTorch), finished; exact methods identical, woodbury_batch 7-33x and fast 19-38x over the GPU numpy method. |
| `run08_warm100_5runs` | 100 % warm start (`WARM_START_FRACTION = 1.0`, jitter 0), library methods, 5 runs per threshold; finished. Identical to run00 at 0.05/0.045; 0.04 not reached (0/5): run00's 0.04 successes used duplicate-port placements. |

Each run keeps its results next to its code: `summary/` (`runs.csv`, `summary.csv`, `summary.txt`, from `summarize_baseline.py`; only the
last `RUN_START` block of each log is used) and `plots/` (speedup figures). The speedup S is relative to the `pure_python` run, so plots exist
only where a pure_python partner is present. Several roots can be merged (a run's library and another run's pure_python) with
`reference/summarize_run.sh <out_dir> <root> [<root> ...]`.

## Measurement notes

Timer: `time.perf_counter()`. Exact methods (`numpy`, `solve`, `sm`, `pure_python`) must produce identical placements
for identical seeds; `iterative` is an approximation (falls back to a full inverse where its series diverges).
Thread settings of runs created with `new_run.py` are recorded in each run's README (BLAS pinned to 1); the thread
setting of run00 was not recorded.

## Local checkout

The laptop copy is a checkout of the same history. It leaves out the tracked bulk data (`Dataset/*.mat`, the old
`Outputs/`, `MinImp*/` logs) via sparse checkout; those live on the server. Outputs of the Runs are copied to the
laptop separately (they are not tracked).
