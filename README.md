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
| `MinTime/` | timing-benchmark results. Old result folders (`ScratchBench2..5`, `ScratchTest*`, ...) are legacy and ignored | only `MinTime/Runs/` is new |
| **`MinTime/Runs/`** | **the home of every ScratchBench run, see below** | code, manifests, READMEs tracked; `output/` and `logs/` ignored |

The scripts take their data from `/DATA/Aurindum/Swarming/Data` (absolute `BASE_DIR`) and write to a relative
`ROOT_OUT`, so run them with the repository root as the working directory.

## Runs convention (`MinTime/Runs/`)

Every run has its own folder with its own copy of the code, so an old run's files are never rewritten:

```
MinTime/Runs/<run_id>/
    code/<part>__<run_id>.py   the script exactly as run (source copy + only the settings listed in its README)
    output/<part>/             ROOT_OUT of that part (ignored by git; copy it around separately)
    logs/                      console output (ignored)
    run.sh                     launcher: BLAS pinned to 1 thread, parts strictly one after the other
    manifest.json, README.txt  md5 of source and as-run code, exact diff, environment, start/end times
MinTime/Runs/INDEX.md          one row per run
MinTime/Runs/new_run.py        creates a new run folder:  python new_run.py create --id ... --desc ... --part name=src.py --set KEY=VALUE
```

Never edit a finished or running run's files; make a new run with `new_run.py` instead.

Each run folder also keeps its results next to its code: `summary/` (`runs.csv`, `summary.csv`, `summary.txt`, from
`summarize_baseline.py`; only the last `RUN_START` block of each log is used) and `plots/` (speedup figures). The speedup
S is relative to the `pure_python` run, so plots exist only where a pure_python partner is present (run00; for the library-only
runs a `NO_PLOTS.txt` explains why and the pair is plotted once its pure_python run finishes). Make or refresh them with
`MinTime/Runs/summarize_run.sh <out_dir> <root> [<root> ...]` (several roots are merged through a symlink farm, so
e.g. run02's library and run04's pure_python can be summarised and plotted together). Only `output/` and `logs/` are git-ignored.

| run | what |
|---|---|
| `run00_original_sep2026` | the original Sep-2026 baseline: FIX 4 warm start (all 50 particles identical in the carried genes), old port resolver. Code and outputs copied unchanged. |
| `run01_hybrid_jitter0.02_partial` | FIX 11-13 code (40 % warm block, elite + jitter 0.02), library part only, stopped part-way. |
| `run02_naive_jitter0` | same code with `WARM_START_JITTER = 0.0` (naive warm block), library part. |
| `run03_original_code_library` | original code (FIX 4), library part, thresholds 0.05 / 0.045 / 0.04, 20 runs. |
| `run04_naive_purepython_5runs` | run02 code, `pure_python`, 5 runs per threshold. |
| `run05_original_purepython_5runs` | original code, `pure_python`, 5 runs per threshold. |

## Measurement notes

Timer: `time.perf_counter()`. Exact methods (`numpy`, `solve`, `sm`, `pure_python`) must produce identical placements
for identical seeds; `iterative` is an approximation (falls back to a full inverse where its series diverges).
Thread settings of runs created with `new_run.py` are recorded in each run's README (BLAS pinned to 1); the thread
setting of run00 was not recorded.

## Local checkout

The laptop copy is a checkout of the same history. It leaves out the tracked bulk data (`Dataset/*.mat`, the old
`Outputs/`, `MinImp*/` logs) via sparse checkout; those live on the server. Outputs of the Runs are copied to the
laptop separately (they are not tracked).
