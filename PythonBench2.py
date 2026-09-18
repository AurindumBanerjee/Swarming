# nohup python ScratchBench_sim.py > scratchbench_sim.log 2>&1 &
# ps -u $USER | grep ScratchBench_sim
# pkill -9 -f ScratchBench_sim.py
#
# ============================================================
# SIMULATION-ONLY SCRIPT -- run this, then run ScratchBench_plot.py
# ============================================================
# This file ONLY runs PSO placements and logs results. All aggregation/
# CSV/figure generation has been split out into ScratchBench_plot.py --
# run that separately once you have the data you want plotted.
#
# INTENDED WORKFLOW (methods run one at a time, or in parallel):
#   1. Edit the METHODS list below to enable exactly the method(s) you
#      want this invocation to run (comment/uncomment lines). Right now
#      only "pure_python" is active -- the other four are commented out.
#   2. nohup python ScratchBench_sim.py > pure_python.log 2>&1 &
#   3. When you're ready for another method, edit METHODS again (e.g.
#      enable only "numpy") and launch another nohup run. Every run
#      writes into the SAME ROOT_OUT ("MinTime/ScratchBench5"), each
#      method's PSO output going to its own `{method}_{threshold}/`
#      subfolder, and every run_pso() call APPENDS one line to the
#      shared results_long.jsonl (opened in "a" mode, one small write()
#      call per record -- safe to append to concurrently from several
#      simultaneously-running nohup processes on Linux).
#   4. Once all 5 methods you care about have been run (check
#      ROOT_OUT/results_long.jsonl), run:
#         python ScratchBench_plot.py
#      to build results_long.csv / summary_table.csv / speedup_matrix.csv
#      / figs/*.png from whatever is in results_long.jsonl at that point.
#
# ============================================================
# WHAT CHANGED VS THE ORIGINAL ScratchBench.py (carried over from
# earlier revisions, still true here)
# ============================================================
# 1. NEW "pure_python" BASELINE METHOD
#    inv_pure_python() inverts the assembled admittance matrix with a
#    hand-written Gauss-Jordan elimination using plain Python lists +
#    the built-in `complex` type only -- zero NumPy/SciPy anywhere on
#    its timed path. It is the new reference baseline: every other
#    method's speedup is reported against it in ScratchBench_plot.py,
#    so the numbers show how much of "LAPACK's speedup" is really
#    coming from compiled linear algebra vs. just avoiding a slow
#    Python-level loop. Correctness is checked against numpy.linalg.inv
#    at startup (_selftest_pure_python_inverse), whenever pure_python is
#    one of the active METHODS.
#
# 2. NO WOODBURY IDENTITY ANYWHERE
#    inv_sm() still only performs the rank-1 Sherman-Morrison update
#    (one capacitor at a time) exactly as before. There is no general
#    low-rank/Woodbury batch-update path in this file.
#
# 3. PAIRED RNG SEEDING ACROSS METHODS
#    np.random.seed(BASE_SEED + run) is set immediately before each
#    (threshold, method, run) call to run_pso(), so run index `run`
#    draws the *same* PSO initialization/velocities regardless of which
#    method/invocation is running it -- this is what keeps separate
#    single-method runs of this script directly, fairly comparable.
#
# 4. STRUCTURED, CRASH-SAFE RESULT LOGGING
#    Every run_pso() call appends one JSON record to
#    ROOT_OUT/results_long.jsonl the moment it finishes -- safe for long
#    unattended nohup jobs, and safe across several methods' processes
#    running at once (see workflow note above).
#
# 5. SYNTHETIC-DATA FALLBACK + QUICK-TEST AUTO-SHRINK
#    If the real y2.mat / decaps.mat files aren't found at DATA_DIR,
#    the script fabricates a small synthetic PDN + capacitor library
#    and shrinks NUM_RUNS/N_PARTICLES/etc. so the file is runnable
#    standalone. Point BASE_DIR at the real cluster data path to run at
#    paper scale.
#
# 6. HYBRID METHOD REMOVED entirely (function, cache, dispatch branch).
#
# Everything else (PSO logic, port-collision resolution, per-run
# plots/logs, folder layout) is intentionally left untouched.
# ============================================================

import os
import sys
import time
import json
import math
import logging
import shutil

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# GLOBAL CONFIG
# ============================================================

ROOT_OUT = "MinTime/ScratchBench5"
os.makedirs(ROOT_OUT, exist_ok=True)

TARGET_PORT = 0
MAX_CAPS = 20

N_PARTICLES = 50
N_ITERATIONS = 15

W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5

NUM_RUNS = 20
TARGETS = [
    # 0.05, 
    0.045, 
    # 0.04, 
    # 0.03,
]

# ---- Active methods for THIS invocation. Comment/uncomment to control
# which method(s) this run executes -- see the workflow note at the top
# of this file. Only "pure_python" is active right now. ----
METHODS = [
    # "numpy",
    # "solve",
    # "sm",
    # "iterative",
    "pure_python",
]

BASE_SEED = 12345  # seed for run `r` is BASE_SEED + r, reused across methods/thresholds


# ---- data locations ----
BASE_DIR = "/DATA/Aurindum/Swarming"
DATA_DIR = os.path.join(BASE_DIR, "Data")
Y_MAT_PATH = os.path.join(DATA_DIR, "y2.mat")
D_MAT_PATH = os.path.join(DATA_DIR, "decaps.mat")

REAL_DATA_AVAILABLE = os.path.exists(Y_MAT_PATH) and os.path.exists(D_MAT_PATH)

if not REAL_DATA_AVAILABLE:
    print(f"[WARN] Real PDN data not found under {DATA_DIR}")
    print("[WARN] Falling back to a small synthetic PDN and a shrunk run "
          "schedule so this script is runnable standalone. Point BASE_DIR "
          "at the real cluster data path to run at paper scale.")
    NUM_RUNS = 3
    TARGETS = [0.05, 0.04]
    MAX_CAPS = 5
    N_PARTICLES = 8
    N_ITERATIONS = 4

# ============================================================
# LOGGING
# ============================================================

def setup_logger(path):
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(path)
    formatter = logging.Formatter('%(asctime)s INFO: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def plot_run_curve(run_caps, run_minz, threshold, method, run_id, out_folder):

    if not run_caps:
        return

    xs = np.array(run_caps, dtype=int)
    ys = np.array(run_minz, dtype=float)
    best_so_far = np.minimum.accumulate(ys)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker='o', linewidth=1.5, label='minZ per n_caps')
    plt.plot(xs, best_so_far, marker='s', linestyle='--', linewidth=1.5, label='best-so-far envelope')
    plt.axhline(y=threshold, linestyle='--', linewidth=1.0, label=f'target={threshold}')
    plt.xlabel("Decaps")
    plt.ylabel("Peak |Z11| (Ohm)")
    plt.title(f"{method} T={threshold} Run {run_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "run_plot.png"), dpi=200)
    plt.close()


def plot_convergence(history_dict, threshold, out_folder):

    if not history_dict:
        return

    plt.figure(figsize=(9, 5))
    for n_caps, hist in history_dict.items():
        if hist:
            plt.plot(hist, label=f"n={n_caps}")

    plt.axhline(y=threshold, linestyle='--', linewidth=1.0)
    plt.xlabel("Iteration")
    plt.ylabel("Peak |Z11| (Ohm)")
    plt.title("Global Convergence")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "global_convergence.png"), dpi=200)
    plt.close()


# ============================================================
# SYNTHETIC DATA FALLBACK (only used when REAL_DATA_AVAILABLE is False)
# ============================================================

def _generate_synthetic_pdn(n_nodes, n_freq, n_cap_models, seed=0):
    """
    Parametric RLC-mesh PDN + a synthetic decoupling-capacitor admittance
    library, shaped exactly like the real y2.mat / decaps.mat arrays
    (y: (F, n, n) complex128 after transpose; d: (C, F) complex128 lookup
    table), so nothing downstream needs to know the data is synthetic.
    """
    rng = np.random.default_rng(seed)
    freqs = np.logspace(4, 8, n_freq)
    w = 2.0 * np.pi * freqs

    R_shunt = rng.uniform(5.0, 20.0, size=n_nodes)
    L_shunt = rng.uniform(1e-9, 8e-9, size=n_nodes)
    C_shunt = rng.uniform(1e-10, 5e-10, size=n_nodes)
    R_link = rng.uniform(0.02, 0.08, size=n_nodes)
    L_link = rng.uniform(0.5e-9, 3e-9, size=n_nodes)

    y = np.zeros((n_freq, n_nodes, n_nodes), dtype=np.complex128)
    Y_shunt = (1.0 / R_shunt)[None, :] + 1.0 / (1j * w[:, None] * L_shunt[None, :]) \
        + 1j * w[:, None] * C_shunt[None, :]
    Z_link = R_link[None, :] + 1j * w[:, None] * L_link[None, :]
    Y_link = 1.0 / Z_link

    for k in range(n_nodes):
        j = (k + 1) % n_nodes
        y[:, k, k] += Y_shunt[:, k] + Y_link[:, k]
        y[:, j, j] += Y_link[:, k]
        y[:, k, j] -= Y_link[:, k]
        y[:, j, k] -= Y_link[:, k]

    y += (1e-6 * np.eye(n_nodes, dtype=np.complex128))[None, :, :]

    d = np.zeros((n_cap_models, n_freq), dtype=np.complex128)
    for m in range(n_cap_models):
        case = rng.integers(0, 5)
        cap_val = 10.0 ** rng.uniform(-9 + case * 0.4, -6 + case * 0.5)
        esl = 10.0 ** rng.uniform(-10.3 + case * 0.15, -9.6 + case * 0.15)
        esr = 10.0 ** rng.uniform(-2.3, -0.7)
        z = esr + 1j * w * esl + 1.0 / (1j * w * cap_val)
        d[m, :] = 1.0 / z

    return y, d


# ============================================================
# LOAD DATA
# ============================================================

print("Loading PDN data...")

if REAL_DATA_AVAILABLE:
    y = sio.loadmat(Y_MAT_PATH)["y"]
    d = sio.loadmat(D_MAT_PATH)["decaps"]
    y = np.transpose(y, (2, 0, 1))
else:
    y, d = _generate_synthetic_pdn(n_nodes=8, n_freq=40, n_cap_models=150, seed=0)

N_FREQS, N_NODES, _ = y.shape
N_CAP_MODELS = d.shape[0]

print(f"N_FREQS={N_FREQS}  N_NODES={N_NODES}  N_CAP_MODELS={N_CAP_MODELS}  "
      f"real_data={REAL_DATA_AVAILABLE}")

# ============================================================
# PRECOMPUTE BASE INVERSE (only needed by the sm / iterative fast paths --
# skipped entirely when neither is in the active METHODS list, e.g. the
# pure_python-only run this file is currently configured for)
# ============================================================

if "sm" in METHODS or "iterative" in METHODS:
    y_inv_base = np.zeros_like(y, dtype=np.complex128)
    for f in range(N_FREQS):
        y_inv_base[f] = np.linalg.inv(y[f])
else:
    y_inv_base = None

# ---- pure-Python precompute: plain nested lists, built once with numpy's
#      own (C-speed) .tolist() marshalling -- no numpy CALLS happen on the
#      timed inv_pure_python() path below, only list indexing/copying. ----
if "pure_python" in METHODS:
    Y_LISTS = [y[f].tolist() for f in range(N_FREQS)]   # F x [n x [n x complex]]
    D_LISTS = d.tolist()                                  # C x [F x complex]
else:
    Y_LISTS = None
    D_LISTS = None

# ============================================================
# PORT RESOLUTION
# ============================================================

def resolve_adjacent_ports(models, ports):

    used = set()

    for i in range(len(ports)):

        if ports[i] not in used:
            used.add(ports[i])
            continue

        for offset in range(1, N_NODES):

            for cand in [ports[i] + offset, ports[i] - offset]:

                if 0 <= cand < N_NODES and cand not in used:
                    ports[i] = cand
                    used.add(cand)
                    break

            if ports[i] in used:
                break

    return models, ports

# ============================================================
# INVERSION METHODS
# ============================================================

def inv_numpy(A):
    return np.linalg.inv(A)[TARGET_PORT, TARGET_PORT]

def inv_solve(A):
    e = np.zeros(N_NODES, dtype=complex)
    e[TARGET_PORT] = 1
    return np.linalg.solve(A, e)[TARGET_PORT]

def inv_sm(config, f):
    """
    Rank-1 Sherman-Morrison update, one capacitor at a time. This is the
    ONLY low-rank update implemented anywhere in this file -- there is no
    general Woodbury (simultaneous multi-rank) update path here or
    elsewhere in the script.
    """
    Z = y_inv_base[f].copy()

    for cap, port in config:

        val = d[cap, f]

        u = np.zeros(N_NODES, dtype=np.complex128)
        v = np.zeros(N_NODES, dtype=np.complex128)

        u[port] = 1
        v[port] = val

        denom = 1 + v @ Z @ u

        if abs(denom) < 1e-12:
            raise np.linalg.LinAlgError

        Z -= np.outer(Z @ u, v @ Z) / denom

    return Z[TARGET_PORT, TARGET_PORT]

def inv_iterative(A, f):

    B = y_inv_base[f]
    E = A @ B - np.eye(N_NODES)

    if np.linalg.norm(E, 1) >= 1:
        return np.linalg.inv(A)[TARGET_PORT, TARGET_PORT]

    corr = np.eye(N_NODES) - E + E @ E
    Z = B @ corr

    return Z[TARGET_PORT, TARGET_PORT]


# ---- pure-Python / math-lib-only baseline ----
def _gauss_jordan_inverse(A):
    """
    Full matrix inverse via Gauss-Jordan elimination with partial
    pivoting. Plain Python lists + the built-in `complex` type only --
    no numpy, no scipy, anywhere in this function. O(n^3).
    """
    n = len(A)
    M = [row[:] + [1.0 + 0.0j if i == r else 0.0 + 0.0j for i in range(n)]
         for r, row in enumerate(A)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-300:
            raise ZeroDivisionError("Singular matrix in pure-Python inversion")
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        inv_pivot = 1.0 / pivot
        row_col = M[col]
        for k in range(2 * n):
            row_col[k] *= inv_pivot

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0:
                continue
            row_r = M[r]
            for k in range(2 * n):
                row_r[k] -= factor * row_col[k]

    return [row[n:] for row in M]


def inv_pure_python(config, f):
    """
    Same math as inv_numpy(), computed with zero NumPy/LAPACK involvement:
    plain Python lists, `complex` arithmetic, Gauss-Jordan elimination.
    This is the baseline method (BASELINE_METHOD in ScratchBench_plot.py)
    -- every other method's speedup is reported relative to this one.
    """
    Yeq = [row[:] for row in Y_LISTS[f]]        # pure-Python copy
    for cap, port in config:
        Yeq[port][port] += D_LISTS[cap][f]       # pure-Python complex add
    inv = _gauss_jordan_inverse(Yeq)
    return inv[TARGET_PORT][TARGET_PORT]


def _selftest_pure_python_inverse():
    """Sanity check: inv_pure_python must agree with inv_numpy to float64
    precision on a handful of random candidates, at every scale where it
    will actually be used. Run once at startup; raises on mismatch."""
    if "pure_python" not in METHODS:
        return
    rng = np.random.default_rng(999)
    max_err = 0.0
    for _ in range(5):
        f = int(rng.integers(0, N_FREQS))
        n_caps = int(rng.integers(1, min(4, N_CAP_MODELS, N_NODES) + 1))
        models = rng.integers(0, N_CAP_MODELS, size=n_caps)
        ports = rng.integers(0, N_NODES, size=n_caps)
        models, ports = resolve_adjacent_ports(list(models), list(ports))
        config = list(zip(models, ports))

        A = y[f].copy()
        for cap, port in config:
            A[port, port] += d[cap, f]
        z_numpy = inv_numpy(A)
        z_pp = inv_pure_python(config, f)
        max_err = max(max_err, abs(z_numpy - z_pp))

    assert max_err < 1e-8, f"pure-Python inverse diverged from numpy: max err {max_err}"
    print(f"[selftest] inv_pure_python matches inv_numpy (max err {max_err:.2e}). OK")

_selftest_pure_python_inverse()

# ============================================================
# FITNESS
# ============================================================

def evaluate_config(config, method):

    peak = 0

    for f in range(N_FREQS):

        try:
            if method == "pure_python":
                val = inv_pure_python(config, f)
            else:
                A = y[f].copy()
                for cap, port in config:
                    A[port, port] += d[cap, f]

                if method == "numpy":
                    val = inv_numpy(A)
                elif method == "solve":
                    val = inv_solve(A)
                elif method == "sm":
                    val = inv_sm(config, f)
                elif method == "iterative":
                    val = inv_iterative(A, f)
                else:
                    raise ValueError(f"unknown method {method!r}")
        except Exception:
            return 1e200

        peak = max(peak, abs(val))

    return peak

# ============================================================
# STRUCTURED RESULT LOGGING (replaces regex log-scraping)
# ============================================================

RESULTS_JSONL_PATH = os.path.join(ROOT_OUT, "results_long.jsonl")

def append_result_record(record):
    """Append one JSON line and flush immediately -- crash-safe for long
    unattended (nohup) jobs; the file is re-read once at the very end."""
    with open(RESULTS_JSONL_PATH, "a") as jf:
        jf.write(json.dumps(record) + "\n")
        jf.flush()

# ============================================================
# PSO
# ============================================================

def run_pso(method, threshold, out_folder, run_id):

    logger = setup_logger(os.path.join(out_folder, f"run_{run_id}.log"))

    run_caps = []
    run_minZ = []
    histories = {}

    start_global = time.time()
    time_to_target = None
    caps_at_target = None
    n_evals = 0

    logger.info(
        "RUN_START | method=%s | threshold=%.6f | run=%d | particles=%d | iterations=%d",
        method, threshold, run_id, N_PARTICLES, N_ITERATIONS
    )

    for n_caps in range(1, MAX_CAPS + 1):

        DIM = 2 * n_caps

        particles = np.random.rand(N_PARTICLES, DIM)
        velocities = np.zeros_like(particles)

        pbest = particles.copy()
        pbest_val = np.full(N_PARTICLES, np.inf)

        gbest = np.inf
        gbest_particle = None
        history = []

        stop_flag = False
        it = 0

        for it in range(N_ITERATIONS):

            for i in range(N_PARTICLES):

                models = np.floor(particles[i, :n_caps] * (N_CAP_MODELS - 1)).astype(int)
                ports = np.floor(particles[i, n_caps:] * (N_NODES - 1)).astype(int)

                models, ports = resolve_adjacent_ports(models, ports)

                config = list(zip(models, ports))

                cost = evaluate_config(config, method)
                n_evals += 1

                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]

                if cost < gbest:
                    gbest = cost
                    gbest_particle = particles[i].copy()

                if gbest <= threshold:
                    stop_flag = True
                    break

            history.append(gbest)

            if stop_flag:
                if time_to_target is None:
                    time_to_target = time.time() - start_global
                    caps_at_target = n_caps
                break

            w = W_MAX - (W_MAX - W_MIN) * (it / N_ITERATIONS)

            velocities = w * velocities \
                + C1 * np.random.rand(*particles.shape) * (pbest - particles) \
                + C2 * np.random.rand(*particles.shape) * (gbest_particle - particles)

            particles += velocities
            particles = np.clip(particles, 0, 1)

        histories[n_caps] = history

        if gbest_particle is None:
            logger.info("n_caps=%d | iter=%d | minZ=inf | placement={}", n_caps, it + 1)
            continue

        best_models = np.floor(gbest_particle[:n_caps] * (N_CAP_MODELS - 1)).astype(int)
        best_ports = np.floor(gbest_particle[n_caps:] * (N_NODES - 1)).astype(int)
        best_models, best_ports = resolve_adjacent_ports(best_models, best_ports)
        placement = {int(port): int(cap) for cap, port in zip(best_models, best_ports)}

        logger.info(
            "n_caps=%d | iter=%d | minZ=%.6f | placement=%s",
            n_caps, it + 1, gbest, placement
        )

        run_caps.append(n_caps)
        run_minZ.append(gbest)

        if gbest <= threshold:
            break

    total_wall_time = time.time() - start_global
    best_minz = min(run_minZ) if run_minZ else float("inf")
    success = time_to_target is not None

    logger.info(
        "RESULT | time_to_target=%.4f | caps=%s | best_minZ=%.6f | total_wall_time=%.4f",
        time_to_target if time_to_target else -1,
        caps_at_target if caps_at_target else -1,
        best_minz,
        total_wall_time,
    )

    plot_run_curve(run_caps, run_minZ, threshold, method, run_id, out_folder)
    plot_convergence(histories, threshold, out_folder)

    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold": threshold,
        "method": method,
        "run_id": run_id,
        "seed": BASE_SEED + run_id,
        "time_to_target_s": time_to_target,
        "total_wall_time_s": total_wall_time,
        "caps_at_target": caps_at_target,
        "best_minz_ohm": best_minz,
        "success": success,
        "n_fitness_evals": n_evals,
        "n_nodes": N_NODES,
        "n_freqs": N_FREQS,
        "n_cap_models": N_CAP_MODELS,
        "folder": out_folder,
    }
    append_result_record(record)
    return record

# ============================================================
# MAIN
# ============================================================

print(f"\nActive METHODS for this invocation: {METHODS}")
print(f"ROOT_OUT: {ROOT_OUT}")
print(f"TARGETS: {TARGETS}  NUM_RUNS: {NUM_RUNS}")

all_records = []

for threshold in TARGETS:

    for method in METHODS:

        print(f"\nMETHOD {method} TARGET {threshold}")

        method_folder = os.path.join(ROOT_OUT, f"{method}_{threshold}")
        os.makedirs(method_folder, exist_ok=True)

        for run in range(1, NUM_RUNS + 1):

            run_folder = os.path.join(method_folder, f"run_{run}")
            os.makedirs(run_folder, exist_ok=True)

            # Paired seeding: run index `run` gets the SAME NumPy global RNG
            # state regardless of method/threshold, so every method sees the
            # same particle initialisation/velocities at this run index --
            # this is what makes the resulting speedup numbers a fair,
            # paired comparison instead of independent noisy trials.
            np.random.seed(BASE_SEED + run)

            try:
                rec = run_pso(method, threshold, run_folder, run)
                all_records.append(rec)
            except Exception as exc:
                logging.getLogger("main").error(
                    "RUN_FAILED | method=%s threshold=%s run=%s | %s",
                    method, threshold, run, exc,
                )
                print(f"[ERROR] run failed: method={method} threshold={threshold} "
                      f"run={run}: {exc}")

print("\nAll runs completed for this invocation.")
print(f"Methods run this time: {METHODS}")
print(f"Results appended to: {RESULTS_JSONL_PATH}")
print("Once every method you need is present in that file, run:")
print("  python ScratchBench_plot.py")
print("to build the summary tables and figures.")