# ============================================================
# CPU BASELINE — Sequential PSO c1 / c2 Parameter Sweep
# Fixed: 50 Particles, 50 Iterations
# Sweeps c1, c2 ∈ [0.1, 3.0] with step 0.1
# Tracks GLOBAL BEST configuration
# ============================================================

import os
import time
import logging
from datetime import datetime

import numpy as np
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
import pyswarms as ps

# ---------------- GLOBAL CONFIG ----------------
BASE_DIR = "/DATA/Aurindum/Swarming"
DATA_DIR = os.path.join(BASE_DIR, "Data")

THRESHOLD = 0.050
MAX_CAPS = 20
VALID_NODES_START = 1
VALID_NODES_END = 20

N_PARTICLES = 50
N_ITERATIONS = 50
WARM_FRACTION = 0.4

W_INERTIA = 0.7

C_RANGE = np.round(np.arange(0.1, 3.01, 0.1), 2)

# ---------------- OUTPUT ROOT ----------------
ROOT_OUT = os.path.join(
    BASE_DIR,
    "Outputs",
    datetime.now().strftime("%Y%m%d_%H%M%S_c1c2_Sweep")
)
os.makedirs(ROOT_OUT, exist_ok=True)

# ---------------- GLOBAL BEST TRACKER ----------------
GLOBAL_BEST = {
    "c1": None,
    "c2": None,
    "n_caps": None,
    "min_z": np.inf,
    "cpu_time": np.inf,
    "folder": None
}

# ---------------- LOGGING ----------------
def setup_logger(log_path):
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# ---------------- PLOTTING ----------------
def plot_convergence(cost_history, n_caps, out_folder):
    iters = np.arange(1, len(cost_history) + 1)
    plt.figure(figsize=(7,4))
    plt.plot(iters, cost_history, marker='o')
    plt.axhline(
        THRESHOLD,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'{THRESHOLD*1e3:.0f} mΩ'
    )
    plt.xlabel("Iteration")
    plt.ylabel("Peak |Z11| (Ohms)")
    plt.title(f"Convergence — n_caps={n_caps}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_folder, f"convergence_n{n_caps}.png"),
        dpi=150
    )
    plt.close()

def plot_global_convergence(histories, out_folder):
    plt.figure(figsize=(9,5))
    for n, hist in histories.items():
        plt.plot(hist, label=f'n={n}')
    plt.axhline(
        THRESHOLD,
        color='red',
        linestyle='--',
        linewidth=2
    )
    plt.xlabel("Iteration")
    plt.ylabel("Peak |Z11| (Ohms)")
    plt.title("Global Convergence — c1/c2 Sweep")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_folder, "global_convergence.png"),
        dpi=150
    )
    plt.close()

# ---------------- MAT LOADING ----------------
def load_mat(filepath):
    try:
        return sio.loadmat(filepath)
    except Exception:
        with h5py.File(filepath, 'r') as f:
            return {k: np.array(v) for k, v in f.items()}

def extract_var(mat):
    return mat[list(mat.keys() - {'__header__','__version__','__globals__'})[0]]

print("Loading PDN data once (shared across all runs)...")

y = extract_var(load_mat(os.path.join(DATA_DIR, "y2.mat")))
d = extract_var(load_mat(os.path.join(DATA_DIR, "decaps.mat")))

if 'real' in str(y.dtype):
    y = y['real'] + 1j * y['imag']
if 'real' in str(d.dtype):
    d = d['real'] + 1j * d['imag']

y = np.transpose(y, (2, 0, 1))
if d.shape[0] == y.shape[0]:
    d = d.T

N_FREQS, N_NODES, _ = y.shape
N_CAP_MODELS = d.shape[0]

print(f"Loaded: N_FREQS={N_FREQS}, N_NODES={N_NODES}, N_CAP_MODELS={N_CAP_MODELS}")

# ---------------- FITNESS ----------------
def fitness_cpu(x, n_caps):
    costs = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        models = np.clip(
            np.floor(x[i, :n_caps]), 0, N_CAP_MODELS - 1
        ).astype(int)

        locs = np.clip(
            np.floor(x[i, n_caps:]),
            VALID_NODES_START, VALID_NODES_END
        ).astype(int)

        used = set()
        for k in range(n_caps):
            while locs[k] in used:
                locs[k] += 1
                if locs[k] > VALID_NODES_END:
                    locs[k] = VALID_NODES_START
            used.add(locs[k])

        y_total = y.copy()
        for k in range(n_caps):
            y_total[:, locs[k], locs[k]] += d[models[k]]

        try:
            z = np.linalg.inv(y_total)
            costs[i] = np.max(np.abs(z[:, 0, 0]))
        except np.linalg.LinAlgError:
            costs[i] = 1e200

    return costs

# ---------------- MAIN SWEEP ----------------
for c1 in C_RANGE:
    for c2 in C_RANGE:

        OPTIONS = {'c1': float(c1), 'c2': float(c2), 'w': W_INERTIA}
        tag = f"c1_{c1:.1f}_c2_{c2:.1f}"

        out_folder = os.path.join(ROOT_OUT, tag)
        os.makedirs(out_folder, exist_ok=True)

        logger = setup_logger(os.path.join(out_folder, "run.log"))
        print(f"\n=== Running {tag} ===")

        histories = {}
        prev_best = None

        total_start = time.perf_counter()
        best_cfg_path = os.path.join(out_folder, "best_config_by_n.txt")

        with open(best_cfg_path, "w") as f:
            f.write("n_caps | min_Z | models | nodes\n")

            for n_caps in range(1, MAX_CAPS + 1):

                lb = np.concatenate([
                    np.zeros(n_caps),
                    np.full(n_caps, VALID_NODES_START)
                ])
                ub = np.concatenate([
                    np.full(n_caps, N_CAP_MODELS - 1 + 0.99),
                    np.full(n_caps, VALID_NODES_END + 0.99)
                ])

                init_pos = None
                if prev_best is not None:
                    warm = int(WARM_FRACTION * N_PARTICLES)
                    init_pos = np.random.uniform(lb, ub, (N_PARTICLES, 2*n_caps))
                    init_pos[:warm, :n_caps-1] = prev_best[:n_caps-1]
                    init_pos[:warm, n_caps:n_caps+(n_caps-1)] = prev_best[n_caps-1:]

                optimizer = ps.single.GlobalBestPSO(
                    n_particles=N_PARTICLES,
                    dimensions=2*n_caps,
                    options=OPTIONS,
                    bounds=(lb, ub),
                    init_pos=init_pos
                )

                cost, pos = optimizer.optimize(
                    lambda x: fitness_cpu(x, n_caps),
                    iters=N_ITERATIONS,
                    verbose=False
                )

                hist = np.array(optimizer.cost_history)
                histories[n_caps] = hist
                plot_convergence(hist, n_caps, out_folder)

                f.write(f"{n_caps} | {cost:.6f}\n")
                prev_best = pos

                if cost <= THRESHOLD:
                    break

        total_time = time.perf_counter() - total_start
        plot_global_convergence(histories, out_folder)

        last_n = max(histories.keys())
        last_min_z = np.min(histories[last_n])

        if last_min_z <= THRESHOLD:
            better = False
            if GLOBAL_BEST["n_caps"] is None:
                better = True
            elif last_n < GLOBAL_BEST["n_caps"]:
                better = True
            elif last_n == GLOBAL_BEST["n_caps"]:
                if last_min_z < GLOBAL_BEST["min_z"]:
                    better = True
                elif abs(last_min_z - GLOBAL_BEST["min_z"]) < 1e-6:
                    if total_time < GLOBAL_BEST["cpu_time"]:
                        better = True

            if better:
                GLOBAL_BEST.update({
                    "c1": c1,
                    "c2": c2,
                    "n_caps": last_n,
                    "min_z": last_min_z,
                    "cpu_time": total_time,
                    "folder": out_folder
                })

        print(f"Completed {tag} | Time = {total_time:.2f}s")

# ---------------- FINAL SUMMARY ----------------
summary_path = os.path.join(ROOT_OUT, "BEST_COMBINATION.txt")
with open(summary_path, "w") as f:
    f.write("===== GLOBAL BEST PSO c1/c2 CONFIGURATION =====\n")
    for k, v in GLOBAL_BEST.items():
        f.write(f"{k:>12} : {v}\n")

print("\n===== GLOBAL BEST CONFIGURATION =====")
for k, v in GLOBAL_BEST.items():
    print(f"{k:>12} : {v}")
print("====================================")

print("\nALL c1/c2 PARAMETER SWEEP RUNS COMPLETED.")
print("Results stored in:", ROOT_OUT)
