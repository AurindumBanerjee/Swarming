# ============================================================
# Sequential PSO + Iterative Inversion (CPU)
# Baseline-Compatible Outputs, Logs, Metrics
# ============================================================


# cd /DATA/Aurindum/Swarming
# nohup python TestIterative.py > iterative.log 2>&1 &


import os
import time
import csv
import logging
from datetime import datetime

import numpy as np
import scipy.io as sio
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyswarms as ps

# ---------------- PATH CONFIG ----------------
BASE_DIR = "/DATA/Aurindum/Swarming"
DATA_DIR = os.path.join(BASE_DIR, "Data")
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")
LOG_FILE = os.path.join(BASE_DIR, "sequential_iterative_inversion.log")

# ---------------- CONFIG ----------------
THRESHOLD = 0.042
MAX_CAPS = 20
VALID_NODES_START = 1
VALID_NODES_END = 20

N_PARTICLES = 50
N_ITERATIONS = 50
N_WARM_PARTICLES = 20

OPTIONS = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

ITER_ORDER = 2            # truncation: I - E + E^2
ERROR_NORM_THRESHOLD = 1  # one-norm convergence condition

# ---------------- LOGGING ----------------
def setup_logger():
    logger = logging.getLogger("cpu_iterative")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(LOG_FILE, mode="w")
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# ---------------- OUTPUT ----------------
def make_output_folder():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        BASE_OUTPUT_DIR,
        f"Method_Sequential_IterativeInversion_{ts}"
    )
    os.makedirs(path, exist_ok=True)
    return path

# ---------------- PLOTTING ----------------
def plot_convergence(hist, n_caps, out_folder):
    plt.figure(figsize=(7,4))
    plt.plot(hist, marker='o')
    plt.axhline(
        THRESHOLD,
        color='red',
        linestyle='--',
        label=f'{THRESHOLD*1e3:.0f} mΩ'
    )
    plt.xlabel("Iteration")
    plt.ylabel("|Z11| (Ohms)")
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
        label=f'{THRESHOLD*1e3:.0f} mΩ'
    )
    plt.xlabel("Iteration")
    plt.ylabel("|Z11| (Ohms)")
    plt.title("Global Convergence — Iterative Inversion")
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

# ---------------- LOAD DATA ----------------
print("Loading PDN data (Iterative Inversion)...")

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

print(f"N_FREQS={N_FREQS}, N_NODES={N_NODES}, N_CAP_MODELS={N_CAP_MODELS}")

print("Precomputing base inverse...")
Z_base = np.linalg.inv(y)

# ---------------- ITERATIVE INVERSION ----------------
def iterative_inverse(A, B_prev):
    I = np.eye(A.shape[0], dtype=A.dtype)

    E = A @ B_prev - I
    err = np.linalg.norm(E, ord=1)

    if err >= ERROR_NORM_THRESHOLD:
        return np.linalg.inv(A), False

    if ITER_ORDER == 1:
        B = B_prev @ (I - E)
    else:
        B = B_prev @ (I - E + E @ E)

    return B, True

# ---------------- FITNESS ----------------
def fitness_iterative(x, n_caps):
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

        Z = Z_base.copy()

        for f in range(N_FREQS):
            Yf = y[f].copy()
            for k in range(n_caps):
                Yf[locs[k], locs[k]] += d[models[k], f]

            Z[f], _ = iterative_inverse(Yf, Z[f])

        costs[i] = np.max(np.abs(Z[:, 0, 0]))

    return costs

# ---------------- SEQUENTIAL PSO ----------------
def run_sequential():
    logger = setup_logger()
    out_folder = make_output_folder()

    histories = {}
    prev_best = None
    metrics = []

    cfg_path = os.path.join(out_folder, "best_config_by_n.txt")
    with open(cfg_path, "w") as f:
        f.write("n_caps | min_Z | models | nodes\n")

        total_start = time.perf_counter()

        for n_caps in range(1, MAX_CAPS + 1):
            print(f"Optimizing n_caps={n_caps}...", end="", flush=True)

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
                init_pos = np.random.uniform(lb, ub, (N_PARTICLES, 2*n_caps))
                init_pos[:N_WARM_PARTICLES, :n_caps-1] = prev_best[:n_caps-1]
                init_pos[:N_WARM_PARTICLES, n_caps:n_caps+(n_caps-1)] = prev_best[n_caps-1:]

            opt = ps.single.GlobalBestPSO(
                n_particles=N_PARTICLES,
                dimensions=2*n_caps,
                options=OPTIONS,
                bounds=(lb, ub),
                init_pos=init_pos
            )

            t0 = time.perf_counter()
            cost, pos = opt.optimize(
                lambda x: fitness_iterative(x, n_caps),
                iters=N_ITERATIONS,
                verbose=False
            )
            elapsed = time.perf_counter() - t0

            hist = np.array(opt.cost_history)
            histories[n_caps] = hist
            plot_convergence(hist, n_caps, out_folder)

            models = np.clip(np.floor(pos[:n_caps]), 0, N_CAP_MODELS - 1).astype(int).tolist()
            nodes = np.clip(np.floor(pos[n_caps:]), VALID_NODES_START, VALID_NODES_END).astype(int).tolist()

            logger.info(
                "n_caps=%d | minZ=%.6f | time=%.2fs | models=%s | nodes=%s",
                n_caps, cost, elapsed, models, nodes
            )

            f.write(f"{n_caps} | {cost:.6f} | {models} | {nodes}\n")
            metrics.append({
                "n_caps": n_caps,
                "min_impedance": cost,
                "time_sec": elapsed,
                "models": models,
                "nodes": nodes
            })

            prev_best = pos
            print(f" done | minZ={cost:.6f}")

            if cost <= THRESHOLD:
                break

    total_time = time.perf_counter() - total_start
    plot_global_convergence(histories, out_folder)

    with open(os.path.join(out_folder, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    with open(os.path.join(out_folder, "metrics.txt"), "w") as f:
        f.write(f"Total CPU Time (s): {total_time:.4f}\n")
        f.write("Method: Iterative Inversion\n")

    logger.info("Total CPU time %.4f s", total_time)

    for h in logger.handlers:
        h.flush()
        h.close()

    print(f"\nTotal CPU time: {total_time:.2f}s")
    print(f"Results stored in: {out_folder}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    run_sequential()
