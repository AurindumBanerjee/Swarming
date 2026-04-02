# nohup python ScratchTest.py > master.log 2>&1 & 
# ps aux | grep ScratchTest.py

import os
import time
import logging
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL CONFIG
# ============================================================

ROOT_OUT = "ScratchTest3"
os.makedirs(ROOT_OUT, exist_ok=True)

NUM_RUNS = 20

TARGET_PORT = 0
THRESHOLD = 0.045
MAX_CAPS = 20

N_PARTICLES = 50
N_ITERATIONS = 15

W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5
ITER_ORDER = 2
WARM_FRACTION = 0.4

METHODS = ["numpy", "solve", "sm", "iterative"]

# ============================================================
# GLOBAL BEST TRACKER
# ============================================================

GLOBAL_BEST = {
    "method": None,
    "run": None,
    "n_caps": None,
    "min_z": np.inf,
    "cpu_time": np.inf,
    "folder": None
}

# ============================================================
# LOGGING
# ============================================================

def setup_logger(log_path):
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# ============================================================
# UTILITY – Adjacent Port Resolution
# ============================================================

def resolve_adjacent_ports(models, ports):
    """
    If a port is already used, shift to nearest available adjacent port.
    """
    used = set()

    for i in range(len(ports)):
        original = ports[i]

        if original not in used:
            used.add(original)
            continue

        # search outward: +1, -1, +2, -2, ...
        found = False
        for offset in range(1, N_NODES):
            for candidate in [original + offset, original - offset]:
                if 0 <= candidate < N_NODES and candidate not in used:
                    ports[i] = candidate
                    used.add(candidate)
                    found = True
                    break
            if found:
                break

        if not found:
            # fallback (should never happen realistically)
            for candidate in range(N_NODES):
                if candidate not in used:
                    ports[i] = candidate
                    used.add(candidate)
                    break

    return models, ports

# ============================================================
# PLOTTING
# ============================================================

def plot_convergence(history_dict, out_folder):
    plt.figure(figsize=(9,5))
    for n_caps, hist in history_dict.items():
        plt.plot(hist, label=f"n={n_caps}")
    plt.axhline(THRESHOLD, linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("Peak |Z11| (Ohm)")
    plt.title("Global Convergence")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "global_convergence.png"), dpi=150)
    plt.close()

def plot_runtime_distribution(runtimes, out_folder):
    plt.figure(figsize=(6,4))
    plt.boxplot(runtimes)
    plt.ylabel("CPU Time (s)")
    plt.title("Runtime Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "runtime_boxplot.png"), dpi=150)
    plt.close()

# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = "/DATA/Aurindum/Swarming"
DATA_DIR = os.path.join(BASE_DIR, "Data")

print("--- Loading PDN data ---")

y = sio.loadmat(os.path.join(DATA_DIR, "y2.mat"))["y"]
d = sio.loadmat(os.path.join(DATA_DIR, "decaps.mat"))["decaps"]

y = np.transpose(y, (2,0,1))
N_FREQS, N_NODES, _ = y.shape
N_CAP_MODELS = d.shape[0]

# Precompute base inverse
y_inv_base = np.zeros_like(y, dtype=np.complex128)
for f in range(N_FREQS):
    y_inv_base[f] = np.linalg.inv(y[f])

# ============================================================
# INVERSE METHODS
# ============================================================

def inv_numpy(A):
    return np.linalg.inv(A)

def inv_solve(A):
    e = np.zeros((N_NODES,), dtype=complex)
    e[TARGET_PORT] = 1.0
    x = np.linalg.solve(A, e)
    return x[TARGET_PORT]

def inv_sm(config, f):
    Z = y_inv_base[f].copy()
    for cap_idx, port in config:
        val = d[cap_idx, f]
        u = np.zeros(N_NODES, dtype=complex)
        v = np.zeros(N_NODES, dtype=complex)
        u[port] = 1.0
        v[port] = val
        denom = 1 + v @ Z @ u
        if abs(denom) < 1e-12:
            raise np.linalg.LinAlgError
        Z -= np.outer(Z @ u, v @ Z) / denom
    return Z

def inv_iterative(A, f):
    B = y_inv_base[f]
    E = A @ B - np.eye(N_NODES)
    if np.linalg.norm(E, ord=1) >= 1:
        return np.linalg.inv(A)
    corr = np.eye(N_NODES) - E
    if ITER_ORDER >= 2:
        corr += E @ E
    return B @ corr

# ============================================================
# FITNESS
# ============================================================

def evaluate_config(config, method):

    peak = 0.0

    for f in range(N_FREQS):

        A = y[f].copy()

        for cap_idx, port in config:
            A[port, port] += d[cap_idx, f]

        try:
            if method == "numpy":
                Z = inv_numpy(A)
                val = Z[TARGET_PORT, TARGET_PORT]
            elif method == "solve":
                val = inv_solve(A)
            elif method == "sm":
                Z = inv_sm(config, f)
                val = Z[TARGET_PORT, TARGET_PORT]
            elif method == "iterative":
                Z = inv_iterative(A, f)
                val = Z[TARGET_PORT, TARGET_PORT]
            else:
                raise ValueError
        except:
            return 1e200

        peak = max(peak, abs(val))

    return peak

# ============================================================
# PSO WITH WARM START
# ============================================================

def run_pso(method, out_folder, run_id):

    logger = setup_logger(os.path.join(out_folder, f"run_{run_id}.log"))

    histories = {}
    gbest_overall = np.inf
    best_n_caps = None
    prev_best_particle = None

    total_start = time.perf_counter()

    for n_caps in range(1, MAX_CAPS + 1):

        DIM = 2 * n_caps
        particles = np.random.rand(N_PARTICLES, DIM)
        velocities = np.zeros_like(particles)

        if prev_best_particle is not None:
            warm_count = int(WARM_FRACTION * N_PARTICLES)
            for i in range(warm_count):
                particles[i, :2*(n_caps-1)] = prev_best_particle
                particles[i, 2*(n_caps-1):] = np.random.rand(2)

        pbest = particles.copy()
        pbest_val = np.full(N_PARTICLES, np.inf)

        gbest = np.inf
        gbest_particle = None
        history = []

        for it in range(N_ITERATIONS):

            for i in range(N_PARTICLES):

                models = np.floor(particles[i,:n_caps] * (N_CAP_MODELS-1)).astype(int)
                ports  = np.floor(particles[i,n_caps:] * (N_NODES-1)).astype(int)

                models, ports = resolve_adjacent_ports(models, ports)

                config = list(zip(models, ports))
                cost = evaluate_config(config, method)

                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]

                if cost < gbest:
                    gbest = cost
                    gbest_particle = particles[i].copy()

            history.append(gbest)

            w = W_MAX - (W_MAX-W_MIN)*(it/N_ITERATIONS)
            velocities = w*velocities + \
                         C1*np.random.rand(*particles.shape)*(pbest-particles) + \
                         C2*np.random.rand(*particles.shape)*(gbest_particle-particles)

            particles += velocities
            particles = np.clip(particles, 0, 1)

        histories[n_caps] = history

        best_models = np.floor(gbest_particle[:n_caps] * (N_CAP_MODELS-1)).astype(int)
        best_ports  = np.floor(gbest_particle[n_caps:] * (N_NODES-1)).astype(int)

        best_models, best_ports = resolve_adjacent_ports(best_models, best_ports)

        placement_dict = {int(port): int(cap) for cap, port in zip(best_models, best_ports)}

        logger.info(
            "n_caps=%d | minZ=%.6f | placement=%s",
            n_caps,
            gbest,
            placement_dict
        )

        prev_best_particle = gbest_particle[:2*n_caps]

        if gbest < gbest_overall:
            gbest_overall = gbest
            best_n_caps = n_caps

        if gbest <= THRESHOLD:
            break

    total_time = time.perf_counter() - total_start
    plot_convergence(histories, out_folder)

    return gbest_overall, best_n_caps, total_time

# ============================================================
# MAIN BENCHMARK
# ============================================================

for method in METHODS:

    print(f"\n===== METHOD: {method} =====")

    method_folder = os.path.join(ROOT_OUT, method)
    os.makedirs(method_folder, exist_ok=True)

    runtimes = []
    best_vals = []

    for run in range(1, NUM_RUNS+1):

        run_folder = os.path.join(method_folder, f"run_{run}")
        os.makedirs(run_folder, exist_ok=True)

        best, best_caps, cpu_time = run_pso(method, run_folder, run)

        runtimes.append(cpu_time)
        best_vals.append(best)

        if best < GLOBAL_BEST["min_z"]:
            GLOBAL_BEST.update({
                "method": method,
                "run": run,
                "n_caps": best_caps,
                "min_z": best,
                "cpu_time": cpu_time,
                "folder": run_folder
            })

        print(f"{method} | Run {run:02d} | n_caps={best_caps} | "
              f"Z={best:.6f} | Time={cpu_time:.2f}s")

    plot_runtime_distribution(runtimes, method_folder)

    # ============================================================
    # METRICS FILE
    # ============================================================

    with open(os.path.join(method_folder, "metrics.txt"), "w") as f:
        f.write(f"Particles       : {N_PARTICLES}\n")
        f.write(f"Iterations      : {N_ITERATIONS}\n")
        f.write(f"Mean Runtime    : {np.mean(runtimes):.4f}\n")
        f.write(f"Std Runtime     : {np.std(runtimes):.4f}\n")
        f.write(f"Best Runtime    : {np.min(runtimes):.4f}\n")
        f.write(f"Mean Impedance  : {np.mean(best_vals):.6f}\n")
        f.write(f"Best Impedance  : {np.min(best_vals):.6f}\n")

# ============================================================
# FINAL SUMMARY
# ============================================================

summary_path = os.path.join(ROOT_OUT, "FINAL_STATISTICS.txt")
with open(summary_path, "w") as f:
    f.write("===== GLOBAL BEST INVERSE METHOD =====\n")
    for k,v in GLOBAL_BEST.items():
        f.write(f"{k:>12} : {v}\n")

print("\nAll ScratchTest runs completed.")
print("Results stored in:", ROOT_OUT)