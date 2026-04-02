# nohup python -u MinImp.py > MinImp.log 2>&1 &
# ps aux | grep MinImp.py
# pkill -9 -f MinImp.py


# ==========================================================
# MinImp — PSO inversion benchmark for PDN decap placement
# ==========================================================

import os
import time
import logging
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# GLOBAL CONFIG
# ==========================================================

ROOT_OUT = "MinImp4"
os.makedirs(ROOT_OUT, exist_ok=True)

NUM_RUNS = 20

TARGET_PORT = 0
MAX_CAPS = 20

PARTICLE_OPTIONS = [20, 30, 50, 80]
ITER_OPTIONS = [10, 15, 25, 40, 50]

W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5

# METHODS = ["numpy", "solve", "sm", "woodbury", "iterative"]
METHODS = ["numpy", "solve", "sm"]

# ==========================================================
# LOGGING
# ==========================================================

def setup_logger(path):

    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(path)
    formatter = logging.Formatter('%(asctime)s INFO: %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


# ==========================================================
# LOAD DATA
# ==========================================================

BASE_DIR = "/DATA/Aurindum/Swarming"
DATA_DIR = os.path.join(BASE_DIR, "Data")

print("Loading PDN data...")

y = sio.loadmat(os.path.join(DATA_DIR, "y2.mat"))["y"]
d = sio.loadmat(os.path.join(DATA_DIR, "decaps.mat"))["decaps"]

y = np.transpose(y, (2, 0, 1))

N_FREQS, N_NODES, _ = y.shape
N_CAP_MODELS = d.shape[0]

# ==========================================================
# PRECOMPUTE BASE INVERSE
# ==========================================================

y_inv_base = np.zeros_like(y, dtype=np.complex128)

for f in range(N_FREQS):
    y_inv_base[f] = np.linalg.inv(y[f])


# ==========================================================
# PORT RESOLUTION
# ==========================================================

def resolve_adjacent_ports(models, ports):

    used = set()

    for i in range(len(ports)):

        if ports[i] not in used:
            used.add(ports[i])
            continue

        for offset in range(1, N_NODES):

            for cand in [ports[i]+offset, ports[i]-offset]:

                if 0 <= cand < N_NODES and cand not in used:
                    ports[i] = cand
                    used.add(cand)
                    break

            if ports[i] in used:
                break

    return models, ports


# ==========================================================
# INVERSION METHODS
# ==========================================================

def inv_numpy(A):
    Z = np.linalg.inv(A)
    return Z[TARGET_PORT, TARGET_PORT]


def inv_solve(A):
    e = np.zeros(N_NODES, dtype=complex)
    e[TARGET_PORT] = 1
    x = np.linalg.solve(A, e)
    return x[TARGET_PORT]


def inv_sm(config, f):

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


def inv_woodbury(config, f):

    Z = y_inv_base[f]

    k = len(config)

    U = np.zeros((N_NODES,k), dtype=np.complex128)
    V = np.zeros((N_NODES,k), dtype=np.complex128)

    for i, (cap, port) in enumerate(config):
        val = d[cap, f]
        U[port, i] = 1
        V[port, i] = val

    middle = np.linalg.inv(np.eye(k) + V.T @ Z @ U)

    Z_new = Z - Z @ U @ middle @ V.T @ Z

    return Z_new[TARGET_PORT, TARGET_PORT]


def inv_iterative(A, f):

    B = y_inv_base[f]
    E = A @ B - np.eye(N_NODES)

    if np.linalg.norm(E, 1) >= 1:
        return np.linalg.inv(A)[TARGET_PORT, TARGET_PORT]

    corr = np.eye(N_NODES) - E + E @ E
    Z = B @ corr

    return Z[TARGET_PORT, TARGET_PORT]


# ==========================================================
# FITNESS
# ==========================================================

def evaluate_config(config, method):

    peak = 0

    for f in range(N_FREQS):

        A = y[f].copy()

        for cap, port in config:
            A[port, port] += d[cap, f]

        try:

            if method == "numpy":
                val = inv_numpy(A)

            elif method == "solve":
                val = inv_solve(A)

            elif method == "sm":
                val = inv_sm(config, f)

            elif method == "woodbury":
                val = inv_woodbury(config, f)

            elif method == "iterative":
                val = inv_iterative(A, f)

        except:
            return 1e200

        peak = max(peak, abs(val))

        if peak > 1.2: 
            return peak

    return peak


# ==========================================================
# PSO
# ==========================================================

def run_pso(method, out_folder, run_id, n_particles, n_iterations):

    logger = setup_logger(os.path.join(out_folder, f"run_{run_id}.log"))

    prev_best = None

    # ---- store for plotting ----
    run_caps = []
    run_minZ = []

    for n_caps in range(1, MAX_CAPS+1):

        DIM = 2 * n_caps

        particles = np.random.rand(n_particles, DIM)
        velocities = np.zeros_like(particles)

        if prev_best is not None:

            warm = int(0.4 * n_particles)

            for i in range(warm):
                particles[i, :2*(n_caps-1)] = prev_best
                particles[i, 2*(n_caps-1):] = np.random.rand(2)

        pbest = particles.copy()
        pbest_val = np.full(n_particles, np.inf)

        gbest = np.inf
        gbest_particle = None

        start = time.time()

        for it in range(n_iterations):

            for i in range(n_particles):

                models = np.floor(
                    particles[i, :n_caps]*(N_CAP_MODELS-1)
                ).astype(int)

                ports = np.floor(
                    particles[i, n_caps:]*(N_NODES-1)
                ).astype(int)

                models, ports = resolve_adjacent_ports(models, ports)

                config = list(zip(models, ports))

                cost = evaluate_config(config, method)

                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]

                if cost < gbest:
                    gbest = cost
                    gbest_particle = particles[i].copy()

            w = W_MAX - (W_MAX-W_MIN)*(it/n_iterations)

            velocities = (
                w*velocities
                + C1*np.random.rand(*particles.shape)*(pbest-particles)
                + C2*np.random.rand(*particles.shape)*(gbest_particle-particles)
            )

            particles += velocities
            particles = np.clip(particles, 0, 1)

        runtime = time.time() - start

        best_models = np.floor(
            gbest_particle[:n_caps]*(N_CAP_MODELS-1)
        ).astype(int)

        best_ports = np.floor(
            gbest_particle[n_caps:]*(N_NODES-1)
        ).astype(int)

        best_models, best_ports = resolve_adjacent_ports(best_models, best_ports)

        placement = {
            int(port): int(cap)
            for cap, port in zip(best_models, best_ports)
        }

        logger.info(
            "particles=%d iter=%d n_caps=%d minZ=%.6f runtime=%.2fs placement=%s",
            n_particles,
            n_iterations,
            n_caps,
            gbest,
            runtime,
            placement
        )

        # ---- store ----
        run_caps.append(n_caps)
        run_minZ.append(gbest)

        prev_best = gbest_particle[:2*n_caps]

    # ==========================================================
    # PER-RUN PLOT (NOT part of timing)
    # ==========================================================

    if len(run_caps) > 0:

        caps_sorted, z_sorted = zip(*sorted(zip(run_caps, run_minZ)))

        plt.figure()

        plt.plot(caps_sorted, z_sorted, marker='o')

        plt.xlabel("Number of Decaps")
        plt.ylabel("Minimum Impedance")

        plt.title(
            f"{method} | P{n_particles} I{n_iterations} | Run {run_id}"
        )

        plt.grid()

        plt.savefig(
            os.path.join(out_folder, f"run_{run_id}_plot.png"),
            dpi=300
        )

        plt.close()


# ==========================================================
# MAIN BENCHMARK
# ==========================================================

for method in METHODS:

    for particles in PARTICLE_OPTIONS:

        for iterations in ITER_OPTIONS:

            print(f"\nMETHOD {method} P{particles} I{iterations}")

            folder = os.path.join(
                ROOT_OUT,
                f"{method}_P{particles}_I{iterations}"
            )

            os.makedirs(folder, exist_ok=True)

            for run in range(1, NUM_RUNS+1):

                run_folder = os.path.join(folder, f"run_{run}")
                os.makedirs(run_folder, exist_ok=True)

                run_pso(
                    method,
                    run_folder,
                    run,
                    particles,
                    iterations
                )

print("All runs completed.")



# ==========================================================
# GLOBAL ANALYSIS + STATISTICS + PLOTS
# ==========================================================

import re
import pandas as pd
import matplotlib.pyplot as plt

print("Starting global analysis...")

records = []

# ==========================================================
# PARSE ALL LOG FILES
# ==========================================================

for root, dirs, files in os.walk(ROOT_OUT):

    for file in files:

        if not file.endswith(".log"):
            continue

        path = os.path.join(root, file)

        # folder format: method_Px_Iy/run_z/
        parts = root.split(os.sep)

        try:
            config_folder = parts[-2]
            method = config_folder.split("_")[0]
            particles = int(config_folder.split("_")[1][1:])
            iterations = int(config_folder.split("_")[2][1:])
            run_id = int(parts[-1].split("_")[1])
        except:
            continue

        with open(path) as f:

            for line in f:

                if "minZ" not in line:
                    continue

                m = re.search(
                    r"n_caps=(\d+).*minZ=([0-9\.]+).*runtime=([0-9\.]+)",
                    line
                )

                if not m:
                    continue

                records.append({
                    "method": method,
                    "particles": particles,
                    "iterations": iterations,
                    "run": run_id,
                    "n_caps": int(m.group(1)),
                    "minZ": float(m.group(2)),
                    "runtime": float(m.group(3))
                })

df = pd.DataFrame(records)

if df.empty:
    print("No data found. Skipping global analysis.")
    exit()

plots_dir = os.path.join(ROOT_OUT, "plots_global")
os.makedirs(plots_dir, exist_ok=True)

# ==========================================================
# GLOBAL PLOT 1 — Impedance vs Decaps (best per method)
# ==========================================================

plt.figure()

for method in df["method"].unique():

    g = df[df["method"] == method]
    g = g.groupby("n_caps")["minZ"].min()

    plt.plot(g.index, g.values, label=method)

plt.xlabel("Number of Decaps")
plt.ylabel("Minimum Impedance")
plt.title("Global Best Impedance vs Decaps")
plt.legend()
plt.grid()

plt.savefig(os.path.join(plots_dir, "global_impedance_vs_decaps.png"), dpi=300)
plt.close()


# ==========================================================
# GLOBAL PLOT 2 — Runtime comparison
# ==========================================================

plt.figure()

runtime = df.groupby("method")["runtime"].mean()

plt.bar(runtime.index, runtime.values)

plt.ylabel("Average Runtime (s)")
plt.title("Average Runtime Comparison")

plt.savefig(os.path.join(plots_dir, "global_runtime.png"), dpi=300)
plt.close()


# ==========================================================
# GLOBAL PLOT 3 — PSO sensitivity (particles)
# ==========================================================

for p in sorted(df["particles"].unique()):

    plt.figure()

    sub = df[df["particles"] == p]

    for it in sorted(sub["iterations"].unique()):

        g = sub[sub["iterations"] == it]
        g = g.groupby("n_caps")["minZ"].min()

        plt.plot(g.index, g.values, label=f"iter={it}")

    plt.xlabel("Decaps")
    plt.ylabel("Min Impedance")
    plt.title(f"Particles = {p}")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(plots_dir, f"particles_{p}.png"), dpi=300)
    plt.close()


# ==========================================================
# STATISTICS — BEST METHOD PER CONFIG
# ==========================================================

print("\n===== BEST METHOD PER (particles, iterations) =====")

grouped = df.groupby(["particles", "iterations", "method"])["minZ"].min().reset_index()

best_per_config = grouped.loc[
    grouped.groupby(["particles", "iterations"])["minZ"].idxmin()
]

for _, row in best_per_config.iterrows():

    print(
        f"P={row['particles']} I={row['iterations']} "
        f"-> Best: {row['method']} (minZ={row['minZ']:.6f})"
    )


# ==========================================================
# GLOBAL BEST (across everything)
# ==========================================================

best_row = df.loc[df["minZ"].idxmin()]

print("\n===== GLOBAL BEST RESULT =====")

print(
    f"Method: {best_row['method']}\n"
    f"Particles: {best_row['particles']}\n"
    f"Iterations: {best_row['iterations']}\n"
    f"Run: {best_row['run']}\n"
    f"Decaps: {best_row['n_caps']}\n"
    f"Minimum Impedance: {best_row['minZ']:.6f}\n"
    f"Runtime: {best_row['runtime']:.2f}s"
)


# ==========================================================
# OPTIONAL — SAVE SUMMARY CSV
# ==========================================================

summary_path = os.path.join(ROOT_OUT, "summary.csv")
df.to_csv(summary_path, index=False)

print("\nSummary saved to:", summary_path)
print("Global analysis complete.")