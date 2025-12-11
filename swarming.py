import scipy.io as sio
import numpy as np
import h5py
import pyswarms as ps
import os
import time
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = 'Data'
LOG_FILE_INDEPENDENT = 'independent_search.log'
LOG_FILE_SEQUENTIAL = 'sequential_search.log'

# Physics Constraints
Z_TARGET = 0.1
MAX_CAPS = 20
VALID_NODES_START = 1
VALID_NODES_END = 20

# PSO Hyperparameters
N_PARTICLES = 50
N_ITERATIONS = 50
OPTIONS = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

# --- LOGGING SETUP ---
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # avoid duplicate handlers if logger already exists
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

# --- PLOTTING / OUTPUT HELPERS ---
def make_output_folder(method_name):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"Outputs/Method_{method_name}_{ts}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def plot_convergence(cost_history, n_caps, out_folder, method_name):
    if cost_history is None or len(cost_history) == 0:
        return
    plt.figure(figsize=(7,4))
    iters = np.arange(1, len(cost_history)+1)
    plt.plot(iters, cost_history, marker='o', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Best Peak Z (Ohms)')
    plt.title(f'{method_name} — Convergence (n_caps={n_caps})')
    plt.grid(True, linestyle='--', alpha=0.5)
    fname = os.path.join(out_folder, f"convergence_n{n_caps}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_summary(min_z_list, out_folder, method_name):
    n_caps_list = np.arange(1, len(min_z_list)+1)
    plt.figure(figsize=(7,4))
    plt.plot(n_caps_list, min_z_list, marker='s', linewidth=1)
    plt.xlabel('Number of Capacitors (N)')
    plt.ylabel('Min Peak Z (Ohms)')
    plt.title(f'{method_name} — Min Peak Z vs N_caps')
    plt.grid(True, linestyle='--', alpha=0.5)
    fname = os.path.join(out_folder, f"summary_minZ_vs_N_{method_name}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# --- DATA LOADING ---
def load_mat_file(filepath):
    try:
        return sio.loadmat(filepath)
    except (OSError, NotImplementedError, ValueError):
        try:
            data = {}
            with h5py.File(filepath, 'r') as f:
                for k, v in f.items():
                    data[k] = np.array(v)
            return data
        except Exception as e:
            raise e

print("Loading Data...")
try:
    y_file = os.path.join(DATA_DIR, 'y2.mat')
    decaps_file = os.path.join(DATA_DIR, 'decaps.mat')
    freq_file = os.path.join(DATA_DIR, 'freq2.mat')

    if not all(os.path.exists(f) for f in [y_file, decaps_file, freq_file]):
        raise FileNotFoundError(f"Files not found in {DATA_DIR}")

    y_data_raw = load_mat_file(y_file)['y']
    decaps_data_raw = load_mat_file(decaps_file)['decaps']
    freq_vec = load_mat_file(freq_file).get('freq', None)
    if freq_vec is not None:
        freq_vec = freq_vec.flatten()

    # Handle Complex Numbers (if loaded via h5py)
    if 'real' in str(y_data_raw.dtype):
        y_data_raw = y_data_raw['real'] + 1j * y_data_raw['imag']
    if 'real' in str(decaps_data_raw.dtype):
        decaps_data_raw = decaps_data_raw['real'] + 1j * decaps_data_raw['imag']

    # Fix Shapes (for numpy processing)
    # Y: (Freq, Node, Node) -> (1391, 21, 21)
    y_data_np = np.transpose(y_data_raw, (2, 0, 1))

    # Decaps: (Models, Freq) -> (3348, 1391)
    if decaps_data_raw.shape[0] == y_data_np.shape[0]:
        decaps_data_np = decaps_data_raw.T
    else:
        decaps_data_np = decaps_data_raw

    N_CAP_MODELS = decaps_data_np.shape[0]
    N_NODES = y_data_np.shape[1]
    N_FREQS = y_data_np.shape[0]

    y_base_cpu = y_data_np
    all_decap_admittances_cpu = decaps_data_np

    print(f"Data Loaded Successfully: {N_NODES} Nodes, {N_FREQS} Frequencies.")

except Exception as e:
    print(f"CRITICAL ERROR LOADING DATA: {e}")
    exit()

# --- FITNESS FUNCTION (original) retained for experimentation ---
def fitness_function_dynamic(particle_batch_np, n_caps_current):
    n_particles = particle_batch_np.shape[0]
    costs = np.zeros(n_particles)
    models_part = particle_batch_np[:, :n_caps_current]
    locs_part = particle_batch_np[:, n_caps_current:]
    models_idx = np.clip(np.floor(models_part), 0, N_CAP_MODELS - 1).astype(int)
    locs_idx = np.clip(np.floor(locs_part), VALID_NODES_START, VALID_NODES_END).astype(int)
    for i in range(n_particles):
        used_locs = set()
        row_locs = locs_idx[i]
        for k in range(n_caps_current):
            l = row_locs[k]
            while l in used_locs:
                l += 1
                if l > VALID_NODES_END: l = VALID_NODES_START
            used_locs.add(l)
            row_locs[k] = l
        locs_idx[i] = row_locs
    for i in range(n_particles):
        p_models = models_idx[i]
        p_locs = locs_idx[i]
        y_decaps_cpu = np.zeros((N_FREQS, N_NODES, N_NODES), dtype=complex)
        current_caps_cpu = all_decap_admittances_cpu[p_models, :]
        for k in range(n_caps_current):
            node_idx = p_locs[k]
            y_decaps_cpu[:, node_idx, node_idx] += current_caps_cpu[k]
        y_total = y_base_cpu + y_decaps_cpu
        try:
            z_total = np.linalg.inv(y_total)
        except np.linalg.LinAlgError:
            costs[i] = 1e200
            continue
        z_at_port_1 = np.abs(z_total[:, 0, 0])
        costs[i] = np.max(z_at_port_1)
    return costs

def decode_solution(pos, n_caps):
    models_part = pos[:n_caps]
    locs_part = pos[n_caps:]
    models_idx = np.clip(np.floor(models_part), 0, N_CAP_MODELS - 1).astype(int)
    locs_idx = np.clip(np.floor(locs_part), VALID_NODES_START, VALID_NODES_END).astype(int)
    used_locs = set()
    for k in range(n_caps):
        l = locs_idx[k]
        while l in used_locs:
            l += 1
            if l > VALID_NODES_END: l = VALID_NODES_START
        used_locs.add(l)
        locs_idx[k] = l
    config_str = ""
    for k in range(n_caps):
        config_str += f"[Node {locs_idx[k]}: Cap #{models_idx[k]}] "
    return config_str, models_idx, locs_idx

# --- STRATEGY 1: INDEPENDENT SEARCH ---
def run_independent_search():
    print("\n--- Starting STRATEGY 1: Independent Search ---")
    logger = setup_logger('independent', LOG_FILE_INDEPENDENT)
    logger.info("Starting Independent Search (1 to 20 caps)")

    out_folder = make_output_folder("Independent")
    min_z_by_n = []

    for n_caps in range(1, MAX_CAPS + 1):
        start_time = time.time()
        print(f"  Optimizing for {n_caps} capacitors...", end="", flush=True)

        current_dims = 2 * n_caps
        lb = np.concatenate([np.zeros(n_caps), np.full(n_caps, VALID_NODES_START)])
        ub = np.concatenate([np.full(n_caps, N_CAP_MODELS), np.full(n_caps, VALID_NODES_END + 0.99)])
        bounds = (lb, ub)

        optimizer = ps.single.GlobalBestPSO(n_particles=N_PARTICLES, dimensions=current_dims, options=OPTIONS, bounds=bounds)

        cost, pos = optimizer.optimize(lambda x: fitness_function_dynamic(x, n_caps), iters=N_ITERATIONS, verbose=False)

        # try to grab cost history (robust)
        cost_history = None
        if hasattr(optimizer, 'cost_history'):
            cost_history = np.array(optimizer.cost_history)
        elif hasattr(optimizer, 'cost_history_'):
            cost_history = np.array(optimizer.cost_history_)
        else:
            # fallback: repeated best_costs not available; create flat history
            cost_history = np.full(N_ITERATIONS, cost)

        plot_convergence(cost_history, n_caps, out_folder, "Independent")

        config_str, _, _ = decode_solution(pos, n_caps)
        duration = time.time() - start_time

        min_z_by_n.append(cost)

        msg = f"Count: {n_caps} | Min Peak Z: {cost:.5f} Ohms | Time: {duration:.2f}s | Config: {config_str}"
        logger.info(msg)
        print(f" Done. Min Z: {cost:.5f}")

    plot_summary(min_z_by_n, out_folder, "Independent")
    print(f"Independent outputs stored in: {out_folder}")

# --- STRATEGY 2: SEQUENTIAL (WARM START) SEARCH ---
def run_sequential_search():
    print("\n--- Starting STRATEGY 2: Sequential (Warm Start) Search ---")
    logger = setup_logger('sequential', LOG_FILE_SEQUENTIAL)
    logger.info("Starting Sequential Search (1 to 20 caps) with Warm Start")

    out_folder = make_output_folder("Sequential")
    min_z_by_n = []

    previous_best_pos = None

    for n_caps in range(1, MAX_CAPS + 1):
        start_time = time.time()
        print(f"  Optimizing for {n_caps} capacitors (Warm Start)...", end="", flush=True)

        current_dims = 2 * n_caps
        lb = np.concatenate([np.zeros(n_caps), np.full(n_caps, VALID_NODES_START)])
        ub = np.concatenate([np.full(n_caps, N_CAP_MODELS), np.full(n_caps, VALID_NODES_END + 0.99)])
        bounds = (lb, ub)

        init_pos = None
        if previous_best_pos is not None:
            n_seeded = N_PARTICLES // 2
            init_pos = np.random.uniform(low=lb, high=ub, size=(N_PARTICLES, current_dims))
            prev_n = n_caps - 1
            if prev_n > 0:
                prev_models = previous_best_pos[:prev_n]
                prev_locs = previous_best_pos[prev_n:]
                init_pos[:n_seeded, :prev_n] = prev_models
                init_pos[:n_seeded, n_caps : n_caps + prev_n] = prev_locs

        optimizer = ps.single.GlobalBestPSO(
            n_particles=N_PARTICLES,
            dimensions=current_dims,
            options=OPTIONS,
            bounds=bounds,
            init_pos=init_pos
        )

        cost, pos = optimizer.optimize(lambda x: fitness_function_dynamic(x, n_caps), iters=N_ITERATIONS, verbose=False)

        cost_history = None
        if hasattr(optimizer, 'cost_history'):
            cost_history = np.array(optimizer.cost_history)
        elif hasattr(optimizer, 'cost_history_'):
            cost_history = np.array(optimizer.cost_history_)
        else:
            cost_history = np.full(N_ITERATIONS, cost)

        plot_convergence(cost_history, n_caps, out_folder, "Sequential")

        previous_best_pos = pos

        config_str, _, _ = decode_solution(pos, n_caps)
        duration = time.time() - start_time

        min_z_by_n.append(cost)

        msg = f"Count: {n_caps} | Min Peak Z: {cost:.5f} Ohms | Time: {duration:.2f}s | Config: {config_str}"
        logger.info(msg)
        print(f" Done. Min Z: {cost:.5f}")

    plot_summary(min_z_by_n, out_folder, "Sequential")
    print(f"Sequential outputs stored in: {out_folder}")

if __name__ == "__main__":
    if os.path.exists(LOG_FILE_INDEPENDENT): os.remove(LOG_FILE_INDEPENDENT)
    if os.path.exists(LOG_FILE_SEQUENTIAL): os.remove(LOG_FILE_SEQUENTIAL)

    run_independent_search()
    run_sequential_search()

    print("\nAll tasks completed. Check .log files and Outputs/ folders for details.")
