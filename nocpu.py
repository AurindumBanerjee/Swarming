import scipy.io as sio
import numpy as np
import h5py
import pyswarms as ps
import os
import time
import logging

# --- CONFIGURATION ---
DATA_DIR = 'Data'
LOG_FILE_INDEPENDENT = 'independent_search.log'
LOG_FILE_SEQUENTIAL = 'sequential_search.log'

# Physics Constraints
Z_TARGET = 0.1 # Not strictly used for stopping, but good for reference
MAX_CAPS = 20  # Max number of capacitors to try
# Node 0 is reserved for Port 1 (Measurement), so valid nodes are 1..20
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
    logger.addHandler(handler)
    return logger

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
    
    if not all(os.path.exists(f) for f in [y_file, decaps_file]):
        raise FileNotFoundError(f"Files not found in {DATA_DIR}")

    # Load Raw Data
    y_data_raw = load_mat_file(y_file)['y']
    decaps_data_raw = load_mat_file(decaps_file)['decaps']

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

    # Metadata
    N_CAP_MODELS = decaps_data_np.shape[0]
    N_NODES = y_data_np.shape[1]
    N_FREQS = y_data_np.shape[0]

    # Keep data as standard numpy arrays (CPU memory)
    y_base_cpu = y_data_np
    all_decap_admittances_cpu = decaps_data_np

    print(f"Data Loaded Successfully: {N_NODES} Nodes, {N_FREQS} Frequencies.")

except Exception as e:
    print(f"CRITICAL ERROR LOADING DATA: {e}")
    exit()

# --- FITNESS FUNCTION ---
def fitness_function_dynamic(particle_batch_np, n_caps_current):
    """
    Calculates Peak Impedance at Port 1 (Node 0) for a batch of particles.
    """
    n_particles = particle_batch_np.shape[0]
    costs = np.zeros(n_particles)
    
    # 1. Decode Batch
    models_part = particle_batch_np[:, :n_caps_current]
    locs_part = particle_batch_np[:, n_caps_current:]
    
    models_idx = np.clip(np.floor(models_part), 0, N_CAP_MODELS - 1).astype(int)
    # Clip locations to 1..20 (Strictly excluding Node 0)
    locs_idx = np.clip(np.floor(locs_part), VALID_NODES_START, VALID_NODES_END).astype(int)
    
    # 2. Collision Resolution
    for i in range(n_particles):
        used_locs = set()
        row_locs = locs_idx[i]
        for k in range(n_caps_current):
            l = row_locs[k]
            while l in used_locs:
                l += 1
                if l > VALID_NODES_END: l = VALID_NODES_START # Wrap around 1..20
            used_locs.add(l)
            row_locs[k] = l
        locs_idx[i] = row_locs

    # 3. CPU Evaluation
    for i in range(n_particles):
        p_models = models_idx[i]
        p_locs = locs_idx[i]
        
        # Initialize diagonal matrix batch (Freq x Node x Node)
        y_decaps_cpu = np.zeros((N_FREQS, N_NODES, N_NODES), dtype=complex)
        
        # Get admittances for chosen caps
        current_caps_cpu = all_decap_admittances_cpu[p_models, :]
        
        # Add to diagonals
        for k in range(n_caps_current):
            node_idx = p_locs[k]
            y_decaps_cpu[:, node_idx, node_idx] += current_caps_cpu[k]
        
        # Solve Circuit
        y_total = y_base_cpu + y_decaps_cpu
        
        try:
            # np.linalg.inv works on stacks of matrices just like cupy
            z_total = np.linalg.inv(y_total)
        except np.linalg.LinAlgError:
            costs[i] = 1e200 # Penalty for singular matrix
            continue
        
        # Measure Port 1 (Index 0,0) Only
        z_at_port_1 = np.abs(z_total[:, 0, 0])
        
        # Cost = Peak Impedance
        costs[i] = np.max(z_at_port_1)
        
    return costs

def decode_solution(pos, n_caps):
    """Helper to convert raw particle position to readable format"""
    models_part = pos[:n_caps]
    locs_part = pos[n_caps:]
    
    models_idx = np.clip(np.floor(models_part), 0, N_CAP_MODELS - 1).astype(int)
    locs_idx = np.clip(np.floor(locs_part), VALID_NODES_START, VALID_NODES_END).astype(int)
    
    # Resolve collisions for final display
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
    
    for n_caps in range(1, MAX_CAPS + 1):
        start_time = time.time()
        print(f"  Optimizing for {n_caps} capacitors...", end="", flush=True)
        
        # Setup PSO
        current_dims = 2 * n_caps
        lb = np.concatenate([np.zeros(n_caps), np.full(n_caps, VALID_NODES_START)])
        ub = np.concatenate([np.full(n_caps, N_CAP_MODELS), np.full(n_caps, VALID_NODES_END + 0.99)])
        bounds = (lb, ub)
        
        optimizer = ps.single.GlobalBestPSO(n_particles=N_PARTICLES, dimensions=current_dims, options=OPTIONS, bounds=bounds)
        
        # Run
        cost, pos = optimizer.optimize(lambda x: fitness_function_dynamic(x, n_caps), iters=N_ITERATIONS, verbose=False)
        
        # Log Results
        config_str, _, _ = decode_solution(pos, n_caps)
        duration = time.time() - start_time
        
        msg = f"Count: {n_caps} | Min Peak Z: {cost:.5f} Ohms | Time: {duration:.2f}s | Config: {config_str}"
        logger.info(msg)
        print(f" Done. Min Z: {cost:.5f}")

# --- STRATEGY 2: SEQUENTIAL (WARM START) SEARCH ---
def run_sequential_search():
    print("\n--- Starting STRATEGY 2: Sequential (Warm Start) Search ---")
    logger = setup_logger('sequential', LOG_FILE_SEQUENTIAL)
    logger.info("Starting Sequential Search (1 to 20 caps) with Warm Start")
    
    previous_best_pos = None
    
    for n_caps in range(1, MAX_CAPS + 1):
        start_time = time.time()
        print(f"  Optimizing for {n_caps} capacitors (Warm Start)...", end="", flush=True)
        
        # Setup PSO
        current_dims = 2 * n_caps
        lb = np.concatenate([np.zeros(n_caps), np.full(n_caps, VALID_NODES_START)])
        ub = np.concatenate([np.full(n_caps, N_CAP_MODELS), np.full(n_caps, VALID_NODES_END + 0.99)])
        bounds = (lb, ub)
        
        # Create Warm Start Population
        init_pos = None
        if previous_best_pos is not None:
            # Seed 50% of swarm
            n_seeded = N_PARTICLES // 2
            init_pos = np.random.uniform(low=lb, high=ub, size=(N_PARTICLES, current_dims))
            
            # Extract previous solution components
            prev_n = n_caps - 1
            prev_models = previous_best_pos[:prev_n]
            prev_locs = previous_best_pos[prev_n:]
            
            # Inject into new particles
            # Copy models to first slots, locations to corresponding slots
            init_pos[:n_seeded, :prev_n] = prev_models
            init_pos[:n_seeded, n_caps : n_caps + prev_n] = prev_locs
        
        optimizer = ps.single.GlobalBestPSO(
            n_particles=N_PARTICLES, 
            dimensions=current_dims, 
            options=OPTIONS, 
            bounds=bounds,
            init_pos=init_pos # Inject seeds
        )
        
        # Run
        cost, pos = optimizer.optimize(lambda x: fitness_function_dynamic(x, n_caps), iters=N_ITERATIONS, verbose=False)
        
        # Save for next iteration
        previous_best_pos = pos
        
        # Log Results
        config_str, _, _ = decode_solution(pos, n_caps)
        duration = time.time() - start_time
        
        msg = f"Count: {n_caps} | Min Peak Z: {cost:.5f} Ohms | Time: {duration:.2f}s | Config: {config_str}"
        logger.info(msg)
        print(f" Done. Min Z: {cost:.5f}")

if __name__ == "__main__":
    # Clear logs if they exist
    if os.path.exists(LOG_FILE_INDEPENDENT): os.remove(LOG_FILE_INDEPENDENT)
    if os.path.exists(LOG_FILE_SEQUENTIAL): os.remove(LOG_FILE_SEQUENTIAL)

    run_independent_search()
    run_sequential_search()
    
    print("\nAll tasks completed. Check .log files for details.")