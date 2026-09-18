import os
import time
import logging
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")

# ============================================================
# GLOBAL CONFIG
# ============================================================

ROOT_OUT = "MinTime/Refer"
os.makedirs(ROOT_OUT, exist_ok=True)

NUM_RUNS = 2

TARGET_PORT = 0
TARGETS = [0.04,0.03]

MAX_CAPS = 20

N_PARTICLES = 50
N_ITERATIONS = 15

W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5

METHODS = ["numpy","solve","sm","iterative"]

# ============================================================
# LOGGING
# ============================================================

def setup_logger(path):

    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(path)

    formatter = logging.Formatter('%(asctime)s INFO: %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = "/DATA/Aurindum/Swarming"
DATA_DIR = os.path.join(BASE_DIR,"Data")

print("Loading PDN data...")

y = sio.loadmat(os.path.join(DATA_DIR,"y2.mat"))["y"]
d = sio.loadmat(os.path.join(DATA_DIR,"decaps.mat"))["decaps"]

y = np.transpose(y,(2,0,1))

N_FREQS,N_NODES,_ = y.shape
N_CAP_MODELS = d.shape[0]


# ============================================================
# PRECOMPUTE BASE INVERSE
# ============================================================

y_inv_base = np.zeros_like(y,dtype=np.complex128)

for f in range(N_FREQS):
    y_inv_base[f] = np.linalg.inv(y[f])


# ============================================================
# PORT RESOLUTION
# ============================================================

def resolve_adjacent_ports(models,ports):

    used=set()

    for i in range(len(ports)):

        if ports[i] not in used:
            used.add(ports[i])
            continue

        for offset in range(1,N_NODES):

            for cand in [ports[i]+offset,ports[i]-offset]:

                if 0<=cand<N_NODES and cand not in used:

                    ports[i]=cand
                    used.add(cand)
                    break

            if ports[i] in used:
                break

    return models,ports


# ============================================================
# INVERSION METHODS
# ============================================================

def inv_numpy(A):

    Z = np.linalg.inv(A)
    return Z[TARGET_PORT,TARGET_PORT]


def inv_solve(A):

    e = np.zeros(N_NODES,dtype=complex)
    e[TARGET_PORT] = 1

    x = np.linalg.solve(A,e)

    return x[TARGET_PORT]


def inv_sm(config,f):

    Z = y_inv_base[f].copy()

    for cap,port in config:

        val = d[cap,f]

        u = np.zeros(N_NODES)
        v = np.zeros(N_NODES)

        u[port] = 1
        v[port] = val

        denom = 1 + v@Z@u

        if abs(denom) < 1e-12:
            raise np.linalg.LinAlgError

        Z -= np.outer(Z@u,v@Z)/denom

    return Z[TARGET_PORT,TARGET_PORT]


def inv_iterative(A,f):

    B = y_inv_base[f]

    E = A@B - np.eye(N_NODES)

    if np.linalg.norm(E,1) >= 1:
        return np.linalg.inv(A)[TARGET_PORT,TARGET_PORT]

    corr = np.eye(N_NODES) - E + E@E

    Z = B@corr

    return Z[TARGET_PORT,TARGET_PORT]


# ============================================================
# FITNESS WITH HARD STOP
# ============================================================

def evaluate_config(config,method,threshold):

    peak = 0

    for f in range(N_FREQS):

        A = y[f].copy()

        for cap,port in config:
            A[port,port] += d[cap,f]

        try:

            if method=="numpy":
                val = inv_numpy(A)

            elif method=="solve":
                val = inv_solve(A)

            elif method=="sm":
                val = inv_sm(config,f)

            elif method=="iterative":
                val = inv_iterative(A,f)

        except:
            return 1e200

        peak = max(peak,abs(val))

        if peak > threshold:
            return peak

    return peak


# ============================================================
# PSO
# ============================================================

def run_pso(method,threshold,out_folder,run_id):

    logger = setup_logger(os.path.join(out_folder,f"run_{run_id}.log"))

    prev_best = None

    for n_caps in range(1,MAX_CAPS+1):

        DIM = 2*n_caps

        particles = np.random.rand(N_PARTICLES,DIM)
        velocities = np.zeros_like(particles)

        if prev_best is not None:

            warm = int(0.4*N_PARTICLES)

            for i in range(warm):

                particles[i,:2*(n_caps-1)] = prev_best
                particles[i,2*(n_caps-1):] = np.random.rand(2)

            rand_cap = int(np.floor(particles[0,n_caps-1]*(N_CAP_MODELS-1)))
            rand_port = int(np.floor(particles[0,2*n_caps-1]*(N_NODES-1)))

            logger.info(
                "init_n_caps=%d | seeded_previous_caps=%d | new_cap_random_init(cap=%d,port=%d)",
                n_caps,
                n_caps-1,
                rand_cap,
                rand_port
            )

        pbest = particles.copy()
        pbest_val = np.full(N_PARTICLES,np.inf)

        gbest = np.inf
        gbest_particle = None

        for it in range(N_ITERATIONS):

            for i in range(N_PARTICLES):

                models = np.floor(particles[i,:n_caps]*(N_CAP_MODELS-1)).astype(int)
                ports  = np.floor(particles[i,n_caps:]*(N_NODES-1)).astype(int)

                models,ports = resolve_adjacent_ports(models,ports)

                config = list(zip(models,ports))

                cost = evaluate_config(config,method,threshold)

                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]

                if cost < gbest:
                    gbest = cost
                    gbest_particle = particles[i].copy()

            if gbest <= threshold:
                break

            w = W_MAX - (W_MAX-W_MIN)*(it/N_ITERATIONS)

            velocities = w*velocities \
                + C1*np.random.rand(*particles.shape)*(pbest-particles) \
                + C2*np.random.rand(*particles.shape)*(gbest_particle-particles)

            particles += velocities
            particles = np.clip(particles,0,1)

        best_models = np.floor(gbest_particle[:n_caps]*(N_CAP_MODELS-1)).astype(int)
        best_ports  = np.floor(gbest_particle[n_caps:]*(N_NODES-1)).astype(int)

        best_models,best_ports = resolve_adjacent_ports(best_models,best_ports)

        placement = {int(port):int(cap) for cap,port in zip(best_models,best_ports)}

        logger.info(
            "n_caps=%d | iter=%d | minZ=%.6f | placement=%s",
            n_caps,
            it+1,
            gbest,
            placement
        )

        prev_best = gbest_particle[:2*n_caps]

        if gbest <= threshold:
            break


# ============================================================
# MAIN BENCHMARK
# ============================================================

for threshold in TARGETS:

    for method in METHODS:

        print(f"\nMETHOD {method} TARGET {threshold}")

        method_folder = os.path.join(ROOT_OUT,f"{method}_{threshold}")
        os.makedirs(method_folder,exist_ok=True)

        for run in range(1,NUM_RUNS+1):

            run_folder = os.path.join(method_folder,f"run_{run}")
            os.makedirs(run_folder,exist_ok=True)

            run_pso(method,threshold,run_folder,run)

print("All runs completed.")