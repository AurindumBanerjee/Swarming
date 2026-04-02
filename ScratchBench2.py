# nohup python ScratchBench2.py > scratchbench2.log 2>&1 &
# ps -u $USER | grep ScratchBench
# pkill -9 -f ScratchBench.py


import os
import time
import logging
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import shutil

# ============================================================
# GLOBAL CONFIG
# ============================================================

ROOT_OUT = "MinTime/ScratchBench4"
os.makedirs(ROOT_OUT, exist_ok=True)

NUM_RUNS = 20

TARGET_PORT = 0
TARGETS = [0.05,0.045,0.04,0.03]

MAX_CAPS = 20

N_PARTICLES = 50
N_ITERATIONS = 15

W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5

# METHODS = ["numpy","solve","sm","iterative","hybrid"]
METHODS = ["sm","iterative", "hybrid"]

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
    return np.linalg.inv(A)[TARGET_PORT,TARGET_PORT]

def inv_solve(A):
    e = np.zeros(N_NODES,dtype=complex)
    e[TARGET_PORT]=1
    return np.linalg.solve(A,e)[TARGET_PORT]

def inv_sm(config,f):

    Z = y_inv_base[f].copy()

    for cap,port in config:

        val = d[cap,f]

        u = np.zeros(N_NODES,dtype=np.complex128)
        v = np.zeros(N_NODES,dtype=np.complex128)

        u[port]=1
        v[port]=val

        denom = 1 + v@Z@u

        if abs(denom)<1e-12:
            raise np.linalg.LinAlgError

        Z -= np.outer(Z@u,v@Z)/denom

    return Z[TARGET_PORT,TARGET_PORT]

def inv_iterative(A,f):

    B = y_inv_base[f]
    E = A@B - np.eye(N_NODES)

    if np.linalg.norm(E,1)>=1:
        return np.linalg.inv(A)[TARGET_PORT,TARGET_PORT]

    corr = np.eye(N_NODES) - E + E@E
    Z = B@corr

    return Z[TARGET_PORT,TARGET_PORT]

# ✅ HYBRID (BEST PRACTICAL METHOD)
prev_inv_cache = [None]*N_FREQS

def inv_hybrid(A,f):

    global prev_inv_cache

    B = prev_inv_cache[f]

    if B is None:
        B = np.linalg.inv(A)
        prev_inv_cache[f] = B
        return B[TARGET_PORT,TARGET_PORT]

    E = A @ B - np.eye(N_NODES)

    if np.linalg.norm(E,1) >= 1:
        B = np.linalg.inv(A)
    else:
        B = B @ (np.eye(N_NODES) - E + E@E)

    prev_inv_cache[f] = B

    return B[TARGET_PORT,TARGET_PORT]

# ============================================================
# FITNESS (FIXED)
# ============================================================

def evaluate_config(config,method):

    peak = 0

    for f in range(N_FREQS):

        A = y[f].copy()

        for cap,port in config:
            A[port,port]+=d[cap,f]

        try:
            if method=="numpy":
                val=inv_numpy(A)
            elif method=="solve":
                val=inv_solve(A)
            elif method=="sm":
                val=inv_sm(config,f)
            elif method=="iterative":
                val=inv_iterative(A,f)
            elif method=="hybrid":
                val=inv_hybrid(A,f)
        except:
            return 1e200

        peak=max(peak,abs(val))

    return peak

# ============================================================
# PSO (EARLY STOP FIXED)
# ============================================================

def run_pso(method,threshold,out_folder,run_id):

    logger = setup_logger(os.path.join(out_folder,f"run_{run_id}.log"))

    run_caps=[]
    run_minZ=[]

    start_global = time.time()
    time_to_target = None
    caps_at_target = None

    prev_best=None

    for n_caps in range(1,MAX_CAPS+1):

        DIM=2*n_caps

        particles=np.random.rand(N_PARTICLES,DIM)
        velocities=np.zeros_like(particles)

        pbest=particles.copy()
        pbest_val=np.full(N_PARTICLES,np.inf)

        gbest=np.inf
        gbest_particle=None

        stop_flag=False

        for it in range(N_ITERATIONS):

            for i in range(N_PARTICLES):

                models=np.floor(particles[i,:n_caps]*(N_CAP_MODELS-1)).astype(int)
                ports=np.floor(particles[i,n_caps:]*(N_NODES-1)).astype(int)

                models,ports=resolve_adjacent_ports(models,ports)

                config=list(zip(models,ports))

                cost=evaluate_config(config,method)

                if cost<pbest_val[i]:
                    pbest_val[i]=cost
                    pbest[i]=particles[i]

                if cost<gbest:
                    gbest=cost
                    gbest_particle=particles[i].copy()

                # ✅ IMMEDIATE STOP
                if gbest<=threshold:
                    stop_flag=True
                    break

            if stop_flag:
                if time_to_target is None:
                    time_to_target = time.time() - start_global
                    caps_at_target = n_caps
                break

            w=W_MAX-(W_MAX-W_MIN)*(it/N_ITERATIONS)

            velocities = w*velocities \
                + C1*np.random.rand(*particles.shape)*(pbest-particles) \
                + C2*np.random.rand(*particles.shape)*(gbest_particle-particles)

            particles+=velocities
            particles=np.clip(particles,0,1)

        logger.info("n_caps=%d | minZ=%.6f", n_caps, gbest)

        run_caps.append(n_caps)
        run_minZ.append(gbest)

        if gbest<=threshold:
            break

    logger.info(
        "RESULT | time_to_target=%.4f | caps=%s",
        time_to_target if time_to_target else -1,
        caps_at_target if caps_at_target else -1
    )

    # PLOT
    if run_caps:
        x,y=zip(*sorted(zip(run_caps,run_minZ)))

        plt.figure()
        plt.plot(x,y,marker='o')
        plt.axhline(y=threshold, linestyle='--')  # ✅ threshold line
        plt.xlabel("Decaps")
        plt.ylabel("MinZ")
        plt.title(f"{method} T{threshold} Run {run_id}")
        plt.grid()
        plt.savefig(os.path.join(out_folder,"run_plot.png"),dpi=300)
        plt.close()

# ============================================================
# MAIN
# ============================================================

for threshold in TARGETS:

    for method in METHODS:

        print(f"\nMETHOD {method} TARGET {threshold}")

        method_folder=os.path.join(ROOT_OUT,f"{method}_{threshold}")
        os.makedirs(method_folder,exist_ok=True)

        for run in range(1,NUM_RUNS+1):

            run_folder=os.path.join(method_folder,f"run_{run}")
            os.makedirs(run_folder,exist_ok=True)

            run_pso(method,threshold,run_folder,run)

print("All runs completed.")

# ============================================================
# GLOBAL ANALYSIS
# ============================================================

records=[]

for root,_,files in os.walk(ROOT_OUT):

    for file in files:

        if not file.endswith(".log"):
            continue

        path=os.path.join(root,file)
        parts=root.split(os.sep)
        method,threshold=parts[-2].split("_")

        with open(path) as f:
            for line in f:

                if "RESULT" in line:
                    m=re.search(r"time_to_target=([0-9\.\-]+).*caps=([0-9\-]+)",line)
                    if m:
                        records.append({
                            "method":method,
                            "threshold":float(threshold),
                            "time":float(m.group(1)),
                            "caps":int(m.group(2)),
                            "folder":root
                        })

# ============================================================
# TOP 10 FASTEST
# ============================================================

valid=[r for r in records if r["time"]>0]
top10=sorted(valid,key=lambda x:x["time"])[:10]

with open(os.path.join(ROOT_OUT,"top10_fastest.txt"),"w") as f:

    for i,r in enumerate(top10,1):
        f.write(f"{i}. {r}\n")

        src=os.path.join(r["folder"],"run_plot.png")
        if os.path.exists(src):
            shutil.copy(src,os.path.join(ROOT_OUT,f"top{i}.png"))

# ============================================================
# STATISTICS
# ============================================================

with open(os.path.join(ROOT_OUT,"statistics.txt"),"w") as f:

    for threshold in TARGETS:

        f.write(f"\n=== Threshold {threshold} ===\n")

        subset=[r for r in records if r["threshold"]==threshold and r["time"]>0]

        if subset:
            best=min(subset,key=lambda x:x["time"])
            f.write(f"BEST: {best}\n")

print("Done.")