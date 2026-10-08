#!/bin/bash
# Rerun with the NAIVE warm start (WARM_START_JITTER = 0.0): Part 1 (library) then Part 2 (pure_python), sequential.
D=/DATA/Aurindum/Swarming
R=$D/MinTime/ScratchBenchRerun_20261006
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
cd "$D" || exit 1
source /DATA/Aurindum/conda/etc/profile.d/conda.sh
conda activate work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[naive] Part 1 (library, jitter 0.0) START: $(stamp)  load: $(cut -d' ' -f1-3 /proc/loadavg)" >> "$R/README.txt"
python -u "$R/ScratchBench_library_naive.py" > "$R/library_console.log" 2>&1
echo "[naive] Part 1 END:   $(stamp)  exit code $?" >> "$R/README.txt"

echo "[naive] Part 2 (pure_python, jitter 0.0) START: $(stamp)  load: $(cut -d' ' -f1-3 /proc/loadavg)" >> "$R/README.txt"
python -u "$R/ScratchBench2_purepython_naive.py" > "$R/pure_python_console.log" 2>&1
echo "[naive] Part 2 END:   $(stamp)  exit code $?" >> "$R/README.txt"
