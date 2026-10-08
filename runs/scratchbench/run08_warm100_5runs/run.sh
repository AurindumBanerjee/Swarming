#!/bin/bash
# Run scratchbench/run08_warm100_5runs. Launch from anywhere; the script runs from the repository root.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN/../../.." || exit 1
source $(conda info --base 2>/dev/null || echo /DATA/Aurindum/conda)/etc/profile.d/conda.sh
conda activate work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RUN_DIR="$RUN" RUN_ID="run08_warm100_5runs"
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "START: $(stamp)  host: $(hostname)  load: $(cut -d" " -f1-3 /proc/loadavg 2>/dev/null)" >> "$RUN/README.txt"
taskset -c 0-7 python -u "$RUN/code/library__run06_warm100_jitter0_library__run08_warm100_5runs.py"  > "$RUN/logs/console.log" 2>&1
echo "END:   $(stamp)  exit code $?" >> "$RUN/README.txt"
