#!/bin/bash
# Run run07_gpubench2: parts execute strictly one after the other. cwd must be /DATA/Aurindum/Swarming (ROOT_OUT is relative).
RUN=/DATA/Aurindum/Swarming/MinTime/Runs/run07_gpubench2
cd /DATA/Aurindum/Swarming || exit 1
source /DATA/Aurindum/conda/etc/profile.d/conda.sh
conda activate gpuwork
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "Part gpubench2 START: $(stamp)  load: $(cut -d" " -f1-3 /proc/loadavg)" >> "$RUN/README.txt"
taskset -c 16-23 python -u "$RUN/code/gpubench2__run07_gpubench2.py" --targets 0.05,0.045,0.04 --runs 20 > "$RUN/logs/gpubench2_console.log" 2>&1
echo "Part gpubench2 END:   $(stamp)  exit code $?" >> "$RUN/README.txt"
