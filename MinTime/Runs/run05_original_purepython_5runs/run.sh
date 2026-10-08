#!/bin/bash
# Run run05_original_purepython_5runs: parts execute strictly one after the other. cwd must be /DATA/Aurindum/Swarming (ROOT_OUT is relative).
RUN=/DATA/Aurindum/Swarming/MinTime/Runs/run05_original_purepython_5runs
cd /DATA/Aurindum/Swarming || exit 1
source /DATA/Aurindum/conda/etc/profile.d/conda.sh
conda activate work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "Part pure_python START: $(stamp)  load: $(cut -d" " -f1-3 /proc/loadavg)" >> "$RUN/README.txt"
python -u "$RUN/code/pure_python__run05_original_purepython_5runs.py" > "$RUN/logs/pure_python_console.log" 2>&1
echo "Part pure_python END:   $(stamp)  exit code $?" >> "$RUN/README.txt"
