#!/bin/bash
# Run run06_warm100_jitter0_library: parts execute strictly one after the other. cwd must be /DATA/Aurindum/Swarming (ROOT_OUT is relative).
RUN=/DATA/Aurindum/Swarming/MinTime/Runs/run06_warm100_jitter0_library
cd /DATA/Aurindum/Swarming || exit 1
source /DATA/Aurindum/conda/etc/profile.d/conda.sh
conda activate work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "Part library START: $(stamp)  load: $(cut -d" " -f1-3 /proc/loadavg)" >> "$RUN/README.txt"
python -u "$RUN/code/library__run06_warm100_jitter0_library.py" > "$RUN/logs/library_console.log" 2>&1
echo "Part library END:   $(stamp)  exit code $?" >> "$RUN/README.txt"
