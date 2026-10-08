#!/bin/bash
# Sequential rerun: Part 1 (library methods) then Part 2 (pure_python). Never overlapping.
# Launch:  cd /DATA/Aurindum/Swarming && nohup setsid bash MinTime/ScratchBenchRerun_20261006/run_rerun.sh > MinTime/ScratchBenchRerun_20261006/driver.log 2>&1 < /dev/null &
D=/DATA/Aurindum/Swarming
R=$D/MinTime/ScratchBenchRerun_20261006
cd "$D" || exit 1
source /DATA/Aurindum/conda/etc/profile.d/conda.sh
conda activate work
# identical BLAS pinning for both parts (the scripts themselves set no thread limits)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "Part 1 (library: numpy solve sm iterative) START: $(stamp)  load: $(cut -d' ' -f1-3 /proc/loadavg)" >> "$R/README.txt"
python -u "$R/ScratchBench_library.py" > "$R/library_console.log" 2>&1
rc1=$?
echo "Part 1 END:   $(stamp)  exit code $rc1" >> "$R/README.txt"

echo "Part 2 (pure_python) START: $(stamp)  load: $(cut -d' ' -f1-3 /proc/loadavg)" >> "$R/README.txt"
python -u "$R/ScratchBench2_purepython.py" > "$R/pure_python_console.log" 2>&1
rc2=$?
echo "Part 2 END:   $(stamp)  exit code $rc2" >> "$R/README.txt"
