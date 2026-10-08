#!/bin/bash
# Replaces run_rerun.sh's tail end: wait for Part 1 (library) -> warm-start comparisons -> Part 2 (pure_python).
# Nothing overlaps Part 1 or Part 2; the two comparison scripts run together (16 threads each) between them.
# Usage: run_chain.sh <pid of the running Part 1 python>
P1=$1
D=/DATA/Aurindum/Swarming
G=/DATA/Aurindum/GPUSwarm
R=$D/MinTime/ScratchBenchRerun_20261006
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
source /DATA/Aurindum/conda/etc/profile.d/conda.sh
conda activate work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[chain] waiting for Part 1 python pid $P1 ($(stamp))" >> "$R/chain.log"
while kill -0 "$P1" 2>/dev/null; do sleep 30; done
echo "Part 1 END:   $(stamp)  (original driver replaced by run_chain.sh; exit code not captured -- see library_console.log)" >> "$R/README.txt"

# ---- warm-start comparisons (GPUSwarm/Comparisons), after Part 1 and before Part 2 ----
C=$G/Comparisons
cd "$C" || exit 1
for f in warmstart_results.jsonl warmstart_arms_results.jsonl; do      # scripts APPEND: never mix old rows in
  [ -e "$f" ] && mv "$f" "$f.old.$(date +%Y%m%d%H%M%S)"
done
echo "Comparisons --estimate START: $(stamp)  load: $(cut -d' ' -f1-3 /proc/loadavg)" >> "$R/README.txt"
{ echo "== warmstart_thresholds.py --estimate (THREADS=16)"; THREADS=16 NUM_RUNS=10 python -u warmstart_thresholds.py --estimate
  echo; echo "== warmstart_arms.py --estimate (THREADS=16)"; THREADS=16 NUM_RUNS=5 python -u warmstart_arms.py --estimate
} > "$R/comparisons_estimates.txt" 2>&1
echo "Comparisons --estimate END:   $(stamp)  (output: comparisons_estimates.txt)" >> "$R/README.txt"

echo "Comparisons runs START: $(stamp)  (thresholds NUM_RUNS=10, arms NUM_RUNS=5, THREADS=16 each)" >> "$R/README.txt"
THREADS=16 NUM_RUNS=10 python -u warmstart_thresholds.py > thr.out 2>&1 &
PT=$!
THREADS=16 NUM_RUNS=5  python -u warmstart_arms.py       > arms.out 2>&1 &
PA=$!
wait $PT; rct=$?
wait $PA; rca=$?
echo "Comparisons runs END:   $(stamp)  exit codes thresholds=$rct arms=$rca" >> "$R/README.txt"

# ---- Part 2 (pure_python), exactly as in run_rerun.sh ----
cd "$D" || exit 1
echo "Part 2 (pure_python) START: $(stamp)  load: $(cut -d' ' -f1-3 /proc/loadavg)" >> "$R/README.txt"
python -u "$R/ScratchBench2_purepython.py" > "$R/pure_python_console.log" 2>&1
rc2=$?
echo "Part 2 END:   $(stamp)  exit code $rc2" >> "$R/README.txt"
