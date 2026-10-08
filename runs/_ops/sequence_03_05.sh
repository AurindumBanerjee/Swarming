#!/bin/bash
# After run02's library part (python pid 141103) exits:  run03 library -> run04 (naive pure_python) -> run05 (original pure_python).
# Strictly sequential; nothing overlaps (timings).
B=/DATA/Aurindum/Swarming
R=$B/MinTime/Runs
RR=$B/MinTime/ScratchBenchRerun_20261006
log=$R/sequence_03_05.log
echo "[seq] waiting for run02 library pid 141103 ($(date))" >> $log
while kill -0 141103 2>/dev/null; do sleep 60; done
echo "[seq] run02 library part ended $(date)" >> $log
echo "[naive/run02] Part 1 (library) END: $(date '+%Y-%m-%d %H:%M:%S %Z')  (chain script replaced by Runs/sequence_03_05.sh; exit code not captured -- see library_console.log; run02 pure_python is now run04, 5 runs/threshold)" >> $RR/README.txt
for r in run03_original_code_library run04_naive_purepython_5runs run05_original_purepython_5runs; do
  echo "[seq] starting $r $(date)" >> $log
  cd $B && bash $R/$r/run.sh > $R/$r/logs/driver.log 2>&1
  echo "[seq] $r finished $(date)" >> $log
done
echo "[seq] all done $(date)" >> $log
