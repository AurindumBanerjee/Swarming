#!/bin/bash
# After run02 library (python pid 141103) exits: HOLD until Runs/GO_RUN3 exists (Swarming formalisation done),
# then run03 library -> run04 (naive pure_python, 5 runs/threshold) -> run05 (original pure_python, 5 runs/threshold). Strictly sequential.
B=/DATA/Aurindum/Swarming
R=$B/MinTime/Runs
RR=$B/MinTime/ScratchBenchRerun_20261006
log=$R/sequence_03_05_gated.log
echo "[seq] waiting for run02 library pid 141103 ($(date))" >> $log
while kill -0 141103 2>/dev/null; do sleep 60; done
END_TS=$(date "+%Y-%m-%d %H:%M:%S %Z")
echo "[seq] run02 library part ended $END_TS" >> $log
echo "[naive/run02] Part 1 (library) END: $END_TS  (launcher replaced by Runs/sequence_03_05_gated.sh; exit code not captured -- see library_console.log; run02 pure_python is now run04)" >> $RR/README.txt
echo "[seq] holding: run03 starts only when $R/GO_RUN3 exists" >> $log
until [ -e $R/GO_RUN3 ]; do sleep 60; done
echo "[seq] GO_RUN3 found $(date)" >> $log
for r in run03_original_code_library run04_naive_purepython_5runs run05_original_purepython_5runs; do
  echo "[seq] starting $r $(date)" >> $log
  cd $B && bash $R/$r/run.sh > $R/$r/logs/driver.log 2>&1
  echo "[seq] $r finished $(date)" >> $log
done
echo "[seq] all done $(date)" >> $log
