#!/bin/bash
# Replaces sequence_03_05_gated.sh (cancelled while run03 was running): after run03's run.sh (pid 70140) ends ->
# run06 (100% warm, library) -> run04 (naive pure_python, 5 runs) -> run05 (original pure_python, 5 runs). Strictly sequential.
B=/DATA/Aurindum/Swarming
R=$B/MinTime/Runs
log=$R/sequence_06_04_05.log
echo "[seq2] waiting for run03 run.sh pid 70140 ($(date))" >> $log
while kill -0 70140 2>/dev/null; do sleep 60; done
echo "[seq2] run03 finished $(date)" >> $log
for r in run06_warm100_jitter0_library run04_naive_purepython_5runs run05_original_purepython_5runs; do
  echo "[seq2] starting $r $(date)" >> $log
  cd $B && bash $R/$r/run.sh > $R/$r/logs/driver.log 2>&1
  echo "[seq2] $r finished $(date)" >> $log
done
echo "[seq2] all done $(date)" >> $log
