#!/bin/bash
# Starts run03 only after the run02 chain (run_chain2.sh, pid 141088: library + pure_python) has exited.
echo "[queue] waiting for pid 141088 ($(date))" >> /DATA/Aurindum/Swarming/MinTime/Runs/run03_original_code_repro/logs/queue.log
while kill -0 141088 2>/dev/null; do sleep 60; done
echo "[queue] run02 chain ended $(date); starting run03" >> /DATA/Aurindum/Swarming/MinTime/Runs/run03_original_code_repro/logs/queue.log
cd /DATA/Aurindum/Swarming && bash /DATA/Aurindum/Swarming/MinTime/Runs/run03_original_code_repro/run.sh
