#!/bin/bash
# After run02 (python pid 141103) ends: run formalise_server.sh, which creates GO_RUN3 only on success (the gated sequencer then starts run03).
B=/DATA/Aurindum/Swarming
R=$B/MinTime/Runs
echo "[watch] waiting for run02 pid 141103 ($(date))" >> $R/formalise_watch.log
while kill -0 141103 2>/dev/null; do sleep 30; done
echo "[watch] run02 ended $(date)" >> $R/formalise_watch.log
for i in $(seq 1 30); do grep -q "holding" $R/sequence_03_05_gated.log 2>/dev/null && break; sleep 20; done   # let the sequencer write run02 END into its README first
bash $R/formalise_server.sh $B 141103 >> $R/formalise_watch.log 2>&1
echo "[watch] formalise exit code $? ($(date))" >> $R/formalise_watch.log
