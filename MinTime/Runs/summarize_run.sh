#!/bin/bash
# summarize_run.sh <out_dir> <root_dir> [<root_dir> ...]
# Runs the output summariser (and, when a pure_python partner is present, the speedup plots) on one or more result roots.
# Each root is a directory that contains <method>__THR__<threshold>/run_N/run_N.log folders. Several roots are merged through a
# temporary symlink farm (e.g. run02 library + run04 pure_python), nothing is copied or modified.
#   <out_dir>/summary/{runs.csv,summary.csv,summary.txt}     always
#   <out_dir>/plots/{ieee,matrix_speedup}/                   only if a pure_python__THR__* folder exists (the speedup S is relative to pure Python)
# Light single-core job; run it with `nice`. Needs the conda env "work" (pandas, scipy, matplotlib).
set -u
B=${SWARMING_ROOT:-/DATA/Aurindum/Swarming}
OUT=${1:?out_dir}; shift
[ $# -ge 1 ] || { echo "need at least one root"; exit 1; }
source /DATA/Aurindum/conda/etc/profile.d/conda.sh && conda activate work
T=$(mktemp -d /tmp/aur_sum.XXXXXX); trap 'rm -rf "$T"' EXIT
for root in "$@"; do
  for g in "$root"/*__THR__*; do [ -d "$g" ] && ln -s "$(readlink -f "$g")" "$T/$(basename "$g")"; done
done
[ -n "$(ls -A "$T")" ] || { echo "no <method>__THR__<x> folders under: $*"; exit 1; }
cd "$B" || exit 1
mkdir -p "$OUT"
nice -n 19 python summarize_baseline.py --root "$T" --output "$OUT/summary" || exit 1
if ls -d "$T"/pure_python__THR__* >/dev/null 2>&1; then
  nice -n 19 python plot_ieee_figures.py --runs "$OUT/summary/runs.csv" --out "$OUT/plots/ieee" || echo "WARNING: plot_ieee_figures failed"
  nice -n 19 python plot_matrix_evaluation_speedup.py --input-dir "$OUT/summary" --output-dir "$OUT/plots/matrix_speedup" || echo "WARNING: plot_matrix_evaluation_speedup failed"
else
  mkdir -p "$OUT/plots"
  echo "No pure_python runs among: $*  -- the speedup plots are relative to pure Python, so none were drawn. Re-run summarize_run.sh with the pure_python root (e.g. run04/run05) to get them." > "$OUT/plots/NO_PLOTS.txt"
fi
echo "summaries -> $OUT/summary ; plots -> $OUT/plots"
