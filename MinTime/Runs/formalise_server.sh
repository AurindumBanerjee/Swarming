#!/bin/bash
# Formalise the Swarming run structure on the server (conservative layout) and commit it.
# Usage: formalise_server.sh <ROOT> <run02_pid>     (ROOT = /DATA/Aurindum/Swarming; staged inputs in ROOT/MinTime/Runs/_formalise_stage)
# Safe to read top to bottom: only `mv` inside one filesystem, copies of staged files, and one git commit. Nothing is deleted
# unless it is an exact duplicate (md5-checked) or an empty directory. Creates Runs/GO_RUN3 only if everything succeeded.
set -euo pipefail
ROOT=${1:?ROOT}
PID=${2:?pid of run02 python}
R=$ROOT/MinTime/Runs
RR=$ROOT/MinTime/ScratchBenchRerun_20261006
ST=$R/_formalise_stage
LOG=$R/formalise.log
exec > >(tee -a "$LOG") 2>&1
echo "=== formalise start $(date) ROOT=$ROOT"

if kill -0 "$PID" 2>/dev/null; then echo "run02 python $PID still running -- refusing"; exit 1; fi
[ -d "$ST" ] || { echo "no staging dir $ST"; exit 1; }
[ -d "$RR" ] || { echo "no $RR"; exit 1; }

same() { [ -f "$1" ] && [ -f "$2" ] && [ "$(md5sum < "$1")" = "$(md5sum < "$2")" ]; }

# ---- 1. real folders for run01 / run02 (symlinks are replaced; nothing is copied, files are moved) ----
for l in run01_hybrid_jitter0.02_partial run02_naive_jitter0; do
  if [ -L "$R/$l" ]; then rm "$R/$l"; fi
  [ -e "$R/$l" ] && { echo "$R/$l exists and is not a symlink -- refusing"; exit 1; }
done
P=$RR/previous_jitter0.02
R1=$R/run01_hybrid_jitter0.02_partial
R2=$R/run02_naive_jitter0
mkdir -p $R1/code $R1/output $R1/logs $R2/code $R2/output $R2/logs $R2/launchers

# run01 = what was archived in previous_jitter0.02/
mv "$P/library"               "$R1/output/library"
mv "$P/library_console.log"   "$R1/logs/library_console.log"
for f in ScratchBench_library.py ScratchBench2_purepython.py; do
  if same "$P/$f" "$RR/$f"; then mv "$RR/$f" "$R1/code/$f"; rm "$P/$f"; else mv "$P/$f" "$R1/code/$f"; fi
done
rmdir "$P"

# run02 = the naive-jitter-0 run in the rerun folder
mv "$RR/library"                         "$R2/output/library"
mv "$RR/library_console.log"             "$R2/logs/library_console.log"
for f in ScratchBench_library_naive.py ScratchBench2_purepython_naive.py; do mv "$RR/$f" "$R2/code/$f"; done
for f in run_rerun.sh run_chain.sh run_chain2.sh; do [ -e "$RR/$f" ] && mv "$RR/$f" "$R2/launchers/$f"; done
for f in chain.log chain.out chain2.out driver.log; do [ -e "$RR/$f" ] && mv "$RR/$f" "$R2/logs/$f"; done
mv "$RR/README.txt" "$R2/README_original_rerun.txt"
if [ -d "$RR/pure_python" ]; then mv "$RR/pure_python" "$R2/output/pure_python"; fi
echo "leftovers in $RR: $(ls -A "$RR" | tr '\n' ' ')"
rmdir "$RR" 2>/dev/null && echo "removed empty $RR" || echo "kept $RR (not empty)"

cat > $R1/README.txt <<EOF
RUN run01_hybrid_jitter0.02_partial  (pre-convention run; folder organised $(date '+%Y-%m-%d') after run02 ended)
Current code (FIX 11-13): hybrid warm start = 40% warm block (particle 0 exact elite, others jittered by 0.02), 60% random.
Library part only; thresholds 0.05/0.045/0.04, 20 runs. Stopped (killed) partway on 2026-10-06 ~21:00 IST after numpy and solve
at 0.05 (20/20 each) and sm at 0.05 (18/20 results). Code: code/ScratchBench_library.py (md5 $(md5sum $R1/code/ScratchBench_library.py | cut -c1-32)),
code/ScratchBench2_purepython.py was never run. Output: output/library/, console: logs/library_console.log.
Original timeline and environment: README_original_rerun.txt in ../run02_naive_jitter0/.
Results: at 0.05 this code needed 11.55 caps on average (old code 5.25) -- see the warm-start comparison in GPUSwarm/Comparisons.
EOF
cat > $R2/README.txt <<EOF
RUN run02_naive_jitter0  (pre-convention run; folder organised $(date '+%Y-%m-%d') after its library part ended)
Current code with WARM_START_JITTER = 0.0 (naive warm block: 20 identical warm particles + 30 random, no elite jitter).
Library part (numpy, solve, sm, iterative), thresholds 0.05/0.045/0.04, 20 runs, BLAS pinned to 1 thread, perf_counter timer.
Code: code/ScratchBench_library_naive.py (md5 $(md5sum $R2/code/ScratchBench_library_naive.py | cut -c1-32)); code/ScratchBench2_purepython_naive.py
was NOT run here (its pure_python part became run04_naive_purepython_5runs). Output: output/library/, logs in logs/, launchers in launchers/.
Full original timeline, environment, md5 sums and notes: README_original_rerun.txt (written as the run went; paths in it refer to the
old location MinTime/ScratchBenchRerun_20261006/, which no longer exists).
EOF

# ---- 2. INDEX.md rows for run01 / run02 ----
python3 - "$R/INDEX.md" <<'PYEOF'
import re, sys
p = sys.argv[1]
out = []
for line in open(p).read().split("\n"):
    if line.startswith("| run01_hybrid_jitter0.02_partial"):
        line = re.sub(r"\|[^|]*\|$", "| Runs/run01_hybrid_jitter0.02_partial/ |", line)
    if line.startswith("| run02_naive_jitter0"):
        line = line.replace("RUNNING (pre-convention run)", "library part finished; its pure_python part became run04 (pre-convention run)")
        line = re.sub(r"\|[^|]*\|$", "| Runs/run02_naive_jitter0/ |", line)
    out.append(line)
open(p, "w").write("\n".join(out))
PYEOF

# ---- 3. apply staged repo files ----
cp "$ST/.gitignore"  "$ROOT/.gitignore"
cp "$ST/README.md"   "$ROOT/README.md"
for f in ScratchBench.py ScratchBench2.py summarize_baseline.py plot_ieee_figures.py plot_matrix_evaluation_speedup.py; do
  cp "$ST/$f" "$ROOT/$f"
done
# ---- 3b. summaries for the finished older runs (run00's were produced earlier; run03-05 are summarised when they finish) ----
SWARMING_ROOT="$ROOT" bash "$R/summarize_run.sh" "$R1" "$R1/output/library" || echo "WARNING: run01 summary failed"
SWARMING_ROOT="$ROOT" bash "$R/summarize_run.sh" "$R2" "$R2/output/library" || echo "WARNING: run02 summary failed"
if [ -d "$ST/analysis" ]; then      # analysis products that existed only on the laptop -> belong to run00
  [ -d "$R/run00_original_sep2026" ] && [ ! -e "$R/run00_original_sep2026/analysis" ] || { echo "cannot place analysis/"; exit 1; }
  mv "$ST/analysis" "$R/run00_original_sep2026/analysis"
fi
rm -r "$ST"

# ---- 4. commit ----
cd "$ROOT"
git add -A
echo "--- git status (short) ---"; git status --short | cut -c1-120 | head -60
echo "files staged: $(git diff --cached --name-only | wc -l)"
if git diff --cached --name-only | grep -E "/(output|logs)/" ; then echo "ERROR: output/logs files staged"; git reset -q; exit 1; fi
if git diff --cached --name-only | grep -vE '^(\.gitignore|README\.md|ScratchBench2?\.py|summarize_baseline\.py|plot_ieee_figures\.py|plot_matrix_evaluation_speedup\.py|MinTime/Runs/)' ; then
  echo "ERROR: files outside the allow-list are staged (listed above)"; git reset -q; exit 1; fi
git commit -q -m "Formalise ScratchBench run structure (MinTime/Runs); latest ScratchBench scripts (FIX 11-13)

- MinTime/Runs/ is the tracked home of every run: run00..run05 code, manifests, READMEs, INDEX.md, new_run.py (outputs and logs ignored).
- run01/run02 moved there from MinTime/ScratchBenchRerun_20261006.
- ScratchBench.py / ScratchBench2.py updated to the FIX 11-13 versions (the original Sep-2026 versions are kept in git history and in Runs/run00_original_sep2026/code).
- README.md documents the layout and the runs convention; .gitignore tracks MinTime/Runs only.
- Adds summarize_baseline.py, plot_ieee_figures.py, plot_matrix_evaluation_speedup.py (previously only on the laptop).
- Adds per-run summaries (summary/) and speedup plots (plots/) for run00-run02 (plots only where a pure_python partner exists).
- Adds run00 analysis products (summaries, speedup figures, plotter.py) that existed only on the laptop.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log --oneline -1
touch "$R/GO_RUN3"
echo "=== formalise done $(date); GO_RUN3 created"
