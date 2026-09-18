# python ScratchBench_plot.py
#
# ============================================================
# STANDALONE PLOTTING / ANALYSIS SCRIPT
# ============================================================
# Run this after ScratchBench_sim.py has produced pure_python data. This
# script does NOT run any PSO -- it reads back whatever result data exists
# under ROOT_OUT and builds the comparison tables/figures. Two on-disk
# formats are merged automatically:
#
#   1. LEGACY per-run .log files -- produced by the *original*
#      ScratchBench.py (numpy/solve/sm/iterative results you already
#      have sitting in ROOT_OUT). Each run's RESULT line is parsed with
#      a regex: this is the same "RESULT | time_to_target=... | caps=...
#      | best_minZ=..." line both the original script and
#      ScratchBench_sim.py write via setup_logger(), so one parser
#      handles logs from either script.
#   2. STRUCTURED results_long.jsonl -- produced by ScratchBench_sim.py
#      (currently just pure_python). Richer (includes success flag,
#      n_fitness_evals, etc.) and takes precedence over a legacy log
#      entry for the same (method, threshold, run_id) if both exist.
#
# This means: you do NOT need to re-run numpy/solve/sm/iterative through
# the new sim script -- your existing ScratchBench.py results are picked
# up automatically, and only pure_python needs to come from
# ScratchBench_sim.py.
#
# CAVEAT: the original ScratchBench.py never seeded NumPy's global RNG,
# so numpy/solve/sm/iterative runs are NOT seed-paired with the new
# (seeded) pure_python runs -- pairing by run_id still happens (matching
# the paper's methodology of averaging per-run ratios), but for the
# legacy methods it pairs runs that used independent, not identical,
# PSO draws. This mainly increases variance in the reported CIs; it does
# not bias the mean.
#
# The primary timing metric used throughout is `time_to_target_s` (time
# to reach the threshold, on successful runs only) rather than total
# wall-clock time, because that's the only timing figure the legacy logs
# contain -- and it's what Table I actually reports.
#
# Produces, under ROOT_OUT:
#   results_long.csv          tidy one-row-per-run table (both sources merged)
#   summary_table.csv         mean +/- 95% CI per (threshold, method)
#   paired_speedup_long.csv   per-run speedup vs BASELINE_METHOD
#   speedup_matrix.csv        method x threshold pivot
#   config.json               what was actually found in the data
#   top10_fastest.txt / statistics.txt
#   figs/speedup_heatmap.png              single-row, avg speedup vs baseline
#   figs/runtime_reduction_vs_threshold.png   absolute time-to-target (log scale)
#   figs/best_minz_vs_threshold.png       avg best |Z11| vs threshold
#
# ROOT_OUT below MUST match the ROOT_OUT used by ScratchBench_sim.py /
# the original ScratchBench.py.
# ============================================================

import os
import re
import time
import json
import math
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# CONFIG
# ============================================================

ROOT_OUT = "MinTime/ScratchBench5"
RESULTS_JSONL_PATH = os.path.join(ROOT_OUT, "results_long.jsonl")

# Full expected 5-way roster (internal keys). NOT a filter -- every method
# actually found in the data gets analyzed regardless of whether it's here;
# this list only drives labeling/ordering and the "still missing" report.
METHODS = ["numpy", "solve", "sm", "iterative", "pure_python"]
BASELINE_METHOD = "pure_python"   # every speedup number is vs. this method

# Candidate method-name prefixes for legacy folder names, longest first so
# "pure_python_..." isn't mis-split by a shorter accidental prefix.
_LEGACY_METHOD_CANDIDATES = sorted(METHODS + ["hybrid"], key=len, reverse=True)

LEGACY_RESULT_RE = re.compile(
    r"time_to_target=([0-9.\-]+)\s*\|\s*caps=([0-9\-]+)\s*\|\s*best_minZ=([0-9.\-]+)"
)

# ============================================================
# DATA LOADING (merges legacy .log files + structured .jsonl)
# ============================================================

def load_jsonl_records(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_legacy_log_records(root_out):
    """
    Walk root_out/{method}_{threshold}/run_{id}/run_{id}.log and parse the
    RESULT line. Works for logs written by either the original
    ScratchBench.py or ScratchBench_sim.py -- both use the same
    setup_logger() + "RESULT | time_to_target=... | caps=... |
    best_minZ=..." line, so the regex matches either (a trailing
    "| total_wall_time=..." field, present only in the new format, is
    simply ignored since the regex has no end anchor).
    """
    records = []
    if not os.path.isdir(root_out):
        return records

    for entry in sorted(os.listdir(root_out)):
        method_folder = os.path.join(root_out, entry)
        if not os.path.isdir(method_folder):
            continue

        method_name = None
        threshold = None
        for cand in _LEGACY_METHOD_CANDIDATES:
            prefix = cand + "_"
            if entry.startswith(prefix):
                try:
                    threshold = float(entry[len(prefix):])
                except ValueError:
                    continue
                method_name = cand
                break
        if method_name is None:
            continue

        for run_dir in sorted(os.listdir(method_folder)):
            run_path = os.path.join(method_folder, run_dir)
            if not os.path.isdir(run_path):
                continue
            m = re.match(r"run_(\d+)$", run_dir)
            if not m:
                continue
            run_id = int(m.group(1))
            log_path = os.path.join(run_path, f"run_{run_id}.log")
            if not os.path.exists(log_path):
                continue

            with open(log_path) as f:
                text = f.read()
            match = LEGACY_RESULT_RE.search(text)
            if not match:
                continue

            ttt = float(match.group(1))
            caps = int(match.group(2))
            minz = float(match.group(3))

            records.append({
                "threshold": threshold,
                "method": method_name,
                "run_id": run_id,
                "seed": None,  # original script never seeded the RNG
                "time_to_target_s": ttt if ttt > 0 else None,
                "caps_at_target": caps if caps > 0 else None,
                "best_minz_ohm": minz,
                "success": ttt > 0,
                "source": "legacy_log",
                "folder": run_path,
            })

    return records


def load_merged_records(root_out, jsonl_path):
    """
    Structured JSONL entries take precedence; legacy log-parsed entries
    fill in only the (method, threshold, run_id) combinations that have
    no JSONL entry at all. This lets old numpy/solve/sm/iterative runs
    (log-only) sit alongside new pure_python runs (JSONL) without
    double-counting anything that happens to exist in both.
    """
    jsonl_records = load_jsonl_records(jsonl_path)
    for r in jsonl_records:
        r.setdefault("source", "jsonl")

    seen_keys = {(r["method"], r["threshold"], r["run_id"]) for r in jsonl_records}

    legacy_records = load_legacy_log_records(root_out)
    legacy_only = [r for r in legacy_records
                   if (r["method"], r["threshold"], r["run_id"]) not in seen_keys]

    return jsonl_records + legacy_only


# ============================================================
# LOAD + REPORT
# ============================================================

records = load_merged_records(ROOT_OUT, RESULTS_JSONL_PATH)
df = pd.DataFrame(records)

if df.empty:
    print(f"[WARN] No results found under {ROOT_OUT} (checked "
          f"{RESULTS_JSONL_PATH} and legacy .log files) -- nothing to plot.")
else:
    n_legacy = int((df["source"] == "legacy_log").sum())
    n_jsonl = int((df["source"] == "jsonl").sum())
    print(f"Loaded {len(df)} run records: {n_jsonl} from results_long.jsonl, "
          f"{n_legacy} from legacy .log files.")

    present_methods = sorted(df["method"].unique())
    missing_methods = [m for m in METHODS if m not in present_methods]
    print(f"Methods present in data: {present_methods}")
    if missing_methods:
        print(f"[NOTE] Expected but not yet found: {missing_methods}")
    if BASELINE_METHOD not in present_methods:
        print(f"[NOTE] Baseline method {BASELINE_METHOD!r} not in the data yet "
              f"-- speedup-based outputs will be skipped until it's present.")

    results_long_csv = os.path.join(ROOT_OUT, "results_long.csv")
    df.to_csv(results_long_csv, index=False)

    # ---- paired speedup vs BASELINE_METHOD, matched by (threshold, run_id),
    #      using time_to_target_s (only metric both data sources have) on
    #      successful runs of both sides ----
    pivot_time = df.pivot_table(index=["threshold", "run_id"], columns="method",
                                 values="time_to_target_s")
    speedup_rows = []
    if BASELINE_METHOD in pivot_time.columns:
        for method in METHODS:
            if method == BASELINE_METHOD or method not in pivot_time.columns:
                continue
            paired = pivot_time[BASELINE_METHOD] / pivot_time[method]
            tmp = paired.reset_index()
            tmp.columns = ["threshold", "run_id", "paired_speedup"]
            tmp["method"] = method
            tmp = tmp.dropna(subset=["paired_speedup"])
            speedup_rows.append(tmp)
    speedup_long = pd.concat(speedup_rows, ignore_index=True) if speedup_rows else pd.DataFrame()
    if not speedup_long.empty:
        speedup_long.to_csv(os.path.join(ROOT_OUT, "paired_speedup_long.csv"), index=False)

    # ---- summary table (Table-I equivalent) ----
    def _ci95(s):
        n = s.count()
        return 0.0 if n < 2 else 1.96 * s.std(ddof=1) / math.sqrt(n)

    summary = df.groupby(["threshold", "method"]).agg(
        time_to_target_mean=("time_to_target_s", "mean"),
        time_to_target_ci95=("time_to_target_s", _ci95),
        caps_mean=("caps_at_target", "mean"),
        best_minz_mean=("best_minz_ohm", "mean"),
        best_minz_ci95=("best_minz_ohm", _ci95),
        success_rate=("success", "mean"),
        n_runs=("time_to_target_s", "count"),
    ).reset_index()

    if not speedup_long.empty:
        speedup_summary = speedup_long.groupby(["threshold", "method"]).agg(
            speedup_mean=("paired_speedup", "mean"),
            speedup_ci95=("paired_speedup", _ci95),
        ).reset_index()
        summary = summary.merge(speedup_summary, on=["threshold", "method"], how="left")

    summary = summary.sort_values(["threshold", "method"])
    summary_csv = os.path.join(ROOT_OUT, "summary_table.csv")
    summary.to_csv(summary_csv, index=False)

    matrix_csv = None
    if not speedup_long.empty:
        matrix = speedup_long.pivot_table(index="method", columns="threshold",
                                           values="paired_speedup", aggfunc="mean")
        matrix_csv = os.path.join(ROOT_OUT, "speedup_matrix.csv")
        matrix.to_csv(matrix_csv)

    # ---- config.json summarizing what was actually found in the data ----
    config_dict = dict(
        ROOT_OUT=ROOT_OUT,
        BASELINE_METHOD=BASELINE_METHOD,
        methods_present=present_methods,
        methods_expected=METHODS,
        methods_missing=missing_methods,
        thresholds_present=sorted(df["threshold"].unique().tolist(), reverse=True),
        n_records_from_jsonl=n_jsonl,
        n_records_from_legacy_logs=n_legacy,
        n_runs_per_method_threshold={
            f"{k[0]}|{k[1]}": v
            for k, v in df.groupby(["threshold", "method"]).size().to_dict().items()
        },
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with open(os.path.join(ROOT_OUT, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # ========================================================
    # FIGURES -- styled to match plotter.py (bold text throughout,
    # inward-facing axis ticks, single-hue heatmap, PDF export alongside
    # PNG). pure_python is never drawn as its own series -- it's the
    # implicit baseline every speedup number is measured against.
    # ========================================================

    DISPLAY_METHODS = ["numpy", "solve", "sm", "iterative"]  # pure_python excluded from series
    METHOD_LABEL = {"numpy": "NumPy", "solve": "Proposed", "sm": "SM",
                     "iterative": "Iterative", "pure_python": "Pure-Python"}
    METHOD_COLORS = {"solve": "#2E7D32", "sm": "#F39C12",
                      "iterative": "#D6336C", "numpy": "#1E88E5"}
    METHOD_MARKERS = {"solve": "s", "sm": "^", "iterative": "D", "numpy": "o"}
    BOLD = "bold"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.weight": BOLD,
        "axes.labelweight": BOLD,
        "axes.titleweight": BOLD,
        "figure.titleweight": BOLD,
        "font.size": 15,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "legend.title_fontsize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    def _bold_ticklabels(ax):
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(BOLD)

    def _inward_yaxis(ax, pad=-30, ha="left"):
        """Y-axis ticks + labels drawn INSIDE the plot (per the standing
        'text facing inward' requirement -- see the note at the top of
        this conversation's turn about plotter.py's unused helper)."""
        ax.tick_params(axis="y", direction="in", length=6, width=1.4, pad=pad)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment(ha)
            label.set_fontweight(BOLD)
            label.set_zorder(5)
        ax.tick_params(axis="x", direction="out", length=5, width=1.2)

    figs_dir = os.path.join(ROOT_OUT, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    def _savefig(fig, name):
        png_path = os.path.join(figs_dir, name)
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.3)
        fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        print(f"[PLOT] {png_path}")

    # ---- Figure 1: single-row heatmap, avg speedup vs baseline ----
    if matrix_csv is not None:
        mat = pd.read_csv(matrix_csv, index_col=0)
        cols = [m for m in DISPLAY_METHODS if m in mat.index]
        avg_speedup = {m: float(np.nanmean(mat.loc[m].values)) for m in cols}
        values = np.array([[avg_speedup[m] for m in cols]])  # shape (1, len(cols))
        col_labels = [METHOD_LABEL[m] for m in cols]

        fig = plt.figure(figsize=(8, 3.6))
        ax = fig.add_axes([0.07, 0.16, 0.66, 0.58])
        cax = fig.add_axes([0.775, 0.16, 0.032, 0.58])

        from matplotlib.colors import Normalize
        vmin, vmax = values.min(), values.max()
        pad = max((vmax - vmin) * 0.15, 0.05 * vmax)
        norm = Normalize(vmin=max(vmin - pad, 0), vmax=vmax + pad)

        im = ax.imshow(values, aspect="equal", cmap="Greens", norm=norm)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(col_labels, fontweight=BOLD, fontsize=12)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0, pad=8)

        ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 1, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2.4)
        ax.grid(which="major", visible=False)

        cmap = im.get_cmap()
        for j, m in enumerate(cols):
            val = values[0, j]
            r, g, b, _ = cmap(norm(val))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.6 else "black"
            ax.text(j, 0, f"{val:.2f}\u00d7", ha="center", va="center",
                     fontsize=14, fontweight=BOLD, color=text_color)

        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
        cbar.set_label(f"Avg. Speedup vs\n{METHOD_LABEL[BASELINE_METHOD]} (\u00d7)",
                        fontweight=BOLD, fontsize=11, rotation=270, labelpad=24)
        cbar.ax.tick_params(labelsize=10)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight(BOLD)

        ax.set_title(
            f"Average Speedup per PSO Run vs {METHOD_LABEL[BASELINE_METHOD]}\n"
            "(Mean over all Target Impedance Thresholds  |  Higher is Better)",
            fontweight=BOLD, fontsize=11.5)

        _savefig(fig, "speedup_heatmap.png")
    else:
        print(f"[NOTE] Skipping speedup_heatmap.png -- baseline method "
              f"{BASELINE_METHOD!r} not paired with any other method yet.")

    # ---- Figure 2: absolute time-to-target vs threshold (log scale) ----
    fig, ax = plt.subplots(figsize=(9, 7.2))
    thresholds_sorted = sorted(df["threshold"].unique(), reverse=True)
    xs = np.array(thresholds_sorted)

    any_series = False
    for method in DISPLAY_METHODS:
        if method not in present_methods:
            continue
        g = summary[summary["method"] == method].set_index("threshold").reindex(thresholds_sorted)
        ys = g["time_to_target_mean"].values
        ci = g["time_to_target_ci95"].values
        if np.all(np.isnan(ys)):
            continue
        any_series = True
        ax.plot(xs, ys, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                 linewidth=3, markersize=10, markeredgecolor="white",
                 markeredgewidth=1.3, label=METHOD_LABEL[method], zorder=3)
        ax.fill_between(xs, ys - ci, ys + ci, color=METHOD_COLORS[method],
                          alpha=0.18, linewidth=0, zorder=2)

    if any_series:
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{t:.3f}" for t in xs])
        ax.set_xlabel("Target Impedance Threshold (Ohm)", fontweight=BOLD)
        ax.set_ylabel("Average Time to Target (s, log scale)", fontweight=BOLD)
        ax.set_title("Average Runtime vs. Target Impedance\n"
                      "(Shaded Band = 95% CI  |  Lower is Better)", fontweight=BOLD)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        _inward_yaxis(ax)
        _bold_ticklabels(ax)
        ax.grid(True, which="major", alpha=0.30, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
        leg = ax.legend(title="Method", loc="best", frameon=True,
                          framealpha=0.95, edgecolor="#888888", borderpad=0.9,
                          handlelength=2.2, labelspacing=0.6)
        leg.get_title().set_fontweight(BOLD)
        for text in leg.get_texts():
            text.set_fontweight(BOLD)
        leg.set_zorder(6)
        fig.tight_layout()
        _savefig(fig, "runtime_reduction_vs_threshold.png")
    else:
        plt.close(fig)
        print("[NOTE] Skipping runtime_reduction_vs_threshold.png -- no method "
              "has any successful (time_to_target) runs yet.")

    # ---- Figure 3: best achieved |Z11| vs threshold ----
    fig, ax = plt.subplots(figsize=(9, 7.2))
    any_series = False
    for method in DISPLAY_METHODS:
        if method not in present_methods:
            continue
        g = summary[summary["method"] == method].set_index("threshold").reindex(thresholds_sorted)
        ys = g["best_minz_mean"].values
        ci = g["best_minz_ci95"].values
        if np.all(np.isnan(ys)):
            continue
        any_series = True
        ax.plot(xs, ys, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                 linewidth=3, markersize=10, markeredgecolor="white",
                 markeredgewidth=1.3, label=METHOD_LABEL[method], zorder=3)
        ax.fill_between(xs, ys - ci, ys + ci, color=METHOD_COLORS[method],
                          alpha=0.15, linewidth=0, zorder=2)

    if any_series:
        for t in thresholds_sorted:
            ax.axhline(t, linestyle=":", linewidth=1.0, color="grey", alpha=0.6, zorder=1)
        ax.invert_xaxis()
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{t:.3f}" for t in xs])
        ax.set_xlabel("Target Impedance Threshold (Ohm)", fontweight=BOLD)
        ax.set_ylabel("Average Best |Z\u2081\u2081| (Ohm)", fontweight=BOLD)
        ax.set_title("Best Achieved Impedance vs. Target\n"
                      "(Shaded Band = 95% CI, Lower is Better)", fontweight=BOLD)
        _inward_yaxis(ax)
        _bold_ticklabels(ax)
        ax.grid(True, alpha=0.30, linewidth=0.8)
        leg = ax.legend(title="Method", loc="best", frameon=True,
                          framealpha=0.95, edgecolor="#888888", borderpad=0.9,
                          handlelength=2.2, labelspacing=0.6)
        leg.get_title().set_fontweight(BOLD)
        for text in leg.get_texts():
            text.set_fontweight(BOLD)
        leg.set_zorder(6)
        fig.tight_layout()
        _savefig(fig, "best_minz_vs_threshold.png")
    else:
        plt.close(fig)
        print("[NOTE] Skipping best_minz_vs_threshold.png -- no data yet.")

    # ---- top10 fastest successful runs + per-threshold best ----
    valid = df[df["success"] & df["time_to_target_s"].notna()]
    top10 = valid.sort_values("time_to_target_s").head(10)

    with open(os.path.join(ROOT_OUT, "top10_fastest.txt"), "w") as f:
        for i, row in enumerate(top10.itertuples(index=False), 1):
            f.write(f"{i}. {row._asdict()}\n")
            src = os.path.join(row.folder, "run_plot.png") if row.folder else None
            if src and os.path.exists(src):
                shutil.copy(src, os.path.join(ROOT_OUT, f"top{i}.png"))

    with open(os.path.join(ROOT_OUT, "statistics.txt"), "w") as f:
        for threshold in sorted(df["threshold"].unique(), reverse=True):
            f.write(f"\n=== Threshold {threshold} ===\n")
            subset = valid[valid["threshold"] == threshold]
            if not subset.empty:
                best = subset.loc[subset["time_to_target_s"].idxmin()]
                f.write(f"BEST: {best.to_dict()}\n")

    print(f"\nStructured outputs written under: {ROOT_OUT}")
    print(f"  {results_long_csv}")
    print(f"  {summary_csv}")
    if matrix_csv:
        print(f"  {matrix_csv}")
    print(f"  {figs_dir}/*.png (+ .pdf)")

print("Done.")