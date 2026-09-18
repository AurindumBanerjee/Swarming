"""
AnalyzeScratchBench.py
======================
Publication-quality analysis of PDN decoupling capacitor PSO benchmark logs.
Supports BOTH old and new log formats automatically.

Old format (no iter / placement / best_minZ in RESULT):
    2026-04-07 11:40:55,355 INFO: n_caps=1 | minZ=0.084480
    2026-04-07 11:59:42,098 INFO: RESULT | time_to_target=1177.26 | caps=19

New format (iter, placement, best_minZ present):
    n_caps=7 | iter=5 | minZ=0.043051 | placement={3: 120, 5: 200}
    RESULT | time_to_target=3.45 | caps=4 | best_minZ=0.029866

Usage:
    python AnalyzeScratchBench.py
"""

import os
import re
import sys
import ast
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROOT_OUT  = Path("MinTime/ScratchBench5")
ANALYSIS  = ROOT_OUT / "Analysis"
METHODS   = ["numpy", "solve", "sm", "iterative"]
TARGETS   = [0.05, 0.045, 0.04, 0.03]
NUM_RUNS  = 20
DPI       = 300

METHOD_COLORS = {
    "numpy":     "#2196F3",
    "solve":     "#4CAF50",
    "sm":        "#FF9800",
    "iterative": "#E91E63",
}
METHOD_MARKERS = {
    "numpy":     "o",
    "solve":     "s",
    "sm":        "^",
    "iterative": "D",
}
THRESHOLD_MARKERS = {0.05: "o", 0.045: "s", 0.04: "^", 0.03: "D"}

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use("default")
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "figure.dpi":        DPI,
    "savefig.dpi":       DPI,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

SUB_DIRS = ["Heatmaps", "Boxplots", "RunGrids", "Scatter", "Frequencies", "CSV"]

def create_output_dirs():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for d in SUB_DIRS:
        (ANALYSIS / d).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {ANALYSIS.resolve()}")

# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS  –  dual-format
# ─────────────────────────────────────────────────────────────────────────────

# Timestamp: "2026-04-07 11:40:55,355"  or  "11:40:55,355"
RE_TIMESTAMP = re.compile(r"(\d{2}:\d{2}:\d{2}),\d+")

# RUN_START (new format only; old logs infer method/threshold/run from path)
RE_RUN_START = re.compile(
    r"RUN_START \| method=(\S+) \| threshold=([\d.]+) \| run=(\d+)"
)

# n_caps line – both formats
#   old: n_caps=1 | minZ=0.084480
#   new: n_caps=1 | iter=5 | minZ=0.084480 | placement={...}
RE_NCAPS_OLD = re.compile(
    r"n_caps=(\d+) \| minZ=([\d.eE+\-]+)"
)
RE_NCAPS_NEW = re.compile(
    r"n_caps=(\d+) \| iter=(\d+) \| minZ=([\d.eE+\-]+) \| placement=(\{[^}]*\})"
)

# RESULT line – best_minZ optional
RE_RESULT = re.compile(
    r"RESULT \| time_to_target=([\d.\-]+) \| caps=([\d\-]+)"
    r"(?:\s*\|\s*best_minZ=([\d.eE+\-]+))?"
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_float(s):
    if s is None:
        return np.nan
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def parse_placement(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            cleaned = re.sub(r"(\d+):\s*(\d+)", r'"\1": \2', s)
            return {int(k): int(v) for k, v in ast.literal_eval(cleaned).items()}
        except Exception:
            return {}

def parse_hms(s):
    """Parse HH:MM:SS string → total seconds as float."""
    try:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return np.nan

def ci95(series):
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    return 1.96 * s.std() / np.sqrt(len(s))

# ─────────────────────────────────────────────────────────────────────────────
# LOG PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(log_path: Path, method_hint: str, threshold_hint: float, run_hint: int):
    """
    Parse one run log.  Supports old and new formats simultaneously.

    Returns
    -------
    run_record   : dict | None
    caps_records : list[dict]
    has_placement: bool  (True only if ≥1 placement dict was found)
    """
    run_record    = None
    caps_records  = []
    has_placement = False

    method    = method_hint
    threshold = threshold_hint
    run_id    = run_hint

    # Accumulate minZ values so we can compute best_minZ when not in RESULT
    minz_list       = []
    # For timestamp-based elapsed time
    first_ts_sec    = None
    prev_ts_sec     = None
    # nc→elapsed mapping
    nc_elapsed      = {}   # {n_caps: elapsed_seconds_since_run_start}
    nc_delta        = {}   # {n_caps: seconds spent on this n_caps}

    try:
        with open(log_path, "r", errors="replace") as fh:
            for line in fh:

                # ── Extract timestamp ─────────────────────────────────────────
                ts_match = RE_TIMESTAMP.search(line)
                current_ts_sec = np.nan
                if ts_match:
                    current_ts_sec = parse_hms(ts_match.group(1))
                    if first_ts_sec is None:
                        first_ts_sec = current_ts_sec

                # ── RUN_START (new format) ────────────────────────────────────
                m = RE_RUN_START.search(line)
                if m:
                    method    = m.group(1)
                    threshold = float(m.group(2))
                    run_id    = int(m.group(3))
                    first_ts_sec = current_ts_sec  # reset reference
                    prev_ts_sec  = current_ts_sec
                    continue

                # ── n_caps: try new format first, fall back to old ─────────────
                m_new = RE_NCAPS_NEW.search(line)
                m_old = RE_NCAPS_OLD.search(line) if not m_new else None

                if m_new or m_old:
                    if m_new:
                        nc        = int(m_new.group(1))
                        iters     = int(m_new.group(2))
                        minz      = safe_float(m_new.group(3))
                        placement = parse_placement(m_new.group(4))
                        has_placement = True
                    else:
                        nc        = int(m_old.group(1))
                        iters     = None
                        minz      = safe_float(m_old.group(2))
                        placement = None

                    # Compute elapsed / delta times from timestamps
                    elapsed = np.nan
                    delta   = np.nan
                    if np.isfinite(current_ts_sec) and first_ts_sec is not None:
                        elapsed = current_ts_sec - first_ts_sec
                        # Handle midnight rollover
                        if elapsed < 0:
                            elapsed += 86400
                    if np.isfinite(current_ts_sec) and prev_ts_sec is not None:
                        delta = current_ts_sec - prev_ts_sec
                        if delta < 0:
                            delta += 86400

                    nc_elapsed[nc] = elapsed
                    nc_delta[nc]   = delta
                    prev_ts_sec    = current_ts_sec

                    minz_list.append(minz)

                    rec = {
                        "method":          method,
                        "threshold":       threshold,
                        "run_id":          run_id,
                        "n_caps":          nc,
                        "iterations_used": iters,
                        "minZ":            minz,
                        "placement":       placement,
                        "elapsed_s":       elapsed,
                        "delta_s":         delta,
                    }
                    caps_records.append(rec)
                    continue

                # ── RESULT ────────────────────────────────────────────────────
                m = RE_RESULT.search(line)
                if m:
                    ttt  = safe_float(m.group(1))
                    caps = int(m.group(2))
                    bz   = safe_float(m.group(3)) if m.group(3) else np.nan

                    # Fall back: compute best_minZ from n_caps rows
                    if np.isnan(bz) and minz_list:
                        valid = [v for v in minz_list if np.isfinite(v)]
                        bz = min(valid) if valid else np.nan

                    run_record = {
                        "method":         method,
                        "threshold":      threshold,
                        "run_id":         run_id,
                        "time_to_target": ttt  if (ttt  is not None and ttt  > 0) else np.nan,
                        "caps_at_target": caps if (caps is not None and caps > 0) else np.nan,
                        "best_minZ":      bz,
                        "success":        (ttt is not None and ttt > 0),
                        "log_path":       str(log_path),
                    }

    except Exception as exc:
        print(f"[WARN] Failed to parse {log_path}: {exc}")

    return run_record, caps_records, has_placement

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT ALL LOGS
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_logs():
    run_records   = []
    caps_records  = []
    log_count     = 0
    any_placement = False

    for method in METHODS:
        for thr in TARGETS:
            folder = ROOT_OUT / f"{method}_{thr}"
            if not folder.exists():
                print(f"[WARN] Missing folder: {folder}")
                continue
            for run_id in range(1, NUM_RUNS + 1):
                log_file = folder / f"run_{run_id}" / f"run_{run_id}.log"
                if not log_file.exists():
                    print(f"[WARN] Missing log: {log_file}")
                    continue
                log_count += 1
                rr, crs, has_pl = parse_log(log_file, method, thr, run_id)
                if rr:
                    run_records.append(rr)
                caps_records.extend(crs)
                if has_pl:
                    any_placement = True

    print(f"[INFO] Logs found       : {log_count}")
    print(f"[INFO] Runs parsed      : {len(run_records)}")
    print(f"[INFO] Cap-rows parsed  : {len(caps_records)}")
    print(f"[INFO] Placement data   : {'YES' if any_placement else 'NO (old format)'}")

    df_runs = pd.DataFrame(run_records) if run_records else pd.DataFrame()
    df_caps = pd.DataFrame(caps_records) if caps_records else pd.DataFrame()

    return df_runs, df_caps, log_count, any_placement

# ─────────────────────────────────────────────────────────────────────────────
# SAVE FIGURE  (PNG + PDF, returns True so guarded() can count it)
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig, subfolder: str, stem: str):
    base = ANALYSIS / subfolder / stem
    fig.savefig(str(base) + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {base}.png/.pdf")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# PIVOT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def pivot_metric(df, col, agg="mean"):
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    piv = (df.groupby(["method", "threshold"])[col]
             .agg(agg)
             .unstack("threshold"))
    piv = piv.reindex([m for m in METHODS if m in piv.index])
    # Enforce TARGETS order (easy → hard: 0.05, 0.045, 0.04, 0.03)
    ordered_cols = [t for t in TARGETS if t in piv.columns]
    piv = piv.reindex(columns=ordered_cols)
    return piv

# ─────────────────────────────────────────────────────────────────────────────
# LINE PLOT WITH CI95 BANDS  (shared by several figures)
# ─────────────────────────────────────────────────────────────────────────────

def _line_with_ci(ax, df, col, x_order_reversed=True):
    """Plot mean ± CI95 line per method on ax, x = thresholds."""
    thr_order = sorted(TARGETS, reverse=x_order_reversed)
    for method in METHODS:
        xs, means, cis = [], [], []
        for thr in thr_order:
            sub = df[(df["method"] == method) & (df["threshold"] == thr)][col].dropna()
            if sub.empty:
                continue
            xs.append(thr)
            means.append(sub.mean())
            cis.append(ci95(sub))
        if not xs:
            continue
        xs, means, cis = np.array(xs), np.array(means), np.array(cis)
        ax.plot(xs, means, marker=METHOD_MARKERS[method], linewidth=2,
                color=METHOD_COLORS[method], label=method.capitalize(), markersize=7)
        ax.fill_between(xs, means - cis, means + cis,
                        alpha=0.18, color=METHOD_COLORS[method])
    if x_order_reversed:
        ax.invert_xaxis()
    ax.set_xticks(sorted(TARGETS, reverse=x_order_reversed))
    ax.legend(title="Method", loc="best")

# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

def export_csvs(df_runs, df_caps):
    csv_dir = ANALYSIS / "CSV"

    # summary_runs
    df_runs.drop(columns=["log_path"], errors="ignore").to_csv(
        csv_dir / "summary_runs.csv", index=False)
    print(f"[CSV] summary_runs.csv")

    # summary_caps (drop placement dict for CSV)
    df_caps.drop(columns=["placement"], errors="ignore").to_csv(
        csv_dir / "summary_caps.csv", index=False)
    print(f"[CSV] summary_caps.csv")

    if df_runs.empty or "time_to_target" not in df_runs.columns:
        return

    # top_runs
    (df_runs.dropna(subset=["time_to_target"])
            .sort_values("time_to_target")
            .head(50)
            .drop(columns=["log_path"], errors="ignore")
            .to_csv(csv_dir / "top_runs.csv", index=False))
    print(f"[CSV] top_runs.csv")

    # performance_summary
    grp = df_runs.groupby(["method", "threshold"])
    perf = grp.agg(
        avg_time  = ("time_to_target", "mean"),
        std_time  = ("time_to_target", "std"),
        avg_caps  = ("caps_at_target", "mean"),
        std_caps  = ("caps_at_target", "std"),
        avg_minz  = ("best_minZ",      "mean"),
        std_minz  = ("best_minZ",      "std"),
        best_time = ("time_to_target", "min"),
        best_minz = ("best_minZ",      "min"),
    ).reset_index()
    perf.to_csv(csv_dir / "performance_summary.csv", index=False)
    print(f"[CSV] performance_summary.csv")

    # statistical_summary  (new)
    rows = []
    for (method, thr), g in df_runs.groupby(["method", "threshold"]):
        n_parsed  = len(g)                       # actual parsed runs, not fixed NUM_RUNS
        n_success = int(g["success"].sum()) if "success" in g.columns else len(g.dropna(subset=["time_to_target"]))
        t = g["time_to_target"].dropna()
        c = g["caps_at_target"].dropna()
        z = g["best_minZ"].dropna()
        rows.append({
            "method":       method,
            "threshold":    thr,
            "n_parsed":     n_parsed,
            "n_success":    n_success,
            "success_rate": n_success / n_parsed if n_parsed > 0 else np.nan,
            "avg_time":     t.mean(),
            "median_time":  t.median(),
            "std_time":     t.std(),
            "ci95_time":    ci95(t),
            "avg_caps":     c.mean(),
            "median_caps":  c.median(),
            "std_caps":     c.std(),
            "ci95_caps":    ci95(c),
            "avg_minz":     z.mean(),
            "median_minz":  z.median(),
            "std_minz":     z.std(),
            "ci95_minz":    ci95(z),
            "best_time":    t.min() if not t.empty else np.nan,
            "best_minz":    z.min() if not z.empty else np.nan,
        })
    pd.DataFrame(rows).to_csv(csv_dir / "statistical_summary.csv", index=False)
    print(f"[CSV] statistical_summary.csv")

    # paper_table.csv  – Fix 6: LaTeX-ready results table with paired speedup
    paper_rows = []
    for method in METHODS:
        for thr in TARGETS:
            g = df_runs[(df_runs["method"] == method) & (df_runs["threshold"] == thr)]
            n_parsed  = len(g)
            n_success = int(g["success"].sum()) if "success" in g.columns else len(g.dropna(subset=["time_to_target"]))
            t = g["time_to_target"].dropna()
            c = g["caps_at_target"].dropna()
            z = g["best_minZ"].dropna()

            # Paired speedup vs numpy for this (method, threshold)
            if method == "numpy":
                speedup_mean = 1.0
            else:
                np_sub = (df_runs[(df_runs["method"] == "numpy") &
                                   (df_runs["threshold"] == thr)]
                          .dropna(subset=["time_to_target"])
                          .set_index("run_id")["time_to_target"])
                m_sub  = (g.dropna(subset=["time_to_target"])
                           .set_index("run_id")["time_to_target"])
                common = np_sub.index.intersection(m_sub.index)
                if len(common) > 0:
                    speedup_mean = (np_sub.loc[common] / m_sub.loc[common]).mean()
                else:
                    speedup_mean = np.nan

            paper_rows.append({
                "Method":       method.capitalize(),
                "Threshold":    thr,
                "Avg_Time_s":   round(t.mean(), 2) if not t.empty else np.nan,
                "Speedup":      round(speedup_mean, 3) if np.isfinite(speedup_mean) else np.nan,
                "Avg_Caps":     round(c.mean(), 2) if not c.empty else np.nan,
                "Avg_MinZ":     round(z.mean(), 6) if not z.empty else np.nan,
                "Success_Rate": round(n_success / n_parsed * 100, 1) if n_parsed > 0 else np.nan,
            })
    pd.DataFrame(paper_rows).to_csv(csv_dir / "paper_table.csv", index=False)
    print(f"[CSV] paper_table.csv")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════

# ─── Fig 1: runtime_heatmap ──────────────────────────────────────────────────

def fig_runtime_heatmap(df_runs):
    piv = pivot_metric(df_runs, "time_to_target", "mean")
    if piv.empty:
        print("[WARN] Skipping runtime_heatmap – no data"); return False
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(piv, ax=ax, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, linecolor="#cccccc",
                annot_kws={"size": 12, "weight": "bold"},
                cbar_kws={"label": "Avg. Time-to-Target (s)"})
    ax.set_title("Average Runtime to Target Impedance\n(Lower is Better)", pad=12)
    ax.set_xlabel("Target Impedance Threshold (Ω)")
    ax.set_ylabel("Inversion Method")
    ax.set_xticklabels([f"{float(c):.3f}" for c in piv.columns])
    fig.tight_layout()
    return save_fig(fig, "Heatmaps", "runtime_heatmap")

# ─── Fig 2: speedup_heatmap ──────────────────────────────────────────────────

def fig_speedup_heatmap(df_runs):
    piv = pivot_metric(df_runs, "time_to_target", "mean")
    if piv.empty or "numpy" not in piv.index:
        print("[WARN] Skipping speedup_heatmap – insufficient data"); return False
    numpy_row = piv.loc["numpy"]
    others    = piv.drop(index="numpy")
    speedup   = others.apply(lambda row: numpy_row / row, axis=1)

    def fmt_cell(v):
        try:
            return f"{v:.2f}×" if np.isfinite(v) else "N/A"
        except Exception:
            return "N/A"
    _mf   = speedup.map if hasattr(speedup, "map") else speedup.applymap
    annot = _mf(fmt_cell)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(speedup, ax=ax, annot=annot, fmt="", cmap="Greens",
                linewidths=0.5, linecolor="#cccccc",
                annot_kws={"size": 12, "weight": "bold"},
                cbar_kws={"label": "Speedup vs NumPy"})
    ax.set_title("Speedup Relative to NumPy Baseline\n(Higher is Better)", pad=12)
    ax.set_xlabel("Target Impedance Threshold (Ω)")
    ax.set_ylabel("Inversion Method")
    ax.set_xticklabels([f"{float(c):.3f}" for c in speedup.columns])
    fig.tight_layout()
    return save_fig(fig, "Heatmaps", "speedup_heatmap")

# ─── Fig 3: caps_heatmap ─────────────────────────────────────────────────────

def fig_caps_heatmap(df_runs):
    piv = pivot_metric(df_runs, "caps_at_target", "mean")
    if piv.empty:
        print("[WARN] Skipping caps_heatmap – no data"); return False
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(piv, ax=ax, annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.5, linecolor="#cccccc",
                annot_kws={"size": 12},
                cbar_kws={"label": "Avg. Capacitors at Target"})
    ax.set_title("Average Capacitor Count at Target\n(Lower is Better)", pad=12)
    ax.set_xlabel("Target Impedance Threshold (Ω)")
    ax.set_ylabel("Inversion Method")
    ax.set_xticklabels([f"{float(c):.3f}" for c in piv.columns])
    fig.tight_layout()
    return save_fig(fig, "Heatmaps", "caps_heatmap")

# ─── Fig 4: minz_heatmap ─────────────────────────────────────────────────────

def fig_minz_heatmap(df_runs):
    piv = pivot_metric(df_runs, "best_minZ", "mean")
    if piv.empty:
        print("[WARN] Skipping minz_heatmap – no data"); return False
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(piv, ax=ax, annot=True, fmt=".4f", cmap="RdYlGn_r",
                linewidths=0.5, linecolor="#cccccc",
                annot_kws={"size": 11},
                cbar_kws={"label": "Avg. Best |Z₁₁| (Ω)"})
    ax.set_title("Average Best Achieved Impedance\n(Lower is Better)", pad=12)
    ax.set_xlabel("Target Impedance Threshold (Ω)")
    ax.set_ylabel("Inversion Method")
    ax.set_xticklabels([f"{float(c):.3f}" for c in piv.columns])
    fig.tight_layout()
    return save_fig(fig, "Heatmaps", "minz_heatmap")

# ─── Fig 5: success_rate_heatmap  (new) ──────────────────────────────────────

def fig_success_rate_heatmap(df_runs):
    if df_runs.empty or "success" not in df_runs.columns:
        print("[WARN] Skipping success_rate_heatmap – no data"); return False

    # Fix 3: denominator = parsed runs for that cell, not fixed NUM_RUNS
    records = []
    for method in METHODS:
        for thr in TARGETS:
            sub = df_runs[(df_runs["method"] == method) & (df_runs["threshold"] == thr)]
            n_parsed = len(sub)
            if n_parsed == 0:
                rate = np.nan
            else:
                n_success = int(sub["success"].sum())
                rate = n_success / n_parsed * 100
            records.append({"method": method, "threshold": thr, "rate": rate})

    df_rec = pd.DataFrame(records)
    piv = df_rec.pivot(index="method", columns="threshold", values="rate")
    piv = piv.reindex([m for m in METHODS if m in piv.index])
    # Enforce TARGETS order (easy → hard)
    ordered_cols = [t for t in TARGETS if t in piv.columns]
    piv = piv.reindex(columns=ordered_cols)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(piv, ax=ax, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=0, vmax=100,
                linewidths=0.5, linecolor="#cccccc",
                annot_kws={"size": 13, "weight": "bold"},
                cbar_kws={"label": "Success Rate (%)"})
    ax.set_title("PSO Success Rate per Method × Threshold\n"
                 "(% of Parsed Runs that Reached Target within MAX_CAPS Budget)", pad=12)
    ax.set_xlabel("Target Impedance Threshold (Ω)")
    ax.set_ylabel("Inversion Method")
    ax.set_xticklabels([f"{float(c):.3f}" for c in piv.columns])
    fig.tight_layout()
    return save_fig(fig, "Heatmaps", "success_rate_heatmap")

# ─── Fig 6: runtime_boxplots ─────────────────────────────────────────────────

def fig_runtime_boxplots(df_runs):
    if df_runs.empty or "time_to_target" not in df_runs.columns:
        print("[WARN] Skipping runtime_boxplots – no data"); return False
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 5), sharey=False)
    fig.suptitle("Distribution of Time-to-Target by Method and Threshold",
                 fontsize=15, fontweight="bold", y=1.01)
    for ax, thr in zip(axes, TARGETS):
        sub  = df_runs[df_runs["threshold"] == thr].dropna(subset=["time_to_target"])
        data = [sub[sub["method"] == m]["time_to_target"].values for m in METHODS]
        bp   = ax.boxplot(data, patch_artist=True, notch=False, widths=0.55,
                          medianprops=dict(color="black", linewidth=2))
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLORS[m]); patch.set_alpha(0.75)
        ax.set_title(f"Threshold = {thr} Ω", fontsize=12)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels(METHODS, rotation=20, ha="right")
        ax.set_xlabel("Method"); ax.set_ylabel("Time to Target (s)")
    fig.tight_layout()
    return save_fig(fig, "Boxplots", "runtime_boxplots")

# ─── Fig 7: caps_boxplots ────────────────────────────────────────────────────

def fig_caps_boxplots(df_runs):
    if df_runs.empty or "caps_at_target" not in df_runs.columns:
        print("[WARN] Skipping caps_boxplots – no data"); return False
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 5), sharey=False)
    fig.suptitle("Distribution of Capacitor Count at Target",
                 fontsize=15, fontweight="bold", y=1.01)
    for ax, thr in zip(axes, TARGETS):
        sub  = df_runs[df_runs["threshold"] == thr].dropna(subset=["caps_at_target"])
        data = [sub[sub["method"] == m]["caps_at_target"].values for m in METHODS]
        bp   = ax.boxplot(data, patch_artist=True, notch=False, widths=0.55,
                          medianprops=dict(color="black", linewidth=2))
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLORS[m]); patch.set_alpha(0.75)
        ax.set_title(f"Threshold = {thr} Ω", fontsize=12)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels(METHODS, rotation=20, ha="right")
        ax.set_xlabel("Method"); ax.set_ylabel("Capacitors at Target")
    fig.tight_layout()
    return save_fig(fig, "Boxplots", "caps_boxplots")

# ─── Fig 8: tradeoff_scatter ─────────────────────────────────────────────────

def fig_tradeoff_scatter(df_runs):
    if df_runs.empty:
        print("[WARN] Skipping tradeoff_scatter – no data"); return False
    fig, ax = plt.subplots(figsize=(10, 7))
    for method in METHODS:
        for thr in TARGETS:
            sub = df_runs[(df_runs["method"] == method) &
                          (df_runs["threshold"] == thr)
                          ].dropna(subset=["time_to_target", "best_minZ"])
            if sub.empty: continue
            ax.scatter(sub["time_to_target"], sub["best_minZ"],
                       color=METHOD_COLORS[method], marker=THRESHOLD_MARKERS[thr],
                       s=70, alpha=0.75, edgecolors="white", linewidths=0.5)
    method_patches = [mpatches.Patch(color=METHOD_COLORS[m], label=m.capitalize())
                      for m in METHODS]
    thr_handles = [plt.Line2D([0], [0], marker=THRESHOLD_MARKERS[t], color="grey",
                               linestyle="None", markersize=8, label=f"Thr={t}")
                   for t in TARGETS]
    leg1 = ax.legend(handles=method_patches, title="Method",
                     loc="upper right", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=thr_handles, title="Threshold (Ω)",
              loc="upper left", framealpha=0.9)
    for thr in TARGETS:
        ax.axhline(thr, linestyle=":", linewidth=0.8, color="grey", alpha=0.5)
    ax.set_xlabel("Time to Target (s)", fontsize=13)
    ax.set_ylabel("Best Achieved |Z₁₁| (Ω)", fontsize=13)
    ax.set_title("Runtime–Solution Quality Trade-off\n(Lower-left is Better)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "Scatter", "tradeoff_scatter")

# ─── Fig 9: Optimization Quality Grid (renamed from "Convergence Profile") ───

def fig_run_grid(df_caps):
    """4×4: rows=methods, cols=thresholds. Mean |Z₁₁| vs n_caps."""
    if df_caps.empty:
        print("[WARN] Skipping run_grid – no cap data"); return False
    fig, axes = plt.subplots(len(METHODS), len(TARGETS), figsize=(18, 14),
                             sharex=False, sharey=False)
    fig.suptitle(
        "Mean Best Impedance vs Capacitor Count\n"
        "(Shaded Band = ±1 SD over Runs)",
        fontsize=15, fontweight="bold", y=1.01)
    for row_idx, method in enumerate(METHODS):
        for col_idx, thr in enumerate(TARGETS):
            ax  = axes[row_idx][col_idx]
            sub = df_caps[(df_caps["method"] == method) &
                          (df_caps["threshold"] == thr)].dropna(subset=["n_caps","minZ"])
            color = METHOD_COLORS[method]
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            else:
                grp = sub.groupby("n_caps")["minZ"].agg(["mean","std"]).reset_index()
                grp["std"] = grp["std"].fillna(0)
                ax.plot(grp["n_caps"], grp["mean"], color=color, linewidth=2.0,
                        marker=METHOD_MARKERS[method], markersize=5, label="Mean |Z₁₁|", zorder=3)
                ax.fill_between(grp["n_caps"],
                                grp["mean"] - grp["std"],
                                grp["mean"] + grp["std"],
                                alpha=0.20, color=color, label="±1 SD")
                ax.axhline(thr, linestyle="--", linewidth=1.3,
                           color="#333333", alpha=0.8, label=f"Target={thr}")
            if row_idx == 0:
                ax.set_title(f"Threshold = {thr} Ω", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{method}\n|Z₁₁| (Ω)", fontsize=10, fontweight="bold")
            if row_idx == len(METHODS) - 1:
                ax.set_xlabel("No. of Capacitors", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return save_fig(fig, "RunGrids", "run_grid")

# ─── Fig 10: best_so_far_grid  (new, primary paper figure) ──────────────────

def fig_best_so_far_grid(df_caps):
    """4×4: mean raw minZ + mean best-so-far envelope + threshold line."""
    if df_caps.empty:
        print("[WARN] Skipping best_so_far_grid – no cap data"); return False

    fig, axes = plt.subplots(len(METHODS), len(TARGETS), figsize=(18, 14),
                             sharex=False, sharey=False)
    fig.suptitle(
        "Best-So-Far Impedance Envelope vs Capacitor Budget\n"
        "(Primary Paper Figure: Optimization Quality × Solver Speed)",
        fontsize=15, fontweight="bold", y=1.01)

    for row_idx, method in enumerate(METHODS):
        for col_idx, thr in enumerate(TARGETS):
            ax  = axes[row_idx][col_idx]
            sub = df_caps[(df_caps["method"] == method) &
                          (df_caps["threshold"] == thr)].dropna(subset=["n_caps","minZ"])
            color = METHOD_COLORS[method]

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            else:
                # Build best-so-far per run, then average across runs
                run_raw  = defaultdict(dict)   # run_id → {nc: minZ}
                run_bsf  = defaultdict(dict)   # run_id → {nc: best_so_far}
                all_ncs  = sorted(sub["n_caps"].unique())

                for _, row in sub.iterrows():
                    run_raw[row["run_id"]][int(row["n_caps"])] = row["minZ"]

                for rid, nc_dict in run_raw.items():
                    sorted_ncs = sorted(nc_dict.keys())
                    best = np.inf
                    for nc in sorted_ncs:
                        v = nc_dict[nc]
                        if np.isfinite(v):
                            best = min(best, v)
                        run_bsf[rid][nc] = best if np.isfinite(best) else np.nan

                # Aggregate
                def agg_over_runs(run_dict):
                    rows_list = []
                    for rid, nc_dict in run_dict.items():
                        for nc, v in nc_dict.items():
                            rows_list.append({"n_caps": nc, "val": v})
                    if not rows_list:
                        return pd.DataFrame()
                    df_t = pd.DataFrame(rows_list)
                    return df_t.groupby("n_caps")["val"].agg(["mean","std"]).reset_index()

                g_raw = agg_over_runs(run_raw)
                g_bsf = agg_over_runs(run_bsf)
                for g in [g_raw, g_bsf]:
                    if not g.empty:
                        g["std"] = g["std"].fillna(0)

                if not g_raw.empty:
                    ax.plot(g_raw["n_caps"], g_raw["mean"],
                            color=color, linewidth=1.2, linestyle="--",
                            alpha=0.55, label="Mean |Z₁₁|")
                if not g_bsf.empty:
                    ax.plot(g_bsf["n_caps"], g_bsf["mean"],
                            color=color, linewidth=2.2, marker=METHOD_MARKERS[method],
                            markersize=4, label="Best-so-far")
                    ax.fill_between(g_bsf["n_caps"],
                                    g_bsf["mean"] - g_bsf["std"],
                                    g_bsf["mean"] + g_bsf["std"],
                                    alpha=0.18, color=color)
                ax.axhline(thr, linestyle="--", linewidth=1.3,
                           color="#333333", alpha=0.85, label=f"Target={thr}")

            if row_idx == 0:
                ax.set_title(f"Threshold = {thr} Ω", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{method}\n|Z₁₁| (Ω)", fontsize=10, fontweight="bold")
            if row_idx == len(METHODS) - 1:
                ax.set_xlabel("No. of Capacitors", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Shared legend strip
    handles = [
        plt.Line2D([0],[0], linestyle="--", color="grey", linewidth=1.4, label="Mean |Z₁₁|"),
        plt.Line2D([0],[0], linestyle="-",  color="grey", linewidth=2.2, label="Best-so-far env."),
        plt.Line2D([0],[0], linestyle="--", color="#333333", linewidth=1.3, label="Target threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout()
    return save_fig(fig, "RunGrids", "best_so_far_grid")

# ─── Fig 11: runtime_accumulation_grid  (new, highly important) ──────────────

def fig_runtime_accumulation_grid(df_caps):
    """4×4: cumulative wall-clock elapsed time vs n_caps."""
    if df_caps.empty or "elapsed_s" not in df_caps.columns:
        print("[WARN] Skipping elapsed_time_accumulation_grid – no timestamp data"); return False
    if df_caps["elapsed_s"].isna().all():
        print("[WARN] Skipping elapsed_time_accumulation_grid – all elapsed_s are NaN"); return False

    fig, axes = plt.subplots(len(METHODS), len(TARGETS), figsize=(18, 14),
                             sharex=False, sharey=False)
    fig.suptitle(
        "Cumulative Wall-Clock Elapsed Time vs Capacitor Budget\n"
        "(Mean ± 95% CI over Runs  |  Demonstrates Solver Speed Differences)",
        fontsize=15, fontweight="bold", y=1.01)

    for row_idx, method in enumerate(METHODS):
        for col_idx, thr in enumerate(TARGETS):
            ax  = axes[row_idx][col_idx]
            sub = df_caps[(df_caps["method"] == method) &
                          (df_caps["threshold"] == thr)].dropna(subset=["n_caps","elapsed_s"])
            color = METHOD_COLORS[method]

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            else:
                grp = sub.groupby("n_caps")["elapsed_s"].agg(
                    mean="mean", std="std", n="count").reset_index()
                grp["ci"] = 1.96 * grp["std"] / np.sqrt(grp["n"].clip(lower=1))
                grp["ci"] = grp["ci"].fillna(0)

                ax.plot(grp["n_caps"], grp["mean"], color=color, linewidth=2.0,
                        marker=METHOD_MARKERS[method], markersize=5, zorder=3)
                ax.fill_between(grp["n_caps"],
                                grp["mean"] - grp["ci"],
                                grp["mean"] + grp["ci"],
                                alpha=0.20, color=color)

            if row_idx == 0:
                ax.set_title(f"Threshold = {thr} Ω", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{method}\nWall-Clock (s)", fontsize=10, fontweight="bold")
            if row_idx == len(METHODS) - 1:
                ax.set_xlabel("No. of Capacitors", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    return save_fig(fig, "RunGrids", "elapsed_time_accumulation_grid")

# ─── Fig 12: runtime_per_cap_grid  (new) ─────────────────────────────────────

def fig_runtime_per_cap_grid(df_caps):
    """4×4: Δwall-clock time per capacitor budget step – reveals scaling behaviour."""
    if df_caps.empty or "delta_s" not in df_caps.columns:
        print("[WARN] Skipping elapsed_time_per_cap_grid – no delta_s data"); return False
    if df_caps["delta_s"].isna().all():
        print("[WARN] Skipping elapsed_time_per_cap_grid – all delta_s are NaN"); return False

    fig, axes = plt.subplots(len(METHODS), len(TARGETS), figsize=(18, 14),
                             sharex=False, sharey=False)
    fig.suptitle(
        "Per-Step Wall-Clock Elapsed Time vs Capacitor Budget\n"
        "(Time Between Consecutive Log Entries  |  Reveals Solver Scaling Behaviour)",
        fontsize=15, fontweight="bold", y=1.01)

    for row_idx, method in enumerate(METHODS):
        for col_idx, thr in enumerate(TARGETS):
            ax  = axes[row_idx][col_idx]
            sub = df_caps[(df_caps["method"] == method) &
                          (df_caps["threshold"] == thr)].dropna(subset=["n_caps","delta_s"])
            # skip n_caps=1 delta (from run-start → first entry, can include setup overhead)
            sub = sub[sub["n_caps"] > 1]
            color = METHOD_COLORS[method]

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            else:
                grp = sub.groupby("n_caps")["delta_s"].agg(
                    mean="mean", std="std", n="count").reset_index()
                grp["ci"] = 1.96 * grp["std"] / np.sqrt(grp["n"].clip(lower=1))
                grp["ci"] = grp["ci"].fillna(0)

                ax.bar(grp["n_caps"], grp["mean"], color=color, alpha=0.75,
                       edgecolor="white", width=0.7, zorder=2)
                ax.errorbar(grp["n_caps"], grp["mean"], yerr=grp["ci"],
                            fmt="none", color="#333333", capsize=3, linewidth=1.2, zorder=3)

            if row_idx == 0:
                ax.set_title(f"Threshold = {thr} Ω", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{method}\nΔ Wall-Clock (s)", fontsize=10, fontweight="bold")
            if row_idx == len(METHODS) - 1:
                ax.set_xlabel("No. of Capacitors", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    return save_fig(fig, "RunGrids", "elapsed_time_per_cap_grid")

# ─── Fig 13: port_frequency ──────────────────────────────────────────────────

def fig_port_frequency(df_caps):
    if df_caps.empty or "placement" not in df_caps.columns:
        print("[WARN] Skipping port_frequency – no placement column"); return False
    port_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df_caps.iterrows():
        p = row["placement"]
        if not isinstance(p, dict): continue
        for port in p.keys():
            port_counts[row["method"]][int(port)] += 1
    if not any(port_counts.values()):
        print("[WARN] Skipping port_frequency – no placement data found"); return False

    total     = defaultdict(int)
    for mc in port_counts.values():
        for p, cnt in mc.items(): total[p] += cnt
    all_ports = sorted(total, key=total.get, reverse=True)[:40]
    all_ports = sorted(all_ports)
    n_m = len(METHODS); width = 0.8 / n_m; x = np.arange(len(all_ports))
    fig, ax = plt.subplots(figsize=(max(16, len(all_ports)*0.45), 6))
    for i, method in enumerate(METHODS):
        counts = [port_counts[method].get(p, 0) for p in all_ports]
        ax.bar(x + i*width, counts, width, label=method.capitalize(),
               color=METHOD_COLORS[method], alpha=0.82, edgecolor="white")
    ax.set_xticks(x + width*(n_m-1)/2)
    ax.set_xticklabels([str(p) for p in all_ports], rotation=60, ha="right", fontsize=9)
    ax.set_xlabel("Port Number"); ax.set_ylabel("Selection Count")
    ax.set_title("Port Selection Frequency by Method\n(All Runs, All Thresholds)",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Method", loc="upper right")
    fig.tight_layout()
    return save_fig(fig, "Frequencies", "port_frequency")

# ─── Fig 14: capacitor_frequency ─────────────────────────────────────────────

def fig_capacitor_frequency(df_caps):
    if df_caps.empty or "placement" not in df_caps.columns:
        print("[WARN] Skipping capacitor_frequency – no placement column"); return False
    cap_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df_caps.iterrows():
        p = row["placement"]
        if not isinstance(p, dict): continue
        for cap_id in p.values():
            cap_counts[row["method"]][int(cap_id)] += 1
    if not any(cap_counts.values()):
        print("[WARN] Skipping capacitor_frequency – no placement data found"); return False

    total  = defaultdict(int)
    for mc in cap_counts.values():
        for c, cnt in mc.items(): total[c] += cnt
    top30  = sorted(total, key=total.get, reverse=True)[:30]
    top30s = sorted(top30)
    n_m = len(METHODS); width = 0.8/n_m; x = np.arange(len(top30s))
    fig, ax = plt.subplots(figsize=(max(14, len(top30s)*0.55), 6))
    for i, method in enumerate(METHODS):
        counts = [cap_counts[method].get(c, 0) for c in top30s]
        ax.bar(x + i*width, counts, width, label=method.capitalize(),
               color=METHOD_COLORS[method], alpha=0.82, edgecolor="white")
    ax.set_xticks(x + width*(n_m-1)/2)
    ax.set_xticklabels([str(c) for c in top30s], rotation=60, ha="right", fontsize=9)
    ax.set_xlabel("Capacitor Model ID"); ax.set_ylabel("Selection Count")
    ax.set_title("Top 30 Most Selected Capacitor Models by Method\n(All Runs, All Thresholds)",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Method", loc="upper right")
    fig.tight_layout()
    return save_fig(fig, "Frequencies", "capacitor_frequency")

# ─── Fig 15: runtime_vs_threshold (CI95 bands) ───────────────────────────────

def fig_runtime_vs_threshold(df_runs):
    if df_runs.empty or "time_to_target" not in df_runs.columns:
        print("[WARN] Skipping runtime_vs_threshold – no data"); return False
    fig, ax = plt.subplots(figsize=(9, 6))
    _line_with_ci(ax, df_runs, "time_to_target")
    ax.set_xlabel("Target Impedance Threshold (Ω)", fontsize=13)
    ax.set_ylabel("Average Time to Target (s)", fontsize=13)
    ax.set_title("Runtime vs. Target Impedance\n(Shaded Band = 95% CI, Lower is Better)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "Scatter", "runtime_vs_threshold")

# ─── Fig 16: caps_vs_threshold (CI95) ────────────────────────────────────────

def fig_caps_vs_threshold(df_runs):
    if df_runs.empty or "caps_at_target" not in df_runs.columns:
        print("[WARN] Skipping caps_vs_threshold – no data"); return False
    fig, ax = plt.subplots(figsize=(9, 6))
    _line_with_ci(ax, df_runs, "caps_at_target")
    ax.set_xlabel("Target Impedance Threshold (Ω)", fontsize=13)
    ax.set_ylabel("Average Capacitors at Target", fontsize=13)
    ax.set_title("Capacitor Count vs. Target Impedance\n(Shaded Band = 95% CI)",
                 fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return save_fig(fig, "Scatter", "caps_vs_threshold")

# ─── Fig 17: best_minz_vs_threshold (CI95) ───────────────────────────────────

def fig_minz_vs_threshold(df_runs):
    if df_runs.empty or "best_minZ" not in df_runs.columns:
        print("[WARN] Skipping best_minz_vs_threshold – no data"); return False
    fig, ax = plt.subplots(figsize=(9, 6))
    _line_with_ci(ax, df_runs, "best_minZ")
    for thr in TARGETS:
        ax.axhline(thr, linestyle=":", linewidth=0.9, color="grey", alpha=0.6)
    ax.set_xlabel("Target Impedance Threshold (Ω)", fontsize=13)
    ax.set_ylabel("Average Best |Z₁₁| (Ω)", fontsize=13)
    ax.set_title("Best Achieved Impedance vs. Target\n(Shaded Band = 95% CI, Lower is Better)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "Scatter", "best_minz_vs_threshold")

# ─── Fig 18: speedup_vs_threshold  ──────────────────────────────────────────
# Fix 2: paired run-wise speedup  speedup_i = numpy_run_i / method_run_i

def fig_speedup_vs_threshold(df_runs):
    if df_runs.empty or "time_to_target" not in df_runs.columns:
        print("[WARN] Skipping speedup_vs_threshold – no data"); return False
    if "numpy" not in df_runs["method"].values:
        print("[WARN] Skipping speedup_vs_threshold – no numpy baseline"); return False

    fig, ax = plt.subplots(figsize=(9, 6))
    # x-axis follows TARGETS order (easy → hard), displayed right-to-left
    thr_order = TARGETS  # [0.05, 0.045, 0.04, 0.03]

    for method in METHODS:
        if method == "numpy":
            continue
        xs, ys, cis = [], [], []
        for thr in thr_order:
            np_sub = (df_runs[(df_runs["method"] == "numpy") &
                               (df_runs["threshold"] == thr)]
                      .dropna(subset=["time_to_target"])
                      .set_index("run_id")["time_to_target"])
            m_sub  = (df_runs[(df_runs["method"] == method) &
                               (df_runs["threshold"] == thr)]
                      .dropna(subset=["time_to_target"])
                      .set_index("run_id")["time_to_target"])
            # Keep only run_ids present in both
            common = np_sub.index.intersection(m_sub.index)
            if len(common) < 2:
                continue
            paired_speedup = np_sub.loc[common] / m_sub.loc[common]
            xs.append(thr)
            ys.append(paired_speedup.mean())
            cis.append(ci95(paired_speedup))

        if not xs:
            continue
        xs  = np.array(xs)
        ys  = np.array(ys)
        cis = np.array(cis)
        ax.plot(xs, ys, marker=METHOD_MARKERS[method], linewidth=2,
                color=METHOD_COLORS[method], label=method.capitalize(), markersize=8)
        ax.fill_between(xs, ys - cis, ys + cis,
                        alpha=0.18, color=METHOD_COLORS[method])

    ax.axhline(1.0, linestyle="--", color="#2196F3", linewidth=1.5,
               label="NumPy baseline (1×)", alpha=0.8)
    ax.set_xlabel("Target Impedance Threshold (Ω)", fontsize=13)
    ax.set_ylabel("Speedup vs NumPy (×)", fontsize=13)
    ax.set_title("Paired Speedup vs. Target Impedance\n"
                 "(Shaded Band = 95% CI of per-run ratios  |  Higher is Better)",
                 fontsize=14, fontweight="bold")
    ax.invert_xaxis()
    ax.set_xticks(thr_order)
    ax.legend(title="Method", loc="best")
    fig.tight_layout()
    return save_fig(fig, "Scatter", "speedup_vs_threshold")

# ─── Fig 19b: runtime_reduction_vs_threshold  (new) ─────────────────────────
# Fix 5: reduction_percent = 100 * (numpy_time - method_time) / numpy_time

def fig_runtime_reduction_vs_threshold(df_runs):
    if df_runs.empty or "time_to_target" not in df_runs.columns:
        print("[WARN] Skipping runtime_reduction_vs_threshold – no data"); return False
    if "numpy" not in df_runs["method"].values:
        print("[WARN] Skipping runtime_reduction_vs_threshold – no numpy baseline"); return False

    fig, ax = plt.subplots(figsize=(9, 6))
    thr_order = TARGETS  # [0.05, 0.045, 0.04, 0.03]

    for method in METHODS:
        if method == "numpy":
            continue
        xs, ys, cis = [], [], []
        for thr in thr_order:
            np_sub = (df_runs[(df_runs["method"] == "numpy") &
                               (df_runs["threshold"] == thr)]
                      .dropna(subset=["time_to_target"])
                      .set_index("run_id")["time_to_target"])
            m_sub  = (df_runs[(df_runs["method"] == method) &
                               (df_runs["threshold"] == thr)]
                      .dropna(subset=["time_to_target"])
                      .set_index("run_id")["time_to_target"])
            common = np_sub.index.intersection(m_sub.index)
            if len(common) < 2:
                continue
            # Paired per-run reduction %
            reduction = 100.0 * (np_sub.loc[common] - m_sub.loc[common]) / np_sub.loc[common]
            xs.append(thr)
            ys.append(reduction.mean())
            cis.append(ci95(reduction))

        if not xs:
            continue
        xs  = np.array(xs)
        ys  = np.array(ys)
        cis = np.array(cis)
        ax.plot(xs, ys, marker=METHOD_MARKERS[method], linewidth=2,
                color=METHOD_COLORS[method], label=method.capitalize(), markersize=8)
        ax.fill_between(xs, ys - cis, ys + cis,
                        alpha=0.18, color=METHOD_COLORS[method])

    ax.axhline(0.0, linestyle="--", color="#2196F3", linewidth=1.5,
               label="NumPy baseline (0%)", alpha=0.8)
    ax.set_xlabel("Target Impedance Threshold (Ω)", fontsize=13)
    ax.set_ylabel("Wall-Clock Time Reduction vs NumPy (%)", fontsize=13)
    ax.set_title("Runtime Reduction Relative to NumPy\n"
                 "(Shaded Band = 95% CI  |  Higher is Better)",
                 fontsize=14, fontweight="bold")
    ax.invert_xaxis()
    ax.set_xticks(thr_order)
    ax.legend(title="Method", loc="best")
    fig.tight_layout()
    return save_fig(fig, "Scatter", "runtime_reduction_vs_threshold")

# ─── Fig 20: pareto_runtime_quality  ─────────────────────────────────────────

def fig_pareto_runtime_quality(df_runs):
    if df_runs.empty:
        print("[WARN] Skipping pareto_runtime_quality – no data"); return False

    fig, ax = plt.subplots(figsize=(10, 7))

    for method in METHODS:
        for thr in TARGETS:
            sub = df_runs[(df_runs["method"] == method) &
                          (df_runs["threshold"] == thr)
                          ].dropna(subset=["time_to_target", "best_minZ"])
            if sub.empty: continue
            mx = sub["time_to_target"].mean()
            my = sub["best_minZ"].mean()
            ex = ci95(sub["time_to_target"])
            ey = ci95(sub["best_minZ"])
            ax.errorbar(mx, my, xerr=ex, yerr=ey,
                        fmt=THRESHOLD_MARKERS[thr],
                        color=METHOD_COLORS[method],
                        markersize=11, alpha=0.88,
                        capsize=4, linewidth=1.4,
                        ecolor=METHOD_COLORS[method])
            ax.annotate(f"{thr}", xy=(mx, my),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color=METHOD_COLORS[method], alpha=0.8)

    method_patches = [mpatches.Patch(color=METHOD_COLORS[m], label=m.capitalize())
                      for m in METHODS]
    thr_handles = [plt.Line2D([0],[0], marker=THRESHOLD_MARKERS[t], color="grey",
                               linestyle="None", markersize=9, label=f"Thr={t}")
                   for t in TARGETS]
    leg1 = ax.legend(handles=method_patches, title="Method",
                     loc="upper right", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=thr_handles, title="Threshold (Ω)",
              loc="upper left", framealpha=0.9)

    for thr in TARGETS:
        ax.axhline(thr, linestyle=":", linewidth=0.8, color="grey", alpha=0.45)

    ax.set_xlabel("Average Time to Target (s)", fontsize=13)
    ax.set_ylabel("Average Best |Z₁₁| (Ω)", fontsize=13)
    ax.set_title("Pareto Front: Runtime vs Optimization Quality\n"
                 "(Error bars = 95% CI  |  Lower-left = Pareto optimal)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "Scatter", "pareto_runtime_quality")

# ─── Fig 20: summary_dashboard ───────────────────────────────────────────────

def fig_summary_dashboard(df_runs):
    if df_runs.empty: return False
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("PSO–PDN Benchmark: Summary Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    x     = np.arange(len(TARGETS))
    width = 0.8 / len(METHODS)

    # A: avg runtime
    ax = axes[0][0]
    for i, method in enumerate(METHODS):
        vals = [df_runs[(df_runs["method"]==method)&(df_runs["threshold"]==t)]
                ["time_to_target"].mean() for t in TARGETS]
        ax.bar(x+i*width, vals, width, label=method.capitalize(),
               color=METHOD_COLORS[method], alpha=0.82, edgecolor="white")
    ax.set_xticks(x+width*(len(METHODS)-1)/2)
    ax.set_xticklabels([str(t) for t in TARGETS])
    ax.set_xlabel("Threshold (Ω)"); ax.set_ylabel("Avg. Time (s)")
    ax.set_title("A)  Average Runtime", fontweight="bold"); ax.legend(fontsize=9)

    # B: speedup
    ax = axes[0][1]
    other_methods = [m for m in METHODS if m != "numpy"]
    for i, method in enumerate(other_methods):
        vals = []
        for t in TARGETS:
            np_t = df_runs[(df_runs["method"]=="numpy")&(df_runs["threshold"]==t)
                           ]["time_to_target"].mean()
            m_t  = df_runs[(df_runs["method"]==method)&(df_runs["threshold"]==t)
                           ]["time_to_target"].mean()
            vals.append(np_t/m_t if (m_t and m_t > 0) else np.nan)
        ax.bar(x+i*width, vals, width, label=method.capitalize(),
               color=METHOD_COLORS[method], alpha=0.82, edgecolor="white")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.2, label="NumPy (1×)")
    ax.set_xticks(x+width)
    ax.set_xticklabels([str(t) for t in TARGETS])
    ax.set_xlabel("Threshold (Ω)"); ax.set_ylabel("Speedup (×)")
    ax.set_title("B)  Speedup vs. NumPy", fontweight="bold"); ax.legend(fontsize=9)

    # C: avg caps
    ax = axes[1][0]
    for i, method in enumerate(METHODS):
        vals = [df_runs[(df_runs["method"]==method)&(df_runs["threshold"]==t)]
                ["caps_at_target"].mean() for t in TARGETS]
        ax.bar(x+i*width, vals, width, label=method.capitalize(),
               color=METHOD_COLORS[method], alpha=0.82, edgecolor="white")
    ax.set_xticks(x+width*(len(METHODS)-1)/2)
    ax.set_xticklabels([str(t) for t in TARGETS])
    ax.set_xlabel("Threshold (Ω)"); ax.set_ylabel("Avg. Capacitors")
    ax.set_title("C)  Average Capacitor Count", fontweight="bold"); ax.legend(fontsize=9)

    # D: best minZ
    ax = axes[1][1]
    for i, method in enumerate(METHODS):
        vals = [df_runs[(df_runs["method"]==method)&(df_runs["threshold"]==t)]
                ["best_minZ"].mean() for t in TARGETS]
        ax.bar(x+i*width, vals, width, label=method.capitalize(),
               color=METHOD_COLORS[method], alpha=0.82, edgecolor="white")
    ax.set_xticks(x+width*(len(METHODS)-1)/2)
    ax.set_xticklabels([str(t) for t in TARGETS])
    ax.set_xlabel("Threshold (Ω)"); ax.set_ylabel("Avg. Best |Z₁₁| (Ω)")
    ax.set_title("D)  Average Best Impedance Achieved", fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    return save_fig(fig, "Heatmaps", "summary_dashboard")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  AnalyzeScratchBench.py  –  PDN PSO Benchmark Analysis")
    print("  Supports old format (no iter/placement/best_minZ) and new.")
    print("=" * 65)

    if not ROOT_OUT.exists():
        print(f"[ERROR] Root not found: {ROOT_OUT.resolve()}")
        sys.exit(1)

    create_output_dirs()

    df_runs, df_caps, log_count, has_placement = collect_all_logs()

    export_csvs(df_runs, df_caps)

    # ── guarded wrapper: counts only if fn returns True ──────────────────────
    plot_count = 0
    def guarded(fn, *args):
        nonlocal plot_count
        try:
            result = fn(*args)
            if result is True:
                plot_count += 1
        except Exception as exc:
            print(f"[WARN] {fn.__name__} failed: {exc}")
            traceback.print_exc()

    print("\n── Heatmaps ────────────────────────────────────────────────────")
    guarded(fig_runtime_heatmap,        df_runs)
    guarded(fig_speedup_heatmap,        df_runs)
    guarded(fig_caps_heatmap,           df_runs)
    guarded(fig_minz_heatmap,           df_runs)
    guarded(fig_success_rate_heatmap,   df_runs)
    guarded(fig_summary_dashboard,      df_runs)

    print("\n── Boxplots ────────────────────────────────────────────────────")
    guarded(fig_runtime_boxplots,       df_runs)
    guarded(fig_caps_boxplots,          df_runs)

    print("\n── Scatter / Line Plots ────────────────────────────────────────")
    guarded(fig_tradeoff_scatter,       df_runs)
    guarded(fig_runtime_vs_threshold,   df_runs)
    guarded(fig_caps_vs_threshold,      df_runs)
    guarded(fig_minz_vs_threshold,      df_runs)
    guarded(fig_speedup_vs_threshold,         df_runs)
    guarded(fig_runtime_reduction_vs_threshold, df_runs)
    guarded(fig_pareto_runtime_quality,         df_runs)

    print("\n── Run / Convergence Grids ─────────────────────────────────────")
    guarded(fig_run_grid,                      df_caps)
    guarded(fig_best_so_far_grid,              df_caps)
    guarded(fig_runtime_accumulation_grid,     df_caps)
    guarded(fig_runtime_per_cap_grid,          df_caps)

    print("\n── Frequency Charts (placement data required) ──────────────────")
    if has_placement:
        guarded(fig_port_frequency,      df_caps)
        guarded(fig_capacitor_frequency, df_caps)
    else:
        print("[WARN] Skipping port_frequency      – old log format, no placement data")
        print("[WARN] Skipping capacitor_frequency – old log format, no placement data")

    print("\n" + "=" * 65)
    print(f"  Total logs found   : {log_count}")
    print(f"  Runs parsed        : {len(df_runs)}")
    print(f"  Cap-rows parsed    : {len(df_caps)}")
    print(f"  Placement data     : {'YES' if has_placement else 'NO'}")
    print(f"  Plots generated    : {plot_count}  (×2 = PNG + PDF)")
    print(f"  Output directory   : {ANALYSIS.resolve()}")
    print("=" * 65)
    print("Done.")

if __name__ == "__main__":
    main()