"""IEEE-style matrix-evaluation speedup figures (regenerated from run logs).

Pipeline:  summarize_baseline.py  ->  runs.csv  ->  this script.

    python summarize_baseline.py --root "<baseline>" --output "<baseline>/summary"
    python plot_ieee_figures.py  --runs "<baseline>/summary/runs.csv" --out PSO_LAPACK_MAPCON/figs_v2

Definitions (match the paper):
    N_matrix  = particles * ((stages-1) * iterations + final_stage_iterations)
                (fitness calls of the LAST run in each log; an upper bound of one
                iteration's early-stop remainder)
    t_matrix  = t_run / N_matrix
    S         = t_matrix(pure Python) / t_matrix(method), paired per (threshold, run)
Runs that failed to reach the target have no logged time and are dropped, for
the method AND its pure-Python partner (no estimated baselines).
Only Z_T = 50, 45, 40 mOhm are used.  95% CI: Student-t on per-run S.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import stats

from plot_matrix_evaluation_speedup import infer_matrix_counts

THRESHOLDS = [0.050, 0.045, 0.040]
TLAB = ["0.050", "0.045", "0.040"]             # ohm (plotter.py format)
ORDER = ["solve", "numpy", "iterative", "sm"]
LABEL = {"solve": "Proposed", "numpy": "NumPy", "iterative": "Iterative", "sm": "Sherman–Morrison"}
BARLAB = {"solve": "Proposed", "numpy": "NumPy", "iterative": "Iterative", "sm": "Sherman–\nMorrison"}
# Okabe-Ito (colour-blind safe); markers/hatches keep it readable in greyscale
COL = {"solve": "#009E73", "numpy": "#0072B2", "iterative": "#CC3B7A", "sm": "#E69F00"}   # colours from plotter.py
MRK = {"solve": "s", "numpy": "o", "iterative": "D", "sm": "^"}   # as plotter.py
LST = {"solve": "-", "numpy": "--", "iterative": "-.", "sm": ":"}
HAT = {"solve": "", "numpy": "////", "iterative": "\\\\\\\\", "sm": "xxxx"}

W = 3.45          # IEEE single-column width (in)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "lines.linewidth": 1.1, "lines.markersize": 4,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "axes.axisbelow": True,
    "savefig.dpi": 600, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def ci_t(x):
    x = np.asarray(x, float); n = len(x)
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def build(runs_csv: Path, particles=50, iterations=15) -> pd.DataFrame:
    runs = pd.read_csv(runs_csv)
    runs = runs[runs["threshold"].round(4).isin([round(t, 4) for t in THRESHOLDS])].copy()
    d = infer_matrix_counts(runs, particles, iterations)
    d["t_matrix"] = d["runtime_s"] / d["matrix_count"].astype(float)
    pp = d[d.method == "pure_python"].set_index(["threshold", "run_id"])["t_matrix"].rename("t_pp")
    d = d[d.method.isin(ORDER)].join(pp, on=["threshold", "run_id"])
    d["S"] = d["t_pp"] / d["t_matrix"]
    return d


def save(fig, out: Path, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_bar(d, out):
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for i, m in enumerate(ORDER):
        v = d.loc[(d.method == m), "S"].dropna().values
        mu, ci = v.mean(), ci_t(v)
        ax.bar(i, mu, 0.62, yerr=ci, capsize=2.5, color=COL[m], hatch=HAT[m], edgecolor="black",
               linewidth=0.5, error_kw=dict(elinewidth=0.7, capthick=0.7), zorder=3)
        ax.text(i, mu + ci + 2.0, f"{mu:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([BARLAB[m] for m in ORDER])
    ax.set_ylabel("Speedup per matrix evaluation, $S$ (×)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08); ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=2.5, direction="in")
    save(fig, out, "matrix_speedup_avg_bar")


def fig_line(d, out):
    fig, ax = plt.subplots(figsize=(W, 2.5))
    x = np.arange(len(THRESHOLDS))
    for m in ORDER:
        mu, ci = [], []
        for t in THRESHOLDS:
            v = d.loc[(d.method == m) & (np.isclose(d.threshold, t)), "S"].dropna().values
            mu.append(v.mean()); ci.append(ci_t(v))
        mu, ci = np.array(mu), np.array(ci)
        ax.plot(x, mu, LST[m], marker=MRK[m], color=COL[m], label=LABEL[m], markeredgecolor="white",
                markeredgewidth=0.4, zorder=3)
        ax.fill_between(x, mu - ci, mu + ci, color=COL[m], alpha=0.18, linewidth=0, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(TLAB)
    ax.set_xlim(-0.15, len(THRESHOLDS) - 0.85)
    ax.set_xlabel(r"Target impedance threshold ($\Omega$)")
    ax.set_ylabel("Speedup per matrix evaluation, $S$ (×)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95, edgecolor="0.6",
              fancybox=False, borderpad=0.35, handlelength=2.2, columnspacing=1.0)
    save(fig, out, "matrix_speedup_vs_threshold")


def fig_heat(d, out):
    Z = np.array([[d.loc[(d.method == m) & np.isclose(d.threshold, t), "S"].dropna().mean()
                   for t in THRESHOLDS] for m in ORDER])
    fig, ax = plt.subplots(figsize=(W, 2.0))
    im = ax.imshow(Z, aspect="auto", cmap="Greens", norm=Normalize(0, Z.max() * 1.15))
    ax.set_xticks(range(3)); ax.set_xticklabels(TLAB)
    ax.set_yticks(range(4)); ax.set_yticklabels([LABEL[m] for m in ORDER])
    ax.set_xlabel(r"Target impedance threshold ($\Omega$)")
    ax.grid(False); ax.tick_params(length=2.5, direction="in", top=True, right=True, color="0.2")
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if Z[i, j] > 0.7 * Z.max() else "black")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("$S$ (×)", fontsize=8, labelpad=4); cb.ax.yaxis.label.set_rotation(270); cb.ax.yaxis.label.set_va("center"); cb.ax.yaxis.label.set_ha("center"); cb.ax.yaxis.set_label_coords(4.6, 0.5); cb.ax.tick_params(labelsize=7, width=0.5, length=2, direction="in")
    cb.outline.set_linewidth(0.5)
    save(fig, out, "matrix_speedup_heatmap")


def report(d, out):
    L = ["Per-evaluation speedup S (paired per run, failed runs dropped), Student-t 95% CI", ""]
    for m in ORDER:
        row = []
        for t, tl in zip(THRESHOLDS, TLAB):
            v = d.loc[(d.method == m) & np.isclose(d.threshold, t), "S"].dropna().values
            row.append(f"{tl} mΩ: {v.mean():7.2f} ±{ci_t(v):5.2f} (n={len(v)})")
        v = d.loc[d.method == m, "S"].dropna().values
        L.append(f"{LABEL[m]:17s} " + " | ".join(row) + f" | pooled: {v.mean():7.2f} ±{ci_t(v):5.2f} (n={len(v)})"
                 f" | mean of 3: {np.mean([d.loc[(d.method==m)&np.isclose(d.threshold,t),'S'].dropna().mean() for t in THRESHOLDS]):.2f}")
    (out / "speedup_stats.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    d.sort_values(["threshold", "method", "run_id"])[
        ["threshold", "run_id", "method", "matrix_count", "runtime_s", "t_matrix", "t_pp", "S"]
    ].to_csv(out / "matrix_evaluation_speedup_runs.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = build(a.runs)
    fig_bar(data, a.out); fig_line(data, a.out); fig_heat(data, a.out); report(data, a.out)
    print("figures in", a.out)
