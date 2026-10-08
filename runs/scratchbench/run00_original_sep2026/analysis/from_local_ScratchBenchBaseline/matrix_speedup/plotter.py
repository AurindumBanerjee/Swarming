"""
plotter.py
==========

Regenerates the three matrix-evaluation speedup figures for the PDN
decoupling-capacitor paper directly from the measured run data in
``matrix_evaluation_speedup_runs.csv`` (Pure Python, NumPy, Proposed
["solve" in the CSV], SM, Iterative; thresholds 0.050 / 0.045 / 0.040 Ω;
0.030 Ω is excluded).

Styling is modeled on the reference paper-figure style: bold text
everywhere (titles, axis labels, tick labels, legend, in-cell
annotations), inward-facing y-axis ticks on the line plot, a single-hue
sequential heatmap colormap with linear normalization, and manual axes
placement for the heatmap so the colorbar never clips.

Produces (written next to this script):
    matrix_speedup_avg_bar.png     - average matrix-eval speedup vs Pure Python (bar + 95% CI)
    matrix_speedup_vs_threshold.png - matrix-eval speedup vs target impedance threshold (line + 95% CI)
    matrix_speedup_heatmap.png     - method x threshold speedup heatmap
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize

# ============================================================
# 1. LOAD + AGGREGATE DATA
# ============================================================

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0] if ("\\" in __file__ or "/" in __file__) else "."
CSV_PATH = f"{HERE}/matrix_evaluation_speedup_runs.csv"

df = pd.read_csv(CSV_PATH)
df = df[df["threshold"].isin([0.050, 0.045, 0.040])]
df = df[df["method"] != "pure_python"]
df = df.dropna(subset=["speedup_vs_pure_python"])

METHOD_KEY_TO_LABEL = {
    "numpy": "NumPy",
    "solve": "Proposed",
    "sm": "SM",
    "iterative": "Iterative",
}
df["method_label"] = df["method"].map(METHOD_KEY_TO_LABEL)

METHODS = ["NumPy", "Proposed", "SM", "Iterative"]
THRESHOLDS = [0.050, 0.045, 0.040]
THRESHOLD_LABELS = ["0.050", "0.045", "0.040"]

# mean +/- 95% CI (normal approx) of paired per-run speedup, per method,
# averaged over all three thresholds
def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    return m, 1.96 * se

AVG_SPEEDUP = {}
AVG_CI = {}
for m in METHODS:
    vals = df.loc[df["method_label"] == m, "speedup_vs_pure_python"].values
    mean, ci = mean_ci95(vals)
    AVG_SPEEDUP[m] = mean
    AVG_CI[m] = ci

# mean +/- 95% CI per (method, threshold) for the line plot / heatmap
SPEEDUP_BY_THRESH = {m: [] for m in METHODS}
CI_BY_THRESH = {m: [] for m in METHODS}
for m in METHODS:
    for t in THRESHOLDS:
        vals = df.loc[(df["method_label"] == m) & (df["threshold"] == t),
                       "speedup_vs_pure_python"].values
        mean, ci = mean_ci95(vals)
        SPEEDUP_BY_THRESH[m].append(mean)
        CI_BY_THRESH[m].append(ci)

# ============================================================
# 2. STYLE
# ============================================================

METHOD_COLORS = {
    "Proposed":  "#2E7D32",   # green
    "NumPy":     "#1E88E5",   # blue
    "SM":        "#F39C12",   # orange
    "Iterative": "#D6336C",   # pink/red
}

METHOD_MARKERS = {
    "Proposed":  "s",
    "NumPy":     "o",
    "SM":        "^",
    "Iterative": "D",
}

BOLD = "bold"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": BOLD,
    "axes.labelweight": BOLD,
    "axes.titleweight": BOLD,
    "figure.titleweight": BOLD,
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
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


def _inward_yaxis(ax, pad=-34, ha="left"):
    """Draw y-axis ticks pointing into the plot with inset bold labels."""
    ax.tick_params(axis="y", direction="in", length=6, width=1.4, pad=pad)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment(ha)
        label.set_fontweight(BOLD)
        label.set_zorder(5)
    ax.tick_params(axis="x", direction="out", length=5, width=1.2)


# ============================================================
# 3. FIGURE 1 — AVERAGE MATRIX-EVALUATION SPEEDUP (BAR + 95% CI)
# ============================================================

def fig_avg_speedup_bar(save_path):
    fig, ax = plt.subplots(figsize=(8.6, 6.4))

    xs = np.arange(len(METHODS))
    heights = [AVG_SPEEDUP[m] for m in METHODS]
    errs = [AVG_CI[m] for m in METHODS]
    colors = [METHOD_COLORS[m] for m in METHODS]

    bars = ax.bar(
        xs, heights,
        yerr=errs, capsize=8,
        color=colors, edgecolor="white", linewidth=1.4,
        width=0.62, zorder=3,
        error_kw=dict(elinewidth=2.0, ecolor="#333333", zorder=4),
    )

    for x, h, e in zip(xs, heights, errs):
        ax.text(
            x, h + e + max(heights) * 0.025, f"{h:.1f}×",
            ha="center", va="bottom", fontsize=14, fontweight=BOLD,
            color="#222222",
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.3, color="grey", alpha=0.7, zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels(METHODS, fontweight=BOLD)
    ax.set_ylabel("Matrix-Evaluation Speedup vs Pure Python (×)", fontweight=BOLD)
    ax.set_title(
        "Average Matrix-Evaluation Speedup vs Pure Python\n"
        "(Mean over 0.050 / 0.045 / 0.040 Ω Thresholds  |  Higher is Better)",
        fontweight=BOLD, fontsize=14.5,
    )

    ax.set_ylim(0, max(heights) * 1.22)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.tick_params(axis="y", direction="out", length=6, width=1.3)
    ax.tick_params(axis="x", length=0, pad=8)
    _bold_ticklabels(ax)
    ax.grid(True, axis="y", alpha=0.30, linewidth=0.8)
    ax.grid(False, axis="x")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[PLOT] {save_path}")


# ============================================================
# 4. FIGURE 2 — MATRIX-EVALUATION SPEEDUP VS THRESHOLD (LINE + CI)
# ============================================================

def fig_speedup_vs_threshold(save_path):
    fig, ax = plt.subplots(figsize=(9, 7.2))

    xs = np.array(THRESHOLDS)

    for m in METHODS:
        ys = np.array(SPEEDUP_BY_THRESH[m])
        ci = np.array(CI_BY_THRESH[m])

        ax.plot(
            xs, ys,
            marker=METHOD_MARKERS[m],
            color=METHOD_COLORS[m],
            linewidth=3,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.3,
            label=m,
            zorder=3,
        )
        ax.fill_between(
            xs, ys - ci, ys + ci,
            color=METHOD_COLORS[m],
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

    ax.invert_xaxis()
    ax.set_xticks(xs)
    ax.set_xticklabels(THRESHOLD_LABELS)
    ax.set_xlabel("Target Impedance Threshold (Ω)", fontweight=BOLD)
    ax.set_ylabel("Matrix-Evaluation Speedup vs\nPure Python (×)", fontweight=BOLD)
    ax.set_title(
        "Matrix-Evaluation Speedup vs. Target Impedance Threshold\n"
        "(Shaded Band = 95% CI  |  Higher is Better)",
        fontweight=BOLD,
    )

    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.tick_params(axis="both", direction="out", length=6, width=1.3)
    _bold_ticklabels(ax)
    ax.grid(True, which="major", alpha=0.30, linewidth=0.8)

    leg = ax.legend(
        title="Method", loc="upper right", frameon=True,
        framealpha=0.95, edgecolor="#888888", borderpad=0.9,
        handlelength=2.2, labelspacing=0.6,
    )
    leg.get_title().set_fontweight(BOLD)
    for text in leg.get_texts():
        text.set_fontweight(BOLD)
    leg.set_zorder(6)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {save_path}")


# ============================================================
# 5. FIGURE 3 — SPEEDUP HEATMAP (METHOD x THRESHOLD)
# ============================================================

def fig_speedup_heatmap(save_path):
    rows = METHODS
    cols = THRESHOLD_LABELS
    values = np.array([SPEEDUP_BY_THRESH[m] for m in rows])  # shape (4, 3)

    fig = plt.figure(figsize=(7.4, 6.0))
    ax  = fig.add_axes([0.20, 0.14, 0.62, 0.72])
    cax = fig.add_axes([0.85, 0.14, 0.035, 0.72])

    norm = Normalize(vmin=values.min() * 0.85, vmax=values.max() * 1.05)
    im = ax.imshow(values, aspect="auto", cmap="Greens", norm=norm)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c} Ω" for c in cols], fontweight=BOLD, fontsize=13)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontweight=BOLD, fontsize=13)

    ax.tick_params(axis="both", length=0, pad=8)

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.6)
    ax.grid(which="major", visible=False)

    cmap = im.get_cmap()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            r, g, b, _ = cmap(norm(val))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.6 else "black"
            ax.text(
                j, i, f"{val:.1f}×",
                ha="center", va="center",
                fontsize=13.5, fontweight=BOLD,
                color=text_color,
            )

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    cbar.set_label(
        "Matrix-Evaluation Speedup\nvs Pure Python (×)",
        fontweight=BOLD, fontsize=11.5, rotation=270, labelpad=26,
    )
    cbar.ax.tick_params(labelsize=11)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight(BOLD)

    ax.set_title(
        "Matrix-Evaluation Speedup Heatmap\n(Method vs. Target Impedance Threshold)",
        fontweight=BOLD, fontsize=14,
    )

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.35)
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"[PLOT] {save_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    fig_avg_speedup_bar(f"{HERE}/matrix_speedup_avg_bar.png")
    fig_speedup_vs_threshold(f"{HERE}/matrix_speedup_vs_threshold.png")
    fig_speedup_heatmap(f"{HERE}/matrix_speedup_heatmap.png")
    print("Done.")
