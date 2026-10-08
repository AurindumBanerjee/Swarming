"""Publication plots for matrix-evaluation performance.

This module uses only the supplied summary artifacts. It deliberately does not
plot optimization-quality metrics such as impedance, capacitor count, or
success rate.

Matrix-evaluation count is inferred from the run-level stage counters:
    particles * ((completed_stages - 1) * iterations + final_stage_iterations)

The default particle and iteration counts match the benchmark configuration.
Because the supplied CSV does not contain per-particle early-stop counts,
this is an explicitly documented estimate of matrix evaluations. Missing
Pure-Python runtimes are estimated from measured Pure-Python time/matrix,
preferably at the same threshold, and are marked in the detailed CSV.

Usage:
    python plot_matrix_evaluation_speedup.py
    python plot_matrix_evaluation_speedup.py \
        --input-dir MinTime/ScratchBenchBaseline/summary \
        --output-dir MinTime/ScratchBenchBaseline/matrix_speedup
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["numpy", "solve", "sm", "iterative"]
INCLUDED_THRESHOLDS = (0.05, 0.045, 0.04)
METHOD_LABELS = {
    "numpy": "NumPy",
    "solve": "Proposed",
    "sm": "SM",
    "iterative": "Iterative",
}
METHOD_COLORS = {
    "numpy": "#1769AA",
    "solve": "#238636",
    "sm": "#C77700",
    "iterative": "#B4236B",
}


def ci95(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    return 1.96 * values.std(ddof=1) / math.sqrt(len(values))


def read_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    runs_path = input_dir / "runs.csv"
    summary_path = input_dir / "summary.csv"
    report_path = input_dir / "summary.txt"
    missing = [str(path) for path in (runs_path, summary_path, report_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))

    runs = pd.read_csv(runs_path)
    summary = pd.read_csv(summary_path)
    report = report_path.read_text(encoding="utf-8")
    required = {
        "threshold", "run_id", "method", "time_to_target_s",
        "stages_completed", "final_stage_iterations",
    }
    missing_columns = sorted(required - set(runs.columns))
    if missing_columns:
        raise ValueError(f"runs.csv is missing required columns: {missing_columns}")
    if not {"method", "threshold"}.issubset(summary.columns):
        raise ValueError("summary.csv must contain method and threshold columns")
    runs["threshold"] = pd.to_numeric(runs["threshold"], errors="coerce")
    summary["threshold"] = pd.to_numeric(summary["threshold"], errors="coerce")
    runs = runs[runs["threshold"].isin(INCLUDED_THRESHOLDS)].copy()
    summary = summary[summary["threshold"].isin(INCLUDED_THRESHOLDS)].copy()
    return runs, summary, report


def infer_matrix_counts(
    runs: pd.DataFrame,
    particles: int,
    iterations: int,
) -> pd.DataFrame:
    result = runs.copy()
    result["stages_completed"] = pd.to_numeric(result["stages_completed"], errors="coerce")
    result["final_stage_iterations"] = pd.to_numeric(
        result["final_stage_iterations"], errors="coerce"
    )
    completed = result["stages_completed"].fillna(0).clip(lower=0)
    final_iterations = result["final_stage_iterations"].fillna(iterations).clip(
        lower=0, upper=iterations
    )
    full_stages = (completed - 1).clip(lower=0)
    result["matrix_count"] = (
        particles * (full_stages * iterations + final_iterations)
    ).astype("Int64")
    result.loc[result["matrix_count"] <= 0, "matrix_count"] = pd.NA
    result["runtime_s"] = pd.to_numeric(result["time_to_target_s"], errors="coerce")
    result.loc[result["runtime_s"] <= 0, "runtime_s"] = np.nan
    result["time_per_matrix_s"] = result["runtime_s"] / result["matrix_count"].astype(float)
    return result


def add_baseline_and_speedups(runs: pd.DataFrame) -> pd.DataFrame:
    result = runs.copy()
    result["baseline_source"] = "not_applicable"
    result["pure_python_runtime_s"] = np.nan
    result["pure_python_time_per_matrix_s"] = np.nan

    pure = result[result["method"] == "pure_python"].copy()
    measured_pure = pure[pure["time_per_matrix_s"].notna()]
    threshold_means = measured_pure.groupby("threshold")["time_per_matrix_s"].mean()
    global_mean = measured_pure["time_per_matrix_s"].mean()

    baseline_lookup = pure.set_index(["threshold", "run_id"])[
        ["runtime_s", "time_per_matrix_s"]
    ].to_dict("index")
    for index, row in result.iterrows():
        if row["method"] == "pure_python":
            continue
        baseline = baseline_lookup.get((row["threshold"], row["run_id"]))
        if baseline and pd.notna(baseline["runtime_s"]):
            result.at[index, "pure_python_runtime_s"] = baseline["runtime_s"]
            result.at[index, "pure_python_time_per_matrix_s"] = baseline["time_per_matrix_s"]
            result.at[index, "baseline_source"] = "measured"
            continue

        per_matrix = threshold_means.get(row["threshold"], global_mean)
        if pd.notna(per_matrix) and pd.notna(row["matrix_count"]):
            result.at[index, "pure_python_runtime_s"] = per_matrix * row["matrix_count"]
            result.at[index, "pure_python_time_per_matrix_s"] = per_matrix
            result.at[index, "baseline_source"] = "estimated"

    result["speedup_vs_pure_python"] = (
        result["pure_python_time_per_matrix_s"] / result["time_per_matrix_s"]
    )
    result.loc[result["method"] == "pure_python", "speedup_vs_pure_python"] = np.nan
    result["speedup_baseline_measured"] = result["baseline_source"].eq("measured")
    result["speedup_baseline_estimated"] = result["baseline_source"].eq("estimated")
    return result


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.1)


def plot_average_speedup(data: pd.DataFrame, output: Path) -> None:
    grouped = data[data["method"].isin(METHODS)].groupby("method")["speedup_vs_pure_python"]
    means = grouped.mean().reindex(METHODS)
    errors = grouped.apply(ci95).reindex(METHODS).fillna(0)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    x = np.arange(len(METHODS))
    ax.bar(
        x, means.values, yerr=errors.values, capsize=5, color=[METHOD_COLORS[m] for m in METHODS],
        edgecolor="black", linewidth=0.7, error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("Matrix-evaluation speedup vs Pure Python")
    ax.set_title("Average Matrix-Evaluation Speedup vs Pure Python")
    ax.text(0.99, 0.02, "Bars: mean; whiskers: 95% CI", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555555")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output / "average_speedup_vs_pure_python.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_vs_threshold(data: pd.DataFrame, output: Path) -> None:
    summary = (
        data[data["method"].isin(METHODS)]
        .groupby(["threshold", "method"])["speedup_vs_pure_python"]
        .agg(mean="mean", ci95=ci95)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for method in METHODS:
        group = summary[summary["method"] == method].sort_values("threshold")
        if group.empty:
            continue
        x = group["threshold"].to_numpy()
        mean = group["mean"].to_numpy()
        ci = group["ci95"].to_numpy()
        ax.plot(x, mean, marker="o", linewidth=2.1, markersize=5.5,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, mean - ci, mean + ci, color=METHOD_COLORS[method], alpha=0.14)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.invert_xaxis()
    ax.set_xlabel("Target impedance threshold (Ohm)")
    ax.set_ylabel("Matrix-evaluation speedup vs Pure Python")
    ax.set_title("Matrix-Evaluation Speedup vs Target Threshold")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output / "speedup_vs_threshold.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(data: pd.DataFrame, output: Path) -> None:
    matrix = data[data["method"].isin(METHODS)].pivot_table(
        index="method", columns="threshold", values="speedup_vs_pure_python", aggfunc="mean"
    ).reindex(METHODS)
    matrix = matrix.sort_index(axis=1, ascending=False)
    values = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(matrix.columns)), [f"{v:.3f}" for v in matrix.columns])
    ax.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
    ax.set_xlabel("Target impedance threshold (Ohm)")
    ax.set_ylabel("Method")
    ax.set_title("Average Matrix-Evaluation Speedup vs Pure Python")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                color = "white" if value > np.nanmean(values) else "black"
                ax.text(column, row, f"{value:.2f}x", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label="Speedup (x)", pad=0.03)
    fig.tight_layout()
    fig.savefig(output / "speedup_heatmap.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("MinTime/ScratchBenchBaseline/summary"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--particles", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    if args.particles <= 0 or args.iterations <= 0:
        parser.error("--particles and --iterations must be positive")

    runs, summary, report = read_inputs(args.input_dir)
    del summary, report  # Loaded to validate the complete supplied input set.
    data = infer_matrix_counts(runs, args.particles, args.iterations)
    data = add_baseline_and_speedups(data)
    output = (args.output_dir or args.input_dir / "matrix_speedup").resolve()
    output.mkdir(parents=True, exist_ok=True)

    detailed_columns = [
        "threshold", "run_id", "method", "matrix_count", "runtime_s",
        "time_per_matrix_s", "pure_python_runtime_s", "pure_python_time_per_matrix_s",
        "speedup_vs_pure_python", "baseline_source", "speedup_baseline_measured",
        "speedup_baseline_estimated",
    ]
    data[detailed_columns].sort_values(["threshold", "method", "run_id"]).to_csv(
        output / "matrix_evaluation_speedup_runs.csv", index=False
    )
    plot_data = data[data["method"].isin(METHODS)].copy()
    plot_average_speedup(plot_data, output)
    plot_speedup_vs_threshold(plot_data, output)
    plot_heatmap(plot_data, output)

    measured = int(data["baseline_source"].eq("measured").sum())
    estimated = int(data["baseline_source"].eq("estimated").sum())
    print(f"Wrote matrix-evaluation run data to {output / 'matrix_evaluation_speedup_runs.csv'}")
    print(f"Pure-Python baselines used: {measured} measured, {estimated} estimated")
    print(f"Wrote figures to {output}")


if __name__ == "__main__":
    main()
