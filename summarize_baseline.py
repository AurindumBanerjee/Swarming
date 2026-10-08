"""Summarize MinTime/ScratchBenchBaseline benchmark results.

Usage:
    python summarize_baseline.py
    python summarize_baseline.py --root MinTime/ScratchBenchBaseline --output MinTime/ScratchBenchBaseline/summary

The script reads every method/threshold folder and every run log below it.
It writes:
    runs.csv       one row per run, including failed runs
    summary.csv    one row per method/threshold group
    summary.txt    human-readable report
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


GROUP_RE = re.compile(r"^(?P<method>.+)__THR__(?P<threshold>[-+]?\d*\.?\d+)$")
RUN_RE = re.compile(r"^run_(?P<run_id>\d+)$")
STAGE_RE = re.compile(
    r"n_caps=(?P<caps>\d+)\s+\|\s+iter=(?P<iterations>\d+)\s+\|\s+"
    r"minZ=(?P<minz>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
RESULT_RE = re.compile(
    r"RESULT\s+\|\s+time_to_target=(?P<time>[-+]?\d*\.?\d+)\s+\|\s+"
    r"caps=(?P<caps>-?\d+)\s+\|\s+best_minZ=(?P<minz>[-+]?\d*\.?\d+)\s+"
    r"\|\s+success=(?P<success>True|False)"
)

RUN_FIELDS = [
    "method", "threshold", "run_id", "folder", "log_file", "status",
    "success", "time_to_target_s", "caps_at_target", "best_minz",
    "stages_completed", "final_stage_iterations", "run_blocks_in_log",
]
SUMMARY_FIELDS = [
    "method", "threshold", "runs_found", "runs_with_results", "successes",
    "failures_or_timeouts", "success_rate", "time_mean_s", "time_median_s",
    "time_min_s", "time_max_s", "caps_mean", "caps_min", "caps_max",
    "best_minz_mean", "best_minz_min", "best_minz_max",
]


def parse_log(log_path: Path) -> dict:
    """Parse the final result and stage metrics from one run log."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # A log file may hold several appended runs (re-launched sweeps append to the
    # same file). Only the LAST run is used, so that the stage counters and the
    # RESULT line always describe the same run.
    parts = text.split("RUN_START")
    run_blocks = len(parts) - 1
    if run_blocks >= 1:
        text = "RUN_START" + parts[-1]
    stages = list(STAGE_RE.finditer(text))
    results = list(RESULT_RE.finditer(text))
    record = {
        "log_file": str(log_path),
        "status": "missing_result",
        "success": False,
        "time_to_target_s": "",
        "caps_at_target": "",
        "best_minz": "",
        "stages_completed": len(stages),
        "final_stage_iterations": stages[-1].group("iterations") if stages else "",
        "run_blocks_in_log": run_blocks,
    }
    if not results:
        return record

    result = results[-1]
    time_to_target = float(result.group("time"))
    caps_at_target = int(result.group("caps"))
    best_minz = float(result.group("minz"))
    success = result.group("success") == "True"
    record.update(
        status="success" if success else "failure_or_timeout",
        success=success,
        time_to_target_s=time_to_target if success and time_to_target >= 0 else "",
        caps_at_target=caps_at_target if success and caps_at_target >= 0 else "",
        best_minz=best_minz,
    )
    return record


def discover_runs(root: Path) -> list[dict]:
    records = []
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        group_match = GROUP_RE.match(group_dir.name)
        if not group_match:
            continue
        method = group_match.group("method")
        threshold = float(group_match.group("threshold"))
        for run_dir in sorted(group_dir.iterdir()):
            if not run_dir.is_dir() or not RUN_RE.match(run_dir.name):
                continue
            run_id = int(RUN_RE.match(run_dir.name).group("run_id"))
            log_path = run_dir / f"run_{run_id}.log"
            record = parse_log(log_path) if log_path.exists() else {
                "log_file": str(log_path),
                "status": "missing_log",
                "success": False,
                "time_to_target_s": "",
                "caps_at_target": "",
                "best_minz": "",
                "stages_completed": 0,
                "final_stage_iterations": "",
                "run_blocks_in_log": 0,
            }
            records.append({
                "method": method,
                "threshold": threshold,
                "run_id": run_id,
                "folder": str(run_dir),
                **record,
            })
    return records


def mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def median(values: list[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def fmt(value) -> str:
    if value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_summary(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["threshold"])].append(record)

    summary = []
    for (method, threshold), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        successful = [r for r in group if r["success"]]
        times = [r["time_to_target_s"] for r in successful if r["time_to_target_s"] != ""]
        caps = [r["caps_at_target"] for r in successful if r["caps_at_target"] != ""]
        minz = [r["best_minz"] for r in group if r["best_minz"] != ""]
        summary.append({
            "method": method,
            "threshold": threshold,
            "runs_found": len(group),
            "runs_with_results": sum(r["status"] != "missing_log" and r["status"] != "missing_result" for r in group),
            "successes": len(successful),
            "failures_or_timeouts": len(group) - len(successful),
            "success_rate": len(successful) / len(group) if group else "",
            "time_mean_s": mean(times),
            "time_median_s": median(times),
            "time_min_s": min(times) if times else "",
            "time_max_s": max(times) if times else "",
            "caps_mean": mean(caps),
            "caps_min": min(caps) if caps else "",
            "caps_max": max(caps) if caps else "",
            "best_minz_mean": mean(minz),
            "best_minz_min": min(minz) if minz else "",
            "best_minz_max": max(minz) if minz else "",
        })
    return summary


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: list[dict], records: list[dict], root: Path) -> None:
    lines = [f"Baseline summary: {root}", f"Run records: {len(records)}", ""]
    for row in summary:
        lines.append(
            f"{row['method']} | threshold={row['threshold']:.6g} | "
            f"runs={row['runs_found']} | success={row['successes']}/{row['runs_found']} "
            f"({row['success_rate']:.1%}) | "
            f"time_s mean/median/min/max={fmt(row['time_mean_s'])}/"
            f"{fmt(row['time_median_s'])}/{fmt(row['time_min_s'])}/{fmt(row['time_max_s'])} | "
            f"caps mean/min/max={fmt(row['caps_mean'])}/{fmt(row['caps_min'])}/{fmt(row['caps_max'])} | "
            f"best_minz mean/min/max={fmt(row['best_minz_mean'])}/"
            f"{fmt(row['best_minz_min'])}/{fmt(row['best_minz_max'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("MinTime/ScratchBenchBaseline"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"baseline folder does not exist: {root}")
    output = (args.output or root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = discover_runs(root)
    summary = build_summary(records)
    write_csv(output / "runs.csv", records, RUN_FIELDS)
    write_csv(output / "summary.csv", summary, SUMMARY_FIELDS)
    write_report(output / "summary.txt", summary, records, root)

    print(f"Scanned {len(records)} run folders in {root}")
    print(f"Wrote {output / 'runs.csv'}")
    print(f"Wrote {output / 'summary.csv'}")
    print(f"Wrote {output / 'summary.txt'}")


if __name__ == "__main__":
    main()
