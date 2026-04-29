"""Utilities for summarizing repeated publication experiments."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _round(value: float) -> float:
    return round(value, 6)


def aggregate_runs(
    runs: Sequence[Mapping],
    metrics: Sequence[str] = ("accuracy", "macro_f1", "macro_recall"),
) -> list[dict]:
    """Aggregate per-seed experiment runs into publication-ready rows."""
    methods = sorted({method for run in runs for method in run.get("methods", {})})
    rows: list[dict] = []

    for method in methods:
        row = {"method": method, "n": 0}
        method_runs = [run["methods"][method] for run in runs if method in run.get("methods", {})]
        row["n"] = len(method_runs)

        for metric in metrics:
            values = [float(item[metric]) for item in method_runs if metric in item]
            if not values:
                continue
            row[f"{metric}_mean"] = _round(sum(values) / len(values))
            row[f"{metric}_std"] = _round(_sample_std(values))

        rows.append(row)

    return rows


def best_by_metric(rows: Sequence[Mapping], metric: str) -> Mapping:
    """Return the row with the highest value for a metric."""
    return max(rows, key=lambda row: row[metric])


def load_runs(path: str | Path) -> list[dict]:
    """Load repeated experiment summaries from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of experiment runs.")
    return data


def write_csv(rows: Iterable[Mapping], path: str | Path) -> None:
    """Write aggregate rows to CSV."""
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty result table.")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    runs = load_runs("results/repeated_runs.json")
    rows = aggregate_runs(runs)
    write_csv(rows, "results/repeated_runs_summary.csv")
    best = best_by_metric(rows, "macro_f1_mean")
    print(f"Best method by macro F1: {best['method']} ({best['macro_f1_mean']:.4f})")


if __name__ == "__main__":
    main()
