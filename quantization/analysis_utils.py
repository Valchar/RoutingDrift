"""
analysis_utils.py

Utilities for:
    - drift vs accuracy-drop correlation
    - layer drift heatmap generation
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def save_rows_csv(rows: list[dict], output_path: str | Path) -> None:
    """Write rows to CSV if rows are present."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pearson_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    x_var = sum((xi - x_mean) ** 2 for xi in x)
    y_var = sum((yi - y_mean) ** 2 for yi in y)
    denom = math.sqrt(x_var * y_var)
    if denom == 0.0:
        return None
    return numerator / denom


def _rankdata(values: list[float]) -> list[float]:
    """Average-tie ranks (1-indexed rank values)."""
    sorted_pairs = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            original_idx = sorted_pairs[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1

    return ranks


def _spearman_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    return _pearson_corr(x_rank, y_rank)


def build_drift_accuracy_rows(
    drift_rows: list[dict],
    eval_rows: list[dict],
    baseline_variant: str = "fp16",
    drift_metric_key: str = "jaccard_drift",
) -> list[dict]:
    """
    Join drift summaries and lm-eval results into task-level drift/drop points.
    """
    baseline_by_task: dict[str, float] = {
        str(row["task"]): float(row["accuracy"])
        for row in eval_rows
        if str(row["variant"]) == baseline_variant
    }
    drift_by_variant: dict[str, float] = {
        str(row["variant"]): float(row[drift_metric_key])
        for row in drift_rows
        if str(row["variant"]) != baseline_variant
    }

    rows: list[dict] = []
    for row in eval_rows:
        variant = str(row["variant"])
        task = str(row["task"])
        if variant == baseline_variant or task not in baseline_by_task or variant not in drift_by_variant:
            continue

        baseline_accuracy = baseline_by_task[task]
        accuracy = float(row["accuracy"])
        rows.append(
            {
                "variant": variant,
                "task": task,
                "drift": drift_by_variant[variant],
                "accuracy": accuracy,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_drop": baseline_accuracy - accuracy,
            }
        )

    return rows


def summarize_correlations(
    drift_accuracy_rows: list[dict],
) -> list[dict]:
    """Return Pearson/Spearman for all tasks and per-task subsets."""
    if not drift_accuracy_rows:
        return []

    tasks = sorted({str(row["task"]) for row in drift_accuracy_rows})
    groups: list[tuple[str, Iterable[dict]]] = [
        ("all", drift_accuracy_rows),
        *[(task, [row for row in drift_accuracy_rows if str(row["task"]) == task]) for task in tasks],
    ]

    summaries: list[dict] = []
    for group_name, group_rows_iter in groups:
        group_rows = list(group_rows_iter)
        x = [float(row["drift"]) for row in group_rows]
        y = [float(row["accuracy_drop"]) for row in group_rows]

        pearson = _pearson_corr(x, y)
        spearman = _spearman_corr(x, y)
        summaries.append(
            {
                "group": group_name,
                "n_points": len(group_rows),
                "pearson": pearson if pearson is not None else "",
                "spearman": spearman if spearman is not None else "",
            }
        )

    return summaries


def plot_layer_heatmap(
    layer_rows: list[dict],
    metric_key: str,
    output_path: str | Path,
    title: str,
) -> None:
    """
    Plot module x variant heatmap for the chosen drift metric.
    """
    if not layer_rows:
        return

    # Local import so plotting stays optional for environments without matplotlib.
    import matplotlib.pyplot as plt

    variants = sorted({str(row["variant"]) for row in layer_rows})
    modules = sorted({str(row["module"]) for row in layer_rows})
    if not variants or not modules:
        return

    matrix = np.full((len(modules), len(variants)), np.nan, dtype=np.float64)
    v_idx = {variant: idx for idx, variant in enumerate(variants)}
    m_idx = {module: idx for idx, module in enumerate(modules)}

    for row in layer_rows:
        module = str(row["module"])
        variant = str(row["variant"])
        if module in m_idx and variant in v_idx:
            matrix[m_idx[module], v_idx[variant]] = float(row[metric_key])

    fig_h = max(5.0, len(modules) * 0.24)
    fig_w = max(7.0, len(variants) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_key)

    ax.set_title(title)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=35, ha="right")
    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Router module")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def aggregate_quantization_metrics(
    summary_rows: list[dict],
    eval_rows: list[dict] | None = None,
    baseline_variant: str = "fp16",
) -> dict:
    """
    Aggregate quantization experiment metrics by variant.

    Args:
        summary_rows: Drift summary rows from quantization run (variant, precision, routing_similarity_rs, etc.)
        eval_rows: Optional lm-eval rows (variant, task, accuracy, metric)
        baseline_variant: Name of baseline variant (default "fp16")

    Returns:
        Dictionary with keys:
            - "by_variant": dict mapping variant -> {drift metrics + eval scores}
            - "summary": overall statistics (mean drift, accuracy drop across variants)
            - "baseline_variant": baseline variant name
    """
    metrics = {
        "by_variant": {},
        "summary": {},
        "baseline_variant": baseline_variant,
    }

    # Group by variant
    variant_data = {}
    for row in summary_rows:
        variant = row.get("variant", "unknown")
        if variant not in variant_data:
            variant_data[variant] = {
                "routing_similarity_rs": row.get("routing_similarity_rs", 1.0),
                "jaccard_drift": row.get("jaccard_drift", 0.0),
                "overlap_at_k": row.get("overlap_at_k", 1.0),
                "selection_shift": row.get("selection_shift", 0.0),
                "precision": row.get("precision", "unknown"),
                "compiler_mode": row.get("compiler_mode", "eager"),
                "variant_type": row.get("variant_type", "unknown"),
                "eval_scores": {},
            }

    # Merge eval scores if provided
    if eval_rows:
        for row in eval_rows:
            variant = row.get("variant")
            task = row.get("task")
            accuracy = row.get("accuracy")
            if variant in variant_data and task and accuracy is not None:
                variant_data[variant]["eval_scores"][task] = float(accuracy)

    metrics["by_variant"] = variant_data

    # Compute summary statistics
    drift_values = [v["jaccard_drift"] for v in variant_data.values() if v["variant_type"] == "quantization"]
    selection_shift_values = [v["selection_shift"] for v in variant_data.values() if v["variant_type"] == "quantization"]
    similarity_values = [v["routing_similarity_rs"] for v in variant_data.values() if v["variant_type"] == "quantization"]

    metrics["summary"] = {
        "num_variants": len(variant_data),
        "num_quantization_variants": sum(1 for v in variant_data.values() if v["variant_type"] == "quantization"),
        "mean_jaccard_drift": float(np.mean(drift_values)) if drift_values else 0.0,
        "max_jaccard_drift": float(np.max(drift_values)) if drift_values else 0.0,
        "mean_selection_shift": float(np.mean(selection_shift_values)) if selection_shift_values else 0.0,
        "mean_routing_similarity": float(np.mean(similarity_values)) if similarity_values else 1.0,
    }

    # Per-task accuracy summary
    all_tasks = set()
    if eval_rows:
        all_tasks = {row.get("task") for row in eval_rows if row.get("task")}

    for task in all_tasks:
        task_accs = []
        for v in variant_data.values():
            if task in v["eval_scores"]:
                task_accs.append(v["eval_scores"][task])
        if task_accs:
            metrics["summary"][f"mean_accuracy_{task}"] = float(np.mean(task_accs))
            metrics["summary"][f"std_accuracy_{task}"] = float(np.std(task_accs))

    return metrics


def print_quantization_metrics(metrics: dict, model_name: str) -> None:
    """Pretty-print aggregated quantization metrics."""
    print(f"\n{'='*80}")
    print(f"  Quantization Metrics Summary — {model_name}")
    print(f"{'='*80}")
    print(f"\nBaseline variant: {metrics['baseline_variant']}")
    print(f"\nSummary statistics:")
    for key, val in metrics["summary"].items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    print(f"\nPer-variant breakdown:")
    for variant, data in metrics["by_variant"].items():
        print(f"\n  {variant}")
        print(f"    Precision: {data['precision']}, Type: {data['variant_type']}")
        print(f"    Routing similarity: {data['routing_similarity_rs']:.4f}")
        print(f"    Jaccard drift: {data['jaccard_drift']:.4f}")
        print(f"    Selection shift: {data['selection_shift']:.4f}")
        if data["eval_scores"]:
            print(f"    Eval scores: {', '.join(f'{t}={s:.4f}' for t, s in data['eval_scores'].items())}")

