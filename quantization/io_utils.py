"""
io_utils.py

Small helpers to save route logs and summary files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import torch


RoutesByModule = Dict[str, List[torch.Tensor]]


def save_routes_json(routes: RoutesByModule, output_path: str | Path) -> None:
    """Save route tensors to JSON lists."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {module_name: [tensor.tolist() for tensor in tensors] for module_name, tensors in routes.items()}

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_summary_csv(rows: List[dict], output_path: str | Path) -> None:
    """Save experiment summary to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_prompts_txt(prompts: List[str], output_path: str | Path) -> None:
    """Save the exact prompts used for reproducibility/fair comparison."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts, start=1):
            f.write(f"{idx}. {prompt}\n")


def save_summary_md(
    model_name: str,
    prompts_count: int,
    top_k: int,
    rows: List[dict],
    output_path: str | Path,
) -> None:
    """Save a short human-readable summary of routing metric results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Routing Drift Summary",
        "",
        f"- Model: `{model_name}`",
        f"- Prompts: {prompts_count}",
        f"- Router top-k: {top_k}",
        "",
    ]

    if not rows:
        lines.extend(
            [
                "No variant comparisons were available.",
                "",
                "Run at least two variants to produce drift metrics.",
            ]
        )
    else:
        lines.extend(
            [
                "## Results",
                "",
                "| Variant | Routing Similarity (RS) | Jaccard Drift | Overlap@k | Selection Shift |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            variant_label = row.get("variant", row.get("precision", "n/a"))
            lines.append(
                f"| {variant_label} | {row['routing_similarity_rs']:.6f} | {row['jaccard_drift']:.6f} | {row['overlap_at_k']:.6f} | {row['selection_shift']:.6f} |"
            )

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "Higher RS and Overlap@k indicate routing closer to baseline behavior.",
                "Lower Jaccard Drift and Selection Shift indicate less routing change from the baseline.",
            ]
        )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
