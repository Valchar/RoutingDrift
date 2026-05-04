"""
drift.py

Computes research-backed MoE routing metrics between FP16 and quantized routes.
"""

from __future__ import annotations

from typing import Dict, List

import torch


RoutesByModule = Dict[str, List[torch.Tensor]]


def _flatten_route_tensor(route_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert route tensor to [num_tokens_or_positions, top_k].
    """
    if route_tensor.ndim == 0:
        return route_tensor.reshape(1, 1)
    if route_tensor.ndim == 1:
        return route_tensor.reshape(-1, 1)
    return route_tensor.reshape(-1, route_tensor.shape[-1])


def _extend_zero_scores(route_tensor: torch.Tensor, rs_scores: List[float], overlap_scores: List[float]) -> None:
    n_rows = _flatten_route_tensor(route_tensor).shape[0]
    if n_rows:
        rs_scores.extend([0.0] * n_rows)
        overlap_scores.extend([0.0] * n_rows)


def _score_route_pair(
    baseline_tensor: torch.Tensor,
    quantized_tensor: torch.Tensor,
    rs_scores: List[float],
    overlap_scores: List[float],
) -> None:
    base = _flatten_route_tensor(baseline_tensor)
    quant = _flatten_route_tensor(quantized_tensor)
    min_rows = min(base.shape[0], quant.shape[0])
    k = max(base.shape[1], quant.shape[1], 1)
    for row_idx in range(min_rows):
        b_set = set(base[row_idx].tolist())
        q_set = set(quant[row_idx].tolist())
        union = len(b_set | q_set)
        intersection = len(b_set & q_set)
        rs_scores.append((intersection / union) if union > 0 else 1.0)
        overlap_scores.append(intersection / k)
    if min_rows < base.shape[0]:
        _extend_zero_scores(base[min_rows:], rs_scores, overlap_scores)
    if min_rows < quant.shape[0]:
        _extend_zero_scores(quant[min_rows:], rs_scores, overlap_scores)


def _collect_row_level_scores_for_module(
    baseline_calls: List[torch.Tensor],
    quantized_calls: List[torch.Tensor],
) -> tuple[list[float], list[float]]:
    rs_scores: list[float] = []
    overlap_scores: list[float] = []
    _collect_pairwise_route_scores(baseline_calls, quantized_calls, rs_scores, overlap_scores)
    return rs_scores, overlap_scores


def _collect_row_level_scores(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
) -> tuple[list[float], list[float]]:
    """
    Returns row-level:
        - routing similarity (RS): Jaccard similarity
        - overlap@k
    """
    rs_scores: list[float] = []
    overlap_scores: list[float] = []

    common_modules = sorted(set(baseline_routes.keys()) & set(quantized_routes.keys()))

    for module_name in common_modules:
        baseline_calls = baseline_routes[module_name]
        quantized_calls = quantized_routes[module_name]
        _collect_pairwise_route_scores(baseline_calls, quantized_calls, rs_scores, overlap_scores)

    return rs_scores, overlap_scores


def _collect_pairwise_route_scores(
    baseline_calls: List[torch.Tensor],
    quantized_calls: List[torch.Tensor],
    rs_scores: List[float],
    overlap_scores: List[float],
) -> None:
    min_calls = min(len(baseline_calls), len(quantized_calls))
    for i in range(min_calls):
        _score_route_pair(baseline_calls[i], quantized_calls[i], rs_scores, overlap_scores)

    for extra in baseline_calls[min_calls:]:
        _extend_zero_scores(extra, rs_scores, overlap_scores)
    for extra in quantized_calls[min_calls:]:
        _extend_zero_scores(extra, rs_scores, overlap_scores)


def compute_routing_similarity_rs(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
) -> float:
    """
    RS (Routing Similarity): average Jaccard similarity between FP16 and quantized
    top-k expert sets across aligned outputs.
    """
    rs_scores, _ = _collect_row_level_scores(baseline_routes, quantized_routes)
    return sum(rs_scores) / len(rs_scores) if rs_scores else 0.0


def compute_overlap_at_k(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
) -> float:
    """
    Overlap@k: average |A intersection B| / k between FP16 and quantized top-k sets.
    """
    _, overlap_scores = _collect_row_level_scores(baseline_routes, quantized_routes)
    return sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0


def summarize_research_metrics(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
) -> dict[str, float]:
    """
    Returns paper-aligned routing metrics:
        - routing_similarity_rs
        - jaccard_drift (1 - RS)
        - overlap_at_k
        - selection_shift (1 - overlap_at_k)
    """
    rs = compute_routing_similarity_rs(baseline_routes, quantized_routes)
    overlap = compute_overlap_at_k(baseline_routes, quantized_routes)
    return {
        "routing_similarity_rs": rs,
        "jaccard_drift": 1.0 - rs,
        "overlap_at_k": overlap,
        "selection_shift": 1.0 - overlap,
    }


def compute_layerwise_metrics(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
) -> dict[str, dict[str, float]]:
    """
    Compute drift metrics per router/gate module (layer-level view).
    """
    results: dict[str, dict[str, float]] = {}
    module_names = sorted(set(baseline_routes.keys()) | set(quantized_routes.keys()))

    for module_name in module_names:
        base_calls = baseline_routes.get(module_name, [])
        quant_calls = quantized_routes.get(module_name, [])

        rs_scores, overlap_scores = _collect_row_level_scores_for_module(base_calls, quant_calls)

        if rs_scores:
            rs = sum(rs_scores) / len(rs_scores)
            overlap = sum(overlap_scores) / len(overlap_scores)
        else:
            rs = 0.0
            overlap = 0.0

        results[module_name] = {
            "routing_similarity_rs": rs,
            "jaccard_drift": 1.0 - rs,
            "overlap_at_k": overlap,
            "selection_shift": 1.0 - overlap,
            "num_rows": float(len(rs_scores)),
        }

    return results


def build_layerwise_rows(
    baseline_routes: RoutesByModule,
    quantized_routes: RoutesByModule,
    variant: str,
) -> list[dict[str, float | str]]:
    """
    Build row-oriented layer metrics table for CSV/heatmaps.
    """
    per_layer = compute_layerwise_metrics(baseline_routes, quantized_routes)
    rows: list[dict[str, float | str]] = []
    for module_name, metrics in per_layer.items():
        rows.append(
            {
                "variant": variant,
                "module": module_name,
                "routing_similarity_rs": metrics["routing_similarity_rs"],
                "jaccard_drift": metrics["jaccard_drift"],
                "overlap_at_k": metrics["overlap_at_k"],
                "selection_shift": metrics["selection_shift"],
                "num_rows": metrics["num_rows"],
            }
        )
    return rows
