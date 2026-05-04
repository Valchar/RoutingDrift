from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import GRAPH11_PATH, LAYER_TYPES, OUTPUT_DIR
from test_results import Metrics


BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"
GRAY = "#7f7f7f"
DARK = "#222222"

COLORS_MODELS={"OLMoE": BLUE, "Mixtral": ORANGE}

LAYER_NAMES={
    "moe_routing": "Expert\nRouting",
    "attention":   "Attention",
    "ffn":         "FFN",
    "rmsnorm":     "RMSNorm",
    "embed":       "Embedding",
    "lm_head":     "LM Head",
    "other":       "Other",
}

REASON_NAMES={
    "data-dependent control flow":  "Data-dep. ctrl flow",
    "dynamic in-place scatter":     "In-place scatter",
    "dynamic shape: nonzero/where": "Dynamic shape (nonzero)",
    "Python for-loop over experts": "Python loop over experts",
    "dynamic shape: one_hot":       "Dynamic shape (one_hot)",
    "in-place mutation":            "In-place mutation",
    "unsupported op":               "Unsupported op",
    "other":                        "Other",
}

REASON_COLORS=[RED, ORANGE, PURPLE, "#8c564b", BLUE, GREEN, GRAY, "#bcbd22"]


def render_graph11(
    olmoe_metrics: Optional[Metrics],
    mixtral_metrics: Optional[Metrics],
    output_path: str=GRAPH11_PATH,
    show: bool=False,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, (ax_bars, ax_reasons)=plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    fig.patch.set_facecolor("white")

    _plot_layer_breaks(ax_bars, olmoe_metrics, mixtral_metrics)
    _plot_reasons_panel(ax_reasons, olmoe_metrics, mixtral_metrics)

    fig.suptitle(
        "Graph Break Analysis: OLMoE vs. Mixtral MoE",
        fontsize=13, fontweight="bold", color=DARK, y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def _style_ax(ax) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors=DARK)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#AAAAAA")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle="--", zorder=0)
    ax.set_axisbelow(True)


def _plot_layer_breaks(ax, olmoe, mixtral) -> None:
    _style_ax(ax)
    ax.set_title("(a) Graph Breaks per Layer Type", fontsize=11, fontweight="bold",
                 color=DARK, pad=8)

    all_layers=_ordered_layers(olmoe, mixtral)
    x=np.arange(len(all_layers))
    w=0.32

    models_data=[]
    if olmoe:
        models_data.append(("OLMoE", olmoe.breaks_per_layer, BLUE))
    if mixtral:
        models_data.append(("Mixtral", mixtral.breaks_per_layer, ORANGE))

    offsets=[-w/2, w/2] if len(models_data) == 2 else [0]
    max_count=1
    for _, breaks, _ in models_data:
        for lt in all_layers:
            max_count=max(max_count, breaks.get(lt, 0))

    for i, (model_name, breaks, color) in enumerate(models_data):
        counts=[breaks.get(lt, 0) for lt in all_layers]
        bars=ax.bar(
            x + offsets[i], counts, w,
            color=color, alpha=0.82, zorder=3, label=model_name,
        )
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max_count*0.04,
                    str(count),
                    ha="center", va="bottom",
                    color=DARK, fontsize=9, fontweight="bold",
                )

    routing_idx=all_layers.index("moe_routing") if "moe_routing" in all_layers else None
    if routing_idx is not None:
        ax.axvspan(routing_idx - 0.5, routing_idx + 0.5,
                   alpha=0.07, color=RED, zorder=1)
        ax.annotate(
            "All breaks\nhere",
            xy=(routing_idx, max_count*0.9),
            xytext=(routing_idx + 1.2, max_count*0.85),
            fontsize=8, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_NAMES.get(lt, lt) for lt in all_layers],
                       fontsize=9, color=DARK)
    ax.set_ylabel("Number of graph breaks", fontsize=10, color=DARK)
    ax.set_ylim(0, max_count*1.55)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#CCCCCC")


def _plot_reasons_panel(ax, olmoe, mixtral) -> None:
    _style_ax(ax)
    ax.set_title("(b) Break Reasons", fontsize=11, fontweight="bold",
                 color=DARK, pad=8)

    primary=olmoe or mixtral
    if primary is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=GRAY)
        return

    all_reasons: Dict[str, int]={}
    for lt, reason_dict in primary.break_reasons.items():
        for reason, count in reason_dict.items():
            all_reasons[reason]=all_reasons.get(reason, 0) + count

    if not all_reasons:
        ax.text(0.5, 0.5, "No breaks detected",
                ha="center", va="center", color=GREEN, fontsize=10)
        return

    reasons=list(all_reasons.keys())
    counts=[all_reasons[r] for r in reasons]
    colors=[REASON_COLORS[i % len(REASON_COLORS)] for i in range(len(reasons))]
    y_pos=np.arange(len(reasons))

    ax.barh(y_pos, counts, 0.55, color=colors, alpha=0.82, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([REASON_NAMES.get(r, r) for r in reasons],
                       fontsize=8, color=DARK)
    ax.set_xlabel("Number of breaks", fontsize=10, color=DARK)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(0, max(counts)*1.4)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.grid(axis="y", visible=False)

    for i, count in enumerate(counts):
        ax.text(count + max(counts)*0.03, i, str(count),
                va="center", fontsize=9, color=DARK, fontweight="bold")

    ax.text(0.98, 0.02, f"Showing: {primary.model_name}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=GRAY)


def _ordered_layers(olmoe, mixtral) -> List[str]:
    preferred=["moe_routing", "attention", "ffn", "rmsnorm",
                "embed", "lm_head", "other"]
    seen=set()
    result=[]
    for lt in preferred:
        has_data=(
            (olmoe and olmoe.breaks_per_layer.get(lt, 0) >= 0) or
            (mixtral and mixtral.breaks_per_layer.get(lt, 0) >= 0)
        )
        if has_data and lt not in seen:
            result.append(lt)
            seen.add(lt)
    for m in [olmoe, mixtral]:
        if m:
            for lt in m.breaks_per_layer:
                if lt not in seen:
                    result.append(lt)
                    seen.add(lt)
    return result or ["moe_routing", "attention", "ffn", "rmsnorm"]
