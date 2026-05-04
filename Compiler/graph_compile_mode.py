from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import COMPILE_MODES, GRAPH12_PATH, OUTPUT_DIR
from test_results import Metrics


BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#7f7f7f"
DARK = "#222222"

COLORS_MODELS={"OLMoE": BLUE, "Mixtral": ORANGE}

MODE_COLORS={
    "eager":           "#7f7f7f",
    "default":         "#1f77b4",
    "reduce-overhead": "#2ca02c",
    "max-autotune":    "#ff7f0e",
}

MODE_LABELS={
    "eager":           "Eager",
    "default":         "Default",
    "reduce-overhead": "Reduce\nOverhead",
    "max-autotune":    "Max\nAutotune",
}


def render_graph12(
    olmoe_metrics: Optional[Metrics],
    mixtral_metrics: Optional[Metrics],
    output_path: str=GRAPH12_PATH,
    show: bool=False,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, (ax_throughput, ax_latency)=plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    _plot_throughput(ax_throughput, olmoe_metrics, mixtral_metrics)
    _plot_latency_summary(ax_latency, olmoe_metrics, mixtral_metrics)

    fig.suptitle(
        "Compile Mode Benchmark: OLMoE vs. Mixtral MoE",
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


def _plot_throughput(ax, olmoe, mixtral) -> None:
    _style_ax(ax)
    ax.set_title("(a) Throughput by Compile Mode",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)

    all_modes=["eager"] + COMPILE_MODES
    models_data=_collect_models(olmoe, mixtral)

    x=np.arange(len(all_modes))
    offsets=np.linspace(-0.22, 0.22, len(models_data)) if len(models_data) > 1 else [0]
    bw=0.38 / max(len(models_data), 1)

    any_data=False
    for i, (model_name, metrics, color) in enumerate(models_data):
        best=metrics.best_compile_mode
        tps_arr=[]
        edge_colors=[]
        lws=[]
        for mode in all_modes:
            r=metrics.compile_results.get(mode, {})
            tps=r.get("throughput_tps", 0)
            if tps == float("inf"):
                tps=0
            tps_arr.append(tps)
            edge_colors.append(DARK if mode == best else "none")
            lws.append(1.5 if mode == best else 0)
            if tps > 0:
                any_data=True

        bars=ax.bar(
            x + offsets[i], tps_arr, bw,
            color=[MODE_COLORS.get(m, GRAY) for m in all_modes],
            alpha=0.82, edgecolor=edge_colors, linewidth=lws,
            label=model_name, zorder=3,
        )

        max_tps=max(tps_arr) if max(tps_arr) > 0 else 1
        for bar, mode, tps in zip(bars, all_modes, tps_arr):
            if tps > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    tps + max_tps*0.015,
                    f"{tps:,.0f}",
                    ha="center", va="bottom",
                    color=DARK, fontsize=8,
                )
            elif mode != "eager":
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    max_tps*0.04, "N/A",
                    ha="center", va="bottom",
                    color=GRAY, fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in all_modes],
                       fontsize=9, color=DARK)
    ax.set_ylabel("Tokens / second", fontsize=10, color=DARK)

    if not any_data:
        ax.text(0.5, 0.5, "No throughput data collected.",
                ha="center", va="center", color=GRAY, fontsize=11,
                transform=ax.transAxes)
    else:
        model_patches=[mpatches.Patch(color=c, label=n) for n, _, c in models_data]
        ax.legend(handles=model_patches, fontsize=9,
                  framealpha=0.9, edgecolor="#CCCCCC")

    ax.text(0.01, 0.97, "N/A = mode requires C++ compiler or GPU",
            transform=ax.transAxes, color=GRAY, fontsize=7, va="top")


def _plot_latency_summary(ax, olmoe, mixtral) -> None:
    _style_ax(ax)
    ax.set_title("(b) Latency by Model (Eager Baseline)",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)

    models_data=_collect_models(olmoe, mixtral)
    names=[]
    p50s=[]
    p90s=[]
    colors=[]

    for name, metrics, color in models_data:
        eager=metrics.compile_results.get("eager", {})
        p50=eager.get("p50_ms", 0)
        p90=eager.get("p90_ms", 0)
        if p50 and p50 != float("inf"):
            names.append(name)
            p50s.append(p50)
            p90s.append(p90)
            colors.append(color)

    if not names:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color=GRAY, fontsize=12, transform=ax.transAxes)
        return

    y_pos=np.arange(len(names))
    bars=ax.barh(y_pos, p50s, 0.45, color=colors, alpha=0.82, zorder=3)

    # p90 extension shown as a lighter segment
    for i, (p50, p90) in enumerate(zip(p50s, p90s)):
        ax.barh(y_pos[i], max(0, p90 - p50), 0.45, left=p50,
                color=colors[i], alpha=0.3, zorder=2)

    for bar, p50 in zip(bars, p50s):
        ax.text(
            bar.get_width() + max(p50s)*0.02,
            bar.get_y() + bar.get_height()/2,
            f"{p50:.1f} ms",
            va="center", color=DARK, fontsize=10, fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11, color=DARK)
    ax.set_xlabel("Milliseconds per forward pass  (lower is better)",
                  fontsize=9, color=DARK)
    ax.set_xlim(0, max(p90s)*1.35)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.grid(axis="y", visible=False)

    ax.text(0.98, 0.04,
            "Solid = p50  ·  Faded = p90",
            transform=ax.transAxes, ha="right", va="bottom",
            color=GRAY, fontsize=7)


def _collect_models(olmoe, mixtral) -> List[Tuple[str, Metrics, str]]:
    result=[]
    if olmoe:
        result.append(("OLMoE", olmoe, COLORS_MODELS["OLMoE"]))
    if mixtral:
        result.append(("Mixtral", mixtral, COLORS_MODELS["Mixtral"]))
    return result
