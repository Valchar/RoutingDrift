from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from config import GRAPH14_PATH, OUTPUT_DIR
from dense_comparison import DenseComparisonResult


BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#7f7f7f"
DARK = "#222222"


def render_graph14(
    results: List[DenseComparisonResult],
    output_path: str=GRAPH14_PATH,
    show: bool=False,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig=plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("white")

    gs=GridSpec(
        2, 2, figure=fig,
        left=0.09, right=0.96,
        top=0.90, bottom=0.09,
        wspace=0.38, hspace=0.45,
    )

    ax_lat=fig.add_subplot(gs[0, 0])
    ax_tps=fig.add_subplot(gs[0, 1])
    ax_spd=fig.add_subplot(gs[1, 0])
    ax_ovh=fig.add_subplot(gs[1, 1])

    _plot_latency(ax_lat, results)
    _plot_throughput(ax_tps, results)
    _plot_speedup(ax_spd, results)
    _plot_overhead(ax_ovh, results)

    fig.suptitle(
        "MoE vs. Dense Baseline: Isolating Routing Overhead",
        fontsize=13, fontweight="bold", color=DARK, y=0.97,
    )
    fig.text(
        0.5, 0.935,
        "Dense baseline has identical per-token FLOPs; latency difference reflects routing cost only.",
        ha="center", fontsize=9, color=GRAY,
    )

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


def _no_data(ax) -> None:
    ax.text(0.5, 0.5, "No data", ha="center", va="center",
            color=GRAY, fontsize=11, transform=ax.transAxes)


def _plot_latency(ax, results: List[DenseComparisonResult]) -> None:
    _style_ax(ax)
    ax.set_title("(a) Latency: MoE vs. Dense",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
    if not results:
        _no_data(ax)
        return
    names=[r.moe_name for r in results]
    x=np.arange(len(names))
    w=0.35
    moe_v=[r.moe_p50_ms for r in results]
    den_v=[r.dense_p50_ms for r in results]
    b1=ax.bar(x - w/2, moe_v, w, color=ORANGE, alpha=0.82, label="MoE", zorder=3)
    b2=ax.bar(x + w/2, den_v, w, color=BLUE, alpha=0.82, label="Dense", zorder=3)
    top=max(moe_v + den_v) if moe_v else 1
    for bar, v in zip(list(b1) + list(b2), moe_v + den_v):
        ax.text(bar.get_x() + bar.get_width()/2, v + top*0.02,
                f"{v:.1f}", ha="center", va="bottom", color=DARK, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color=DARK)
    ax.set_ylabel("Milliseconds (lower is better)", fontsize=9, color=DARK)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#CCCCCC")


def _plot_throughput(ax, results: List[DenseComparisonResult]) -> None:
    _style_ax(ax)
    ax.set_title("(b) Throughput: MoE vs. Dense",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
    if not results:
        _no_data(ax)
        return
    names=[r.moe_name for r in results]
    x=np.arange(len(names))
    w=0.35
    moe_v=[r.moe_throughput_tps for r in results]
    den_v=[r.dense_throughput_tps for r in results]
    b1=ax.bar(x - w/2, moe_v, w, color=ORANGE, alpha=0.82, label="MoE", zorder=3)
    b2=ax.bar(x + w/2, den_v, w, color=BLUE, alpha=0.82, label="Dense", zorder=3)
    top=max(moe_v + den_v) if moe_v else 1
    for bar, v in zip(list(b1) + list(b2), moe_v + den_v):
        ax.text(bar.get_x() + bar.get_width()/2, v + top*0.02,
                f"{v:,.0f}", ha="center", va="bottom", color=DARK, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color=DARK)
    ax.set_ylabel("Tokens per second (higher is better)", fontsize=9, color=DARK)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#CCCCCC")


def _plot_speedup(ax, results: List[DenseComparisonResult]) -> None:
    _style_ax(ax)
    ax.set_title("(c) MoE Speedup vs. Dense Baseline",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
    if not results:
        _no_data(ax)
        return
    names=[r.moe_name for r in results]
    speedups=[r.speedup_moe_vs_dense for r in results]
    colors=[GREEN if s >= 1.0 else RED for s in speedups]
    x=np.arange(len(names))
    bars=ax.bar(x, speedups, 0.5, color=colors, alpha=0.82, zorder=3)
    top=max(speedups) if speedups else 2
    ax.axhline(1.0, color=DARK, linewidth=1.0, linestyle="--", alpha=0.5)
    ax.text(len(names) - 0.45, 1.0 + top*0.02, "1× (baseline)",
            color=DARK, fontsize=7.5, ha="right")
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top*0.02,
                f"{s:.2f}×", ha="center", va="bottom",
                color=DARK, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color=DARK)
    ax.set_ylabel("Speedup (dense / MoE latency)", fontsize=9, color=DARK)
    ax.text(0.01, 0.97, ">1 = MoE faster  <1 = Dense faster",
            transform=ax.transAxes, color=GRAY, fontsize=7.5, va="top")


def _plot_overhead(ax, results: List[DenseComparisonResult]) -> None:
    _style_ax(ax)
    ax.set_title("(d) Routing Overhead (MoE − Dense latency)",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
    if not results:
        _no_data(ax)
        return
    names=[r.moe_name for r in results]
    overheads=[r.routing_overhead_ms for r in results]
    colors=[RED if o > 0 else GREEN for o in overheads]
    x=np.arange(len(names))
    bars=ax.bar(x, overheads, 0.5, color=colors, alpha=0.82, zorder=3)
    ax.axhline(0.0, color=DARK, linewidth=0.8, linestyle="--", alpha=0.4)
    span=max(abs(o) for o in overheads) if overheads else 1
    for bar, o in zip(bars, overheads):
        text_y=o + span*0.05 if o >= 0 else o - span*0.05
        va="bottom" if o >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, text_y,
                f"{o:+.1f} ms", ha="center", va=va,
                color=DARK, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color=DARK)
    ax.set_ylabel("Overhead (ms)", fontsize=9, color=DARK)
    ax.text(0.01, 0.97, "Positive = routing costs time  ·  Negative = MoE faster",
            transform=ax.transAxes, color=GRAY, fontsize=7.5, va="top")
