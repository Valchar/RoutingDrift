from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import GRAPH13_PATH, OUTPUT_DIR
from roofline_analyzer import RooflineReport


BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"
GRAY = "#7f7f7f"
DARK = "#222222"

LAYER_COLORS={
    "attention": BLUE,
    "ffn": ORANGE,
    "moe_routing": RED,
    "rmsnorm": GREEN,
    "embed": PURPLE,
    "lm_head": "#8c564b",
    "other": GRAY,
}


def render_graph13(
    olmoe_report: Optional[RooflineReport],
    mixtral_report: Optional[RooflineReport],
    output_path: str=GRAPH13_PATH,
    show: bool=False,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reports=[r for r in [olmoe_report, mixtral_report] if r is not None]
    n=len(reports)
    if n == 0:
        fig, ax=plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.text(0.5, 0.5, "No roofline data", ha="center", va="center",
                color=GRAY, fontsize=12, transform=ax.transAxes)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path

    panel_labels=["(a)", "(b)"]
    subtitles={
        "OLMoE": "OLMoE (64 experts, 8 active/token)",
        "Mixtral": "Mixtral (8 experts, 2 active/token)",
    }

    fig, axes=plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1:
        axes=[axes]
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Roofline Analysis: Memory- vs. Compute-Bound Operations",
        fontsize=13, fontweight="bold", color=DARK, y=1.02,
    )

    for idx, (ax, report) in enumerate(zip(axes, reports)):
        label=panel_labels[idx] if idx < len(panel_labels) else ""
        _draw_roofline(ax, report,
                       f"{label} {subtitles.get(report.model_name, report.model_name)}")

    patches=[mpatches.Patch(color=c, label=lt) for lt, c in LAYER_COLORS.items()]
    fig.legend(
        handles=patches,
        loc="lower center", ncol=len(LAYER_COLORS),
        framealpha=0.9, edgecolor="#CCCCCC",
        fontsize=8, bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def _draw_roofline(ax, report: RooflineReport, title: str) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors=DARK)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#AAAAAA")
    ax.set_xscale("log")
    ax.set_yscale("log")

    hw=report.hw
    peak_flops=hw.peak_flops_tflops * 1e12
    peak_bw=hw.peak_bandwidth_gbps * 1e9
    ridge=report.ridge_point

    all_ai=[r.arithmetic_intensity for r in report.records if r.flops > 0]
    if not all_ai:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color=GRAY, transform=ax.transAxes)
        return

    x_min=max(1e-3, min(all_ai)*0.3)
    x_max=max(all_ai)*3.0
    x_arr=np.logspace(np.log10(x_min), np.log10(x_max), 400)

    roofline=np.minimum(peak_bw * x_arr, peak_flops)
    ax.plot(x_arr, roofline, color=DARK, linewidth=1.8, zorder=5)

    ax.axvline(ridge, color=GRAY, linewidth=1.0, linestyle="--", alpha=0.7, zorder=4)
    ax.text(ridge*1.06, peak_flops*0.5, f"ridge\n{ridge:.1f} FLOP/B",
            color=DARK, fontsize=7.5, va="center")

    # Lightly shade memory/compute regions to guide the eye
    ax.fill_between(x_arr, 1e8, roofline, where=(x_arr < ridge),
                    alpha=0.04, color=RED, zorder=0)
    ax.fill_between(x_arr, 1e8, roofline, where=(x_arr >= ridge),
                    alpha=0.04, color=GREEN, zorder=0)

    for rec in report.records:
        if rec.flops <= 0:
            continue
        color=LAYER_COLORS.get(rec.layer_type, GRAY)
        # Dot placed at its theoretical peak — shows which side of the ridge it falls on
        y=min(peak_bw * rec.arithmetic_intensity, peak_flops)
        ax.scatter(rec.arithmetic_intensity, y, color=color,
                   s=60, zorder=6, alpha=0.9, edgecolors=DARK, linewidths=0.4)
        ax.text(rec.arithmetic_intensity*1.1, y*1.12,
                rec.op_name, color=color, fontsize=7.5)

    ax.text(x_min*1.6, peak_flops*0.015, "memory-bound",
            color=RED, fontsize=8, alpha=0.8, style="italic")
    ax.text(x_max*0.3, peak_flops*0.015, "compute-bound",
            color=GREEN, fontsize=8, alpha=0.8, style="italic")

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=10, color=DARK)
    ax.set_ylabel("Attainable Performance (FLOP / s)", fontsize=10, color=DARK)
    ax.set_title(
        f"{title}\n[{hw.device_name}]  "
        f"mem-bound: {report.memory_bound_ops}  "
        f"compute-bound: {report.compute_bound_ops}",
        fontsize=10, fontweight="bold", color=DARK, pad=8,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(1e9, peak_flops*3)
    ax.grid(axis="both", color="#DDDDDD", linewidth=0.5, linestyle="--", alpha=1.0)
