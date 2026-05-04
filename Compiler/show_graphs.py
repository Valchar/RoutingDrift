import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs("outputs", exist_ok=True)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"
GRAY = "#7f7f7f"
DARK = "#222222"

COLORS_MODELS={"OLMoE": BLUE, "Mixtral": ORANGE}
COLORS_MODES={
    "eager": "#7f7f7f",
    "default": "#1f77b4",
    "reduce-overhead": "#2ca02c",
    "max-autotune": "#ff7f0e",
}


def _style(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors=DARK)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color("#AAAAAA")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)


fig11, (ax_grp, ax_stk)=plt.subplots(1, 2, figsize=(11, 4.5))
fig11.patch.set_facecolor("white")

layer_types=["moe_routing", "attention", "ffn", "rmsnorm"]
display_names=["Expert\nRouting", "Attention", "FFN", "RMSNorm"]

olmoe_breaks=[4, 0, 0, 0]
mixtral_breaks=[3, 0, 0, 0]

x, w=np.arange(4), 0.35

_style(ax_grp)
b1=ax_grp.bar(x - w/2, olmoe_breaks, w, color=BLUE, alpha=0.82, label="OLMoE", zorder=3)
b2=ax_grp.bar(x + w/2, mixtral_breaks, w, color=ORANGE, alpha=0.82, label="Mixtral", zorder=3)
for b, n in [(bar, v) for bar, v in zip(list(b1)+list(b2), olmoe_breaks+mixtral_breaks) if v > 0]:
    ax_grp.text(b.get_x()+b.get_width()/2, n+0.08, str(n),
                ha="center", va="bottom", color=DARK, fontsize=10, fontweight="bold")
ax_grp.axvspan(-0.5, 0.5, alpha=0.07, color=RED, zorder=1)
ax_grp.annotate("All breaks\nhere", xy=(0, 4.0), xytext=(1.0, 3.8),
                fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax_grp.set_xticks(x)
ax_grp.set_xticklabels(display_names, fontsize=9, color=DARK)
ax_grp.set_ylabel("Number of graph breaks", fontsize=10, color=DARK)
ax_grp.set_ylim(0, 5.5)
ax_grp.set_title("(a) Graph Breaks per Layer Type", fontsize=11, fontweight="bold",
                 color=DARK, pad=8)
ax_grp.legend(fontsize=9, framealpha=0.9, edgecolor="#CCCCCC")
ax_grp.text(0.02, 0.97,
    "OLMoE: 4 breaks / 5 subgraphs / 20% compiled\n"
    "Mixtral: 3 breaks / 4 subgraphs / 25% compiled",
    transform=ax_grp.transAxes, color=GRAY, fontsize=7, va="top")

reasons=[
    ("topk: data-dep. indices",    RED,    1),
    ("Python for-loop over experts", PURPLE, 1),
    ("index_add_: dynamic index",  ORANGE, 1),
    ("expert_mask.any(): data-dep.", "#8c564b", 1),
]
_style(ax_stk)
bottom=0
patches=[]
for label, color, count in reasons:
    ax_stk.bar([0], [count], 0.45, bottom=bottom, color=color, alpha=0.82, zorder=3)
    ax_stk.text(0.28, bottom + 0.5, label, va="center", color=DARK, fontsize=8)
    patches.append(mpatches.Patch(color=color, label=label))
    bottom+=count
ax_stk.set_xticks([0])
ax_stk.set_xticklabels(["Expert\nRouting"], fontsize=10, color=DARK)
ax_stk.set_xlim(-0.7, 1.8)
ax_stk.set_ylim(0, 4.8)
ax_stk.set_ylabel("Break count by reason", fontsize=10, color=DARK)
ax_stk.set_title("(b) Break Reasons — OLMoE Routing Layer",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
ax_stk.legend(handles=patches, fontsize=7, loc="upper right",
              framealpha=0.9, edgecolor="#CCCCCC")

fig11.suptitle("Graph Break Analysis: OLMoE vs. Mixtral",
               fontsize=13, fontweight="bold", color=DARK, y=1.01)
plt.tight_layout()
fig11.savefig("outputs/graph11_graph_breaks.png", dpi=150,
              bbox_inches="tight", facecolor="white")
plt.close(fig11)
print("saved graph11")


modes=["eager", "default", "reduce-overhead", "max-autotune"]
mode_labels=["Eager", "Default", "Reduce\nOverhead", "Max\nAutotune"]

olmoe_p50=np.array([45.2, 32.1, 27.8, 24.3])
olmoe_p90=np.array([47.1, 33.8, 29.2, 25.9])
olmoe_tps=np.array([2832, 3984, 4603, 5267])
olmoe_spd=np.array([1.0, 1.41, 1.63, 1.86])
olmoe_ct=np.array([0, 8.2, 12.4, 38.7])

mix_p50=np.array([82.3, 61.4, 54.2, 48.7])
mix_p90=np.array([85.1, 63.2, 56.1, 50.4])
mix_tps=np.array([1552, 2084, 2361, 2628])
mix_spd=np.array([1.0, 1.34, 1.52, 1.69])
mix_ct=np.array([0, 14.3, 21.7, 67.2])

fig12, (ax1, ax2, ax3)=plt.subplots(1, 3, figsize=(14, 4.5))
fig12.patch.set_facecolor("white")
mode_colors=[COLORS_MODES[m] for m in modes]
x, w=np.arange(4), 0.32
best=2  # reduce-overhead

_style(ax1)
ax1.bar(x - w/2, olmoe_p50, w, color=mode_colors, alpha=0.82, label="OLMoE", zorder=3)
ax1.bar(x + w/2, mix_p50, w, color=mode_colors, alpha=0.45, label="Mixtral", zorder=3)
ax1.errorbar(x - w/2, olmoe_p50, yerr=[np.zeros(4), olmoe_p90 - olmoe_p50],
             fmt="none", color=DARK, capsize=3, lw=1, alpha=0.5)
ax1.errorbar(x + w/2, mix_p50, yerr=[np.zeros(4), mix_p90 - mix_p50],
             fmt="none", color=DARK, capsize=3, lw=1, alpha=0.5)
ax1.text(x[best] - w/2, olmoe_p50[best] + 1.5, "★", ha="center", color=DARK, fontsize=12)
ax1.text(x[best] + w/2, mix_p50[best] + 1.5, "★", ha="center", color=DARK, fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(mode_labels, fontsize=8, color=DARK)
ax1.set_ylabel("Latency p50 (ms)  — lower is better", fontsize=9, color=DARK)
ax1.set_title("(a) Latency per Compile Mode", fontsize=11, fontweight="bold",
              color=DARK, pad=8)
ax1.text(0.02, 0.97, "Error bars = p90  ·  ★ = best mode",
         transform=ax1.transAxes, color=GRAY, fontsize=7, va="top")
ax1.legend(fontsize=8, framealpha=0.9, edgecolor="#CCCCCC")

_style(ax2)
ax2.bar(x - w/2, olmoe_tps, w, color=mode_colors, alpha=0.82, label="OLMoE", zorder=3)
ax2.bar(x + w/2, mix_tps, w, color=mode_colors, alpha=0.45, label="Mixtral", zorder=3)
ax2.text(x[best] - w/2, olmoe_tps[best] + 50, "★", ha="center", color=DARK, fontsize=12)
ax2.text(x[best] + w/2, mix_tps[best] + 50, "★", ha="center", color=DARK, fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(mode_labels, fontsize=8, color=DARK)
ax2.set_ylabel("Throughput (tokens/sec)  — higher is better", fontsize=9, color=DARK)
ax2.set_title("(b) Throughput per Compile Mode", fontsize=11, fontweight="bold",
              color=DARK, pad=8)
ax2.legend(fontsize=8, framealpha=0.9, edgecolor="#CCCCCC")

_style(ax3)
for i in range(1, 4):
    m=modes[i]
    ax3.scatter(olmoe_ct[i], olmoe_spd[i],
                s=120 if i == best else 60, c=COLORS_MODES[m], marker="o", zorder=4,
                edgecolors=DARK if i == best else "none", linewidths=1.2)
    ax3.annotate(f"OLMoE / {mode_labels[i]}" + (" ★" if i == best else ""),
                 (olmoe_ct[i], olmoe_spd[i]), textcoords="offset points",
                 xytext=(6, 4), fontsize=7, color=DARK)
    ax3.scatter(mix_ct[i], mix_spd[i],
                s=120 if i == best else 60, c=COLORS_MODES[m], marker="^", zorder=4,
                edgecolors=DARK if i == best else "none", linewidths=1.2)
    ax3.annotate(f"Mixtral / {mode_labels[i]}" + (" ★" if i == best else ""),
                 (mix_ct[i], mix_spd[i]), textcoords="offset points",
                 xytext=(6, -12), fontsize=7, color=DARK)
ax3.axhline(1.0, color=GRAY, lw=1.0, linestyle="--", alpha=0.6)
ax3.text(1, 1.02, "eager baseline", color=GRAY, fontsize=7)
ax3.set_xlabel("Compile time (s)", fontsize=9, color=DARK)
ax3.set_ylabel("Speedup vs. eager", fontsize=9, color=DARK)
ax3.set_title("(c) Speedup vs. Compile-Time Trade-off",
              fontsize=11, fontweight="bold", color=DARK, pad=8)
leg=[
    plt.Line2D([0],[0], marker="o", color="none", markerfacecolor=GRAY, markersize=7, label="OLMoE"),
    plt.Line2D([0],[0], marker="^", color="none", markerfacecolor=GRAY, markersize=7, label="Mixtral"),
] + [mpatches.Patch(color=COLORS_MODES[m], label=l) for m, l in zip(modes[1:], mode_labels[1:])]
ax3.legend(handles=leg, fontsize=7, framealpha=0.9, edgecolor="#CCCCCC")
ax3.grid(axis="both", color="#DDDDDD", linewidth=0.5, linestyle="--")

fig12.suptitle("Compile Mode Comparison: OLMoE and Mixtral MoE",
               fontsize=13, fontweight="bold", color=DARK, y=1.01)
fig12.text(0.5, -0.02,
    "Best mode → OLMoE: reduce-overhead (1.63×)  ·  Mixtral: reduce-overhead (1.52×)",
    ha="center", fontsize=8, color=GRAY, style="italic")
plt.tight_layout()
fig12.savefig("outputs/graph12_compile_modes.png", dpi=150,
              bbox_inches="tight", facecolor="white")
plt.close(fig12)
print("saved graph12")


fig_r, ax=plt.subplots(figsize=(10, 11))
fig_r.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 13)
ax.set_ylim(0, 13)
ax.axis("off")


def node(ax, cx, cy, w, h, text, fc="#EEF2F7", ec=BLUE, tc=DARK, fs=9, bold=False):
    r=FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                     boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", multialignment="center")


def arr(ax, x1, y1, x2, y2, color=DARK, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12))


def break_tag(ax, rx, cy, num, reason, detail):
    ax.plot([rx, rx + 0.25], [cy, cy], color=RED, lw=2.0)
    ax.text(rx + 0.35, cy + 0.18, f"BREAK {num}", color=RED,
            fontsize=8, fontweight="bold", va="bottom")
    ax.text(rx + 0.35, cy - 0.18, f"{reason}", color=RED, fontsize=7.5, va="top")
    ax.text(rx + 0.35, cy - 0.52, f"{detail}", color=GRAY, fontsize=6.5,
            va="top", style="italic")


# OK nodes: light blue fill; break nodes: light red fill
OK_FC, OK_EC="#EBF4FB", BLUE
BRK_FC, BRK_EC="#FDECEA", RED
BOOK_FC, BOOK_EC="#F5F5F5", GRAY

node(ax, 4, 12.2, 5, 0.7, "Input  x: (batch, seq, hidden)", fc=BOOK_FC, ec=BOOK_EC)
arr(ax, 4, 11.85, 4, 11.35)

node(ax, 4, 11.0, 5, 0.7, "x_flat = x.view(-1, hidden)  [static reshape — OK]",
     fc=OK_FC, ec=OK_EC, fs=8.5)
arr(ax, 4, 10.65, 4, 10.15)

node(ax, 4, 9.8, 5, 0.7, "router_logits = gate(x_flat)  [linear — OK]",
     fc=OK_FC, ec=OK_EC, fs=8.5)
arr(ax, 4, 9.45, 4, 8.9)

node(ax, 4, 8.55, 5.2, 0.72,
     "routing_weights, selected_experts\n= torch.topk(router_logits, top_k)",
     fc=BRK_FC, ec=BRK_EC, fs=8.5, bold=True)
break_tag(ax, 6.65, 8.55, 1, "data-dependent indices",
          "topk values drive later branches → Dynamo inserts runtime guards")
arr(ax, 4, 8.19, 4, 7.65)

node(ax, 4, 7.3, 5, 0.7, "routing_weights = F.softmax(routing_weights)  [OK]",
     fc=OK_FC, ec=OK_EC, fs=8.5)
arr(ax, 4, 6.95, 4, 6.4)

node(ax, 4, 6.05, 5, 0.72, "for expert_idx in range(num_experts):",
     fc=BRK_FC, ec=BRK_EC, fs=8.5, bold=True)
break_tag(ax, 6.65, 6.05, 2, "Python for-loop over experts",
          "each iteration uses a different dynamic mask; Inductor cannot fuse across loops")
arr(ax, 4, 5.69, 4, 5.15)

node(ax, 4, 4.8, 5.2, 0.72, "expert_mask.any()  /  torch.where(expert_mask)",
     fc=BRK_FC, ec=BRK_EC, fs=8.5, bold=True)
break_tag(ax, 6.65, 4.8, 3, "data-dependent branch",
          ".any() returns a bool whose value depends on routing → unresolvable at trace time")
arr(ax, 4, 4.44, 4, 3.9)

node(ax, 4, 3.55, 5, 0.7, "expert_out = experts[i](x_flat[token_indices])  [per-expert — OK]",
     fc=OK_FC, ec=OK_EC, fs=8)
arr(ax, 4, 3.19, 4, 2.65)

node(ax, 4, 2.3, 5.2, 0.72,
     "final_hidden.index_add_(0, token_indices, expert_out * w)",
     fc=BRK_FC, ec=BRK_EC, fs=8.5, bold=True)
break_tag(ax, 6.65, 2.3, 4, "in-place mutation on dynamic index",
          "index_add_ with variable-length token_indices forces a new subgraph per iteration")
arr(ax, 4, 1.94, 4, 1.4)

node(ax, 4, 1.05, 5, 0.7, "Output  final_hidden.view(orig_shape)",
     fc=BOOK_FC, ec=BOOK_EC)

sub_spans=[
    (12.2, 9.45, BLUE,   "subgraph 1\n(compiled)"),
    (8.9,  6.4,  GRAY,   "subgraph 2"),
    (6.4,  4.44, GRAY,   "subgraph 3"),
    (4.44, 1.94, GRAY,   "subgraph 4"),
    (1.94, 0.7,  GRAY,   "subgraph 5"),
]
for top, bot, color, label in sub_spans:
    mid=(top + bot) / 2
    ax.plot([0.55, 0.55], [bot + 0.05, top - 0.05], color=color, lw=1.8)
    ax.plot([0.55, 0.75], [top - 0.05, top - 0.05], color=color, lw=1.8)
    ax.plot([0.55, 0.75], [bot + 0.05, bot + 0.05], color=color, lw=1.8)
    ax.text(0.45, mid, label, ha="right", va="center", color=color, fontsize=7)

leg_items=[
    mpatches.Patch(fc=BRK_FC, ec=BRK_EC, label="graph break (new subgraph)"),
    mpatches.Patch(fc=OK_FC, ec=OK_EC, label="compiled cleanly"),
    mpatches.Patch(fc=BOOK_FC, ec=BOOK_EC, label="I/O node"),
]
ax.legend(handles=leg_items, loc="lower right", fontsize=8,
          framealpha=0.95, edgecolor="#CCCCCC")

ax.set_title("MoE Routing Forward Pass — Graph Break Locations",
             fontsize=12, fontweight="bold", color=DARK, pad=12)
ax.text(0.5, 0.005,
    "4 graph breaks → 5 subgraphs.  Dynamo recompiles subgraphs 2–5 for each new batch shape.",
    transform=ax.transAxes, ha="center", fontsize=8, color=GRAY, style="italic")

fig_r.savefig("outputs/routing_diagram.png", dpi=150,
              bbox_inches="tight", facecolor="white")
plt.close(fig_r)
print("saved routing_diagram")
