"""
results_table.py — MSML 605 (Gokul)
Reads benchmark and profiling CSVs, prints results table, generates all plots.
All graphs compare only baseline vs kernel optimization.

Usage:
    python results_table.py --out /path/to/results
    python results_table.py --out /path/to/results --model Mixtral
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# A100 hardware constants
A100_PEAK_FLOPS=312e12
A100_PEAK_BW=2000e9

# OLMoE-1B-7B architecture constants for roofline
OLMOE_HIDDEN=2048
OLMOE_LAYERS=16
OLMOE_TOP_K=2
OLMOE_FFN_DIM=2048
BYTES_PER_PARAM=2  # fp16

CONFIGS=["baseline", "kernels_only"]
LABELS={"baseline": "Baseline", "kernels_only": "Kernel Opt"}
COLORS={"baseline": "#2196F3", "kernels_only": "#FF9800"}

plt.style.use("seaborn-v0_8-paper")


def load_results(out_dir, model_name):
    path=os.path.join(out_dir, f"benchmark_{model_name.lower()}.csv")
    if not os.path.exists(path):
        print(f"not found: {path}"); return None
    return pd.read_csv(path)


def _rep_pt(seq_len, batch_size, df):
    """Return representative row for a config at the given seq_len and batch_size."""
    return df[(df["seq_len"]==seq_len)&(df["batch_size"]==batch_size)]


def plot_speedup(df, model_name, plots_dir):
    sub=_rep_pt(512, 4, df).set_index("config").reindex(CONFIGS).reset_index()
    fig, ax=plt.subplots(figsize=(6, 5))
    bars=ax.bar([LABELS[c] for c in CONFIGS], sub["speedup"],
                color=[COLORS[c] for c in CONFIGS], edgecolor="white", width=0.5)
    for bar, val in zip(bars, sub["speedup"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Speedup over Baseline")
    ax.set_title(f"{model_name} — Speedup (seq=512, batch=4)")
    ax.set_ylim(0, sub["speedup"].max()*1.25)
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"1_speedup_{model_name}.png")


def plot_latency_vs_seqlen(df, model_name, plots_dir):
    sub=df[df["batch_size"]==4]
    fig, ax=plt.subplots(figsize=(8, 5))
    for cfg in CONFIGS:
        d=sub[sub["config"]==cfg].sort_values("seq_len")
        if d.empty: continue
        ax.plot(d["seq_len"], d["latency_p50_ms"], marker="o",
                label=LABELS[cfg], color=COLORS[cfg], linewidth=2, markersize=6)
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title(f"{model_name} — Latency vs Sequence Length (batch=4)")
    ax.set_xticks(sorted(df["seq_len"].unique()))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"2_latency_seqlen_{model_name}.png")


def plot_latency_vs_batchsize(df, model_name, plots_dir):
    sub=df[df["seq_len"]==512]
    fig, ax=plt.subplots(figsize=(8, 5))
    for cfg in CONFIGS:
        d=sub[sub["config"]==cfg].sort_values("batch_size")
        if d.empty: continue
        ax.plot(d["batch_size"], d["latency_p50_ms"], marker="s",
                label=LABELS[cfg], color=COLORS[cfg], linewidth=2, markersize=6)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title(f"{model_name} — Latency vs Batch Size (seq=512)")
    ax.set_xticks(sorted(df["batch_size"].unique()))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"3_latency_batchsize_{model_name}.png")


def _roofline_point(tokens_per_sec, latency_ms, seq_len, batch_size):
    """
    AI = total_FLOPs / weight_bytes. Weights are read once per forward pass —
    FLOPs scale with batch*seq while bytes do not.
    """
    H,L,K,FFN=OLMOE_HIDDEN,OLMOE_LAYERS,OLMOE_TOP_K,OLMOE_FFN_DIM
    flops_per_token=L*(3*2*H*H+2*H*H+K*3*2*H*FFN)
    weight_bytes=L*((3*H*H+H*H)+K*3*H*FFN)*BYTES_PER_PARAM
    total_flops=flops_per_token*batch_size*seq_len
    ai=total_flops/weight_bytes
    achieved_tflops=(total_flops*(1000.0/latency_ms))/1e12
    return ai, achieved_tflops


def plot_roofline(df, model_name, plots_dir):
    fig, ax=plt.subplots(figsize=(9, 6))
    ai_range=np.logspace(-2, 4, 1000)
    roofline=np.minimum(A100_PEAK_BW*ai_range, np.full_like(ai_range, A100_PEAK_FLOPS))
    ridge=A100_PEAK_FLOPS/A100_PEAK_BW
    ax.loglog(ai_range, roofline/1e12, "k-", linewidth=2, label="A100 Roofline")
    ax.axvline(x=ridge, color="gray", linestyle="--", alpha=0.5, label=f"Ridge ({ridge:.0f} FLOPs/byte)")
    ax.axvspan(1e-2, ridge, alpha=0.05, color="blue", label="Memory-bound")
    ax.axvspan(ridge, 1e4, alpha=0.05, color="orange", label="Compute-bound")
    sub=_rep_pt(512, 4, df)
    for cfg in CONFIGS:
        d=sub[sub["config"]==cfg]
        if d.empty: continue
        ai, tflops=_roofline_point(d["tokens_per_sec"].iloc[0], d["latency_p50_ms"].iloc[0],
                                   int(d["seq_len"].iloc[0]), int(d["batch_size"].iloc[0]))
        peak=min(A100_PEAK_BW*ai, A100_PEAK_FLOPS)/1e12
        pct=tflops/peak*100 if peak>0 else 0
        ax.scatter(ai, tflops, color=COLORS[cfg], s=200, zorder=5, edgecolors="white", linewidth=1.5,
                   label=f"{LABELS[cfg]} — {tflops:.1f} TFLOP/s ({pct:.1f}%)")
        ax.annotate(f"{pct:.1f}%", xy=(ai, tflops), xytext=(12, 4),
                    textcoords="offset points", fontsize=9, color=COLORS[cfg])
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved Performance (TFLOP/s)")
    ax.set_title(f"{model_name} — Roofline (A100, seq=512, batch=4)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"4_roofline_{model_name}.png")


def plot_bandwidth(df, model_name, plots_dir):
    sub=_rep_pt(512, 4, df).set_index("config").reindex(CONFIGS).reset_index()
    fig, ax=plt.subplots(figsize=(6, 5))
    bars=ax.bar([LABELS[c] for c in CONFIGS], sub["bw_utilization"],
                color=[COLORS[c] for c in CONFIGS], edgecolor="white", width=0.5)
    for bar, val in zip(bars, sub["bw_utilization"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="Peak BW")
    ax.set_ylabel("Memory Bandwidth Utilization (%)")
    ax.set_title(f"{model_name} — Bandwidth Utilization")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"5_bandwidth_{model_name}.png")


def plot_latency_percentiles(df, model_name, plots_dir):
    sub=_rep_pt(512, 4, df).set_index("config").reindex(CONFIGS).reset_index()
    x=np.arange(len(CONFIGS)); w=0.25
    fig, ax=plt.subplots(figsize=(7, 5))
    ax.bar(x-w, sub["latency_p50_ms"], w, label="p50", color="#2196F3", alpha=0.9)
    ax.bar(x,   sub["latency_p90_ms"], w, label="p90", color="#FF9800", alpha=0.9)
    ax.bar(x+w, sub["latency_p99_ms"], w, label="p99", color="#E91E63", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONFIGS])
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{model_name} — Latency Percentiles (seq=512, batch=4)")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"6_latency_percentiles_{model_name}.png")


def plot_memory(df, model_name, plots_dir):
    sub=_rep_pt(512, 4, df).set_index("config").reindex(CONFIGS).reset_index()
    fig, ax=plt.subplots(figsize=(6, 5))
    bars=ax.bar([LABELS[c] for c in CONFIGS], sub["peak_mem_mb"]/1024,
                color=[COLORS[c] for c in CONFIGS], edgecolor="white", width=0.5)
    for bar, val in zip(bars, sub["peak_mem_mb"]/1024):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{val:.1f} GB", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title(f"{model_name} — Peak VRAM Usage")
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"7_memory_{model_name}.png")


def plot_throughput(df, model_name, plots_dir):
    """Throughput across all (seq_len, batch_size) combos — baseline vs kernel side by side."""
    fig, ax=plt.subplots(figsize=(10, 5))
    combos=[(s,b) for s in sorted(df["seq_len"].unique()) for b in sorted(df["batch_size"].unique())]
    x=np.arange(len(combos)); w=0.35
    for i, cfg in enumerate(CONFIGS):
        vals=[]
        for s, b in combos:
            row=df[(df["config"]==cfg)&(df["seq_len"]==s)&(df["batch_size"]==b)]
            vals.append(row["tokens_per_sec"].iloc[0] if not row.empty else 0)
        ax.bar(x+(i-0.5)*w, vals, w, label=LABELS[cfg], color=COLORS[cfg], alpha=0.9, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s={s}\nb={b}" for s,b in combos], fontsize=8)
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title(f"{model_name} — Throughput: Baseline vs Kernel Optimization")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"8_throughput_{model_name}.png")


def plot_op_speedup(out_dir, plots_dir, model_name):
    rn_path=os.path.join(out_dir, "profile_rmsnorm_isolated.csv")
    sfx_path=os.path.join(out_dir, "profile_softmax_isolated.csv")
    if not os.path.exists(rn_path) or not os.path.exists(sfx_path):
        print("  skip graph 9: run profile_ops.py first"); return
    rn=pd.read_csv(rn_path); sfx=pd.read_csv(sfx_path)
    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(11, 5))
    rn_k=rn[rn["config"]=="kernel"].sort_values("hidden")
    ax1.bar(range(len(rn_k)), rn_k["speedup"].values, color=COLORS["kernels_only"], edgecolor="white")
    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax1.set_xticks(range(len(rn_k)))
    ax1.set_xticklabels([str(h) for h in rn_k["hidden"].values])
    ax1.set_xlabel("Hidden Size"); ax1.set_ylabel("Speedup (x)")
    ax1.set_title("RMSNorm — Kernel vs Baseline")
    for i, v in enumerate(rn_k["speedup"].values):
        ax1.text(i, v+0.01, f"{v:.2f}x", ha="center", va="bottom", fontsize=9)
    sfx_k=sfx[sfx["config"]=="kernel"]
    ax2.bar(range(len(sfx_k)), sfx_k["speedup"].values, color=COLORS["kernels_only"], edgecolor="white")
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax2.set_xticks(range(len(sfx_k)))
    ax2.set_xticklabels([f"{r['model']} (N={r['num_experts']})" for _, r in sfx_k.iterrows()])
    ax2.set_xlabel("Model"); ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Softmax — Kernel vs Baseline")
    for i, v in enumerate(sfx_k["speedup"].values):
        ax2.text(i, v+0.01, f"{v:.2f}x", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"{model_name} — Per-Kernel Isolated Speedup", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"9_op_speedup_{model_name}.png")


def plot_op_breakdown(out_dir, plots_dir, model_name):
    base_path=os.path.join(out_dir, "profile_model_ops_baseline.csv")
    kern_path=os.path.join(out_dir, "profile_model_ops_kernel.csv")
    if not os.path.exists(base_path) or not os.path.exists(kern_path):
        print("  skip graph 10: run profile_ops.py first"); return
    base=pd.read_csv(base_path).head(10)
    kern=pd.read_csv(kern_path).head(10)
    ops=list(base["op"].values)
    base_t=base.set_index("op")["pct_total"]
    kern_t=kern.set_index("op").reindex(ops, fill_value=0)["pct_total"]
    x=np.arange(len(ops)); w=0.35
    fig, ax=plt.subplots(figsize=(12, 6))
    ax.barh(x+w/2, base_t.values, w, label="Baseline", color=COLORS["baseline"], alpha=0.85)
    ax.barh(x-w/2, kern_t.values, w, label="Kernel Opt", color=COLORS["kernels_only"], alpha=0.85)
    ax.set_yticks(x); ax.set_yticklabels([o[:45] for o in ops], fontsize=8)
    ax.set_xlabel("% of Total CUDA Time")
    ax.set_title(f"{model_name} — Op-Level Time Breakdown (top 10)")
    ax.legend(); ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"10_op_breakdown_{model_name}.png")


def plot_amdahl(out_dir, plots_dir, model_name):
    path=os.path.join(out_dir, "profile_amdahl.csv")
    if not os.path.exists(path):
        print("  skip graph 11: run profile_ops.py first"); return
    d=pd.read_csv(path).set_index("metric")["value"]
    rn_pct=d.get("rmsnorm_pct", 0); sfx_pct=d.get("softmax_pct", 0)
    other=100-rn_pct-sfx_pct
    rn_sp=d.get("rmsnorm_speedup", 1.0); sfx_sp=d.get("softmax_speedup", 1.0)
    avg_sp=d.get("effective_speedup", (rn_sp+sfx_sp)/2)
    combined=d.get("combined_pct", 0)/100
    predicted=d.get("predicted_e2e_speedup", 1.0)
    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(11, 5))
    ax1.pie([rn_pct, sfx_pct, other],
            labels=[f"RMSNorm\n{rn_pct:.1f}%", f"Softmax\n{sfx_pct:.1f}%", f"Other\n{other:.1f}%"],
            colors=[COLORS["kernels_only"], "#4CAF50", "#CCCCCC"],
            startangle=90, autopct="%1.1f%%", pctdistance=0.75)
    ax1.set_title(f"{model_name} — Forward Pass Op Fractions")
    fracs=np.linspace(0, 1, 200)
    curve=1.0/((1.0-fracs)+fracs/avg_sp)
    ax2.plot(fracs*100, curve, "k-", linewidth=2, label=f"Amdahl (effective kernel={avg_sp:.1f}x)")
    ax2.axvline(x=combined*100, color="red", linestyle="--", alpha=0.7,
                label=f"Actual op fraction ({combined*100:.1f}%)")
    ax2.scatter([combined*100], [predicted], color="red", s=150, zorder=5,
                label=f"Predicted e2e = {predicted:.2f}x")
    ax2.set_xlabel("% of Forward Pass in Optimized Ops")
    ax2.set_ylabel("System Speedup (x)")
    ax2.set_title(f"{model_name} — Amdahl Analysis")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_ylim(1.0, avg_sp+0.5)
    plt.tight_layout()
    _save_fig(fig, plots_dir, f"11_amdahl_{model_name}.png")


def print_results_table(df, model_name):
    print(f"\n{'='*80}")
    print(f"  {model_name} — Results (seq=512, batch=4)")
    print(f"{'='*80}")
    print(f"  {'Config':<20} {'p50(ms)':>8} {'p99(ms)':>8} {'Tok/s':>10} {'Mem(GB)':>8} {'BW%':>6} {'Speedup':>8}")
    print(f"  {'-'*72}")
    sub=_rep_pt(512, 4, df)
    for cfg in CONFIGS:
        r=sub[sub["config"]==cfg]
        if r.empty: continue
        r=r.iloc[0]
        print(f"  {cfg:<20} {r['latency_p50_ms']:>8.2f} {r['latency_p99_ms']:>8.2f} "
              f"{r['tokens_per_sec']:>10.0f} {r['peak_mem_mb']/1024:>8.2f} "
              f"{r['bw_utilization']:>6.1f} {r['speedup']:>8.2f}x")


def _save_fig(fig, plots_dir, name):
    os.makedirs(plots_dir, exist_ok=True)
    path=os.path.join(plots_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Generate results table and plots from benchmark CSVs")
    parser.add_argument("--out", required=True, help="Directory containing benchmark CSVs")
    parser.add_argument("--model", default="OLMoE", choices=["OLMoE", "Mixtral"])
    args=parser.parse_args()

    plots_dir=os.path.join(args.out, "plots")
    df=load_results(args.out, args.model)
    if df is None:
        raise SystemExit(f"no benchmark CSV found in {args.out}")

    print_results_table(df, args.model)
    print(f"\ngenerating plots -> {plots_dir}")
    plot_speedup(df, args.model, plots_dir)
    plot_latency_vs_seqlen(df, args.model, plots_dir)
    plot_latency_vs_batchsize(df, args.model, plots_dir)
    plot_roofline(df, args.model, plots_dir)
    plot_bandwidth(df, args.model, plots_dir)
    plot_latency_percentiles(df, args.model, plots_dir)
    plot_memory(df, args.model, plots_dir)
    plot_throughput(df, args.model, plots_dir)
    # profiling graphs — only generated if profile_ops.py has been run
    plot_op_speedup(args.out, plots_dir, args.model)
    plot_op_breakdown(args.out, plots_dir, args.model)
    plot_amdahl(args.out, plots_dir, args.model)
    print("done.")
