"""
profile_ops.py — MSML 605 (Gokul)
Profiles RMSNorm and Softmax in isolation and inside OLMoE/Mixtral forward pass.
Outputs CSVs for results_table.py profiling graphs.

Usage:
    python profile_ops.py --out /path/to/results --model OLMoE
    python profile_ops.py --out /path/to/results --model Mixtral
"""
import argparse
import csv
import os
import torch
import triton.testing
from torch.profiler import profile, ProfilerActivity
from rms_norm import fused_rms_norm, torch_rms_norm
from softmax import fused_softmax, torch_softmax
from patch_models import load_olmoe, load_mixtral

DEVICE="cuda"
WARMUP=25
REP=100
HIDDEN=2048
SEQ_LEN=512
BATCH=4


def bench(fn):
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)


def profile_rmsnorm_isolated(out_dir):
    print("=== RMSNorm isolated ===")
    rows=[]
    M=BATCH*SEQ_LEN
    for hidden in (512, 1024, 2048, 4096):
        x=torch.randn(M, hidden, dtype=torch.float16, device=DEVICE)
        w=torch.ones(hidden, dtype=torch.float16, device=DEVICE)
        t_base=bench(lambda: torch_rms_norm(x, w))
        t_kern=bench(lambda: fused_rms_norm(x, w))
        mem_bytes=3*M*hidden*2
        rows.append({"op":"rmsnorm","hidden":hidden,"config":"baseline","time_ms":round(t_base,4),"speedup":1.0,"bandwidth_gbs":round((mem_bytes/(t_base/1000))/1e9,2)})
        rows.append({"op":"rmsnorm","hidden":hidden,"config":"kernel","time_ms":round(t_kern,4),"speedup":round(t_base/t_kern,3),"bandwidth_gbs":round((mem_bytes/(t_kern/1000))/1e9,2)})
        print(f"  hidden={hidden:5d} | base={t_base:.3f}ms | kernel={t_kern:.3f}ms | speedup={t_base/t_kern:.2f}x")
    _save(rows, os.path.join(out_dir, "profile_rmsnorm_isolated.csv"))
    return rows


def profile_softmax_isolated(out_dir):
    print("\n=== Softmax isolated ===")
    rows=[]
    M=BATCH*SEQ_LEN
    for name, N in [("OLMoE", 64), ("Mixtral", 8)]:
        x=torch.randn(M, N, dtype=torch.float16, device=DEVICE)
        t_base=bench(lambda: torch_softmax(x))
        t_kern=bench(lambda: fused_softmax(x))
        mem_bytes=2*M*N*2
        rows.append({"op":"softmax","model":name,"num_experts":N,"config":"baseline","time_ms":round(t_base,4),"speedup":1.0,"bandwidth_gbs":round((mem_bytes/(t_base/1000))/1e9,2)})
        rows.append({"op":"softmax","model":name,"num_experts":N,"config":"kernel","time_ms":round(t_kern,4),"speedup":round(t_base/t_kern,3),"bandwidth_gbs":round((mem_bytes/(t_kern/1000))/1e9,2)})
        print(f"  {name:8s} (N={N:3d}) | base={t_base:.3f}ms | kernel={t_kern:.3f}ms | speedup={t_base/t_kern:.2f}x")
    _save(rows, os.path.join(out_dir, "profile_softmax_isolated.csv"))
    return rows


def profile_model_ops(out_dir, model_name="OLMoE", kernels=False):
    label="kernel" if kernels else "baseline"
    print(f"\n=== {model_name} op profile [{label}] ===")
    load_fn=load_olmoe if model_name=="OLMoE" else load_mixtral
    precision="fp16" if model_name=="OLMoE" else "gptq"
    model, tok=load_fn(precision=precision, kernels=kernels)
    inputs={"input_ids": torch.randint(0, model.config.vocab_size, (BATCH, SEQ_LEN), device=DEVICE)}
    for _ in range(5):
        with torch.no_grad(): model(**inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        with torch.no_grad(): model(**inputs)
    avgs=prof.key_averages()
    total=sum(a.cuda_time for a in avgs)
    rows=[]
    for a in sorted(avgs, key=lambda x: x.cuda_time, reverse=True)[:15]:
        rows.append({"config":label,"op":a.key,"cuda_time_us":round(a.cuda_time,1),"pct_total":round(a.cuda_time/total*100,2) if total>0 else 0,"count":a.count})
        print(f"  {a.key:50s} {a.cuda_time:8.1f} us  {a.cuda_time/total*100:5.1f}%")
    del model; torch.cuda.empty_cache()
    _save(rows, os.path.join(out_dir, f"profile_model_ops_{label}.csv"))
    return rows, total


def compute_amdahl(out_dir, rn_rows, sfx_rows, base_ops, model_name="OLMoE"):
    rn_kw=["elementwise_kernel", "vectorized_elementwise", "unrolled_elementwise", "pow_tensor"]
    sfx_kw=["softmax"]
    ops_clean=[{**r, "op": r["op"].strip('"')} for r in base_ops]
    rn_pct=sum(r["pct_total"] for r in ops_clean if any(k in r["op"] for k in rn_kw))/100
    sfx_pct=sum(r["pct_total"] for r in ops_clean if any(k in r["op"].lower() for k in sfx_kw))/100
    combined=rn_pct+sfx_pct
    rn_speedup=next((r["speedup"] for r in rn_rows if r["config"]=="kernel" and r["hidden"]==HIDDEN), 1.0)
    sfx_speedup=next((r["speedup"] for r in sfx_rows if r["config"]=="kernel" and r["model"]==model_name), 1.0)
    # Combine operation speedups by their contribution to the optimized fraction.
    effective_speedup=1.0/((rn_pct/combined)/rn_speedup + (sfx_pct/combined)/sfx_speedup) if combined > 0 else 1.0
    # Amdahl: max system speedup = 1 / ((1-f) + f/s)
    predicted=1.0/((1.0-combined)+combined/effective_speedup)
    rows=[
        {"metric":"rmsnorm_pct","value":round(rn_pct*100,2)},
        {"metric":"softmax_pct","value":round(sfx_pct*100,2)},
        {"metric":"combined_pct","value":round(combined*100,2)},
        {"metric":"rmsnorm_speedup","value":round(rn_speedup,3)},
        {"metric":"softmax_speedup","value":round(sfx_speedup,3)},
        {"metric":"effective_speedup","value":round(effective_speedup,3)},
        {"metric":"predicted_e2e_speedup","value":round(predicted,3)},
    ]
    _save(rows, os.path.join(out_dir, "profile_amdahl.csv"))
    print(f"\n=== Amdahl ===")
    print(f"  RMSNorm {rn_pct*100:.1f}% | Softmax {sfx_pct*100:.1f}% | combined {combined*100:.1f}%")
    print(f"  RMSNorm speedup {rn_speedup:.2f}x | Softmax speedup {sfx_speedup:.2f}x")
    print(f"  Predicted e2e speedup: {predicted:.2f}x")


def _save(rows, path):
    if not rows: return
    with open(path, "w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"saved: {path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Profile RMSNorm/Softmax kernels and model op breakdown")
    parser.add_argument("--out", required=True, help="Output directory for profiling CSVs")
    parser.add_argument("--model", default="OLMoE", choices=["OLMoE", "Mixtral"], help="Model to profile")
    args=parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rn_rows=profile_rmsnorm_isolated(args.out)
    sfx_rows=profile_softmax_isolated(args.out)
    base_ops, _=profile_model_ops(args.out, model_name=args.model, kernels=False)
    profile_model_ops(args.out, model_name=args.model, kernels=True)
    compute_amdahl(args.out, rn_rows, sfx_rows, base_ops, model_name=args.model)
    print("\nDone.")
