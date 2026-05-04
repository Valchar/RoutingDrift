# Routing Fidelity as a Systems Metric: Characterizing Optimization-Induced Drift in MoE

**MSML 605 Final Project · University of Maryland**

---

## What is this?

Mixture-of-Experts (MoE) models like OLMoE and Mixtral work by routing each token to a small subset of specialized "expert" sub-networks instead of running everything through one big dense network. That makes them much more parameter-efficient at inference time, but it also makes them surprisingly painful to optimize.

The routing decision is dynamic and data-dependent, which breaks a lot of the tools you'd normally reach for. `torch.compile` trips over the routing logic and can't fuse key ops. Quantizing the weights shifts the routing distribution in ways that may or may not hurt accuracy. And custom kernels can interact badly with quantized weight formats.

This project looks at all three of those problems together. We picked two real MoE models, **OLMoE-1B-7B** (64 experts, top-8 routing, 16 layers) and **Mixtral-8x7B-GPTQ** (8 experts, top-2 routing), and ran three independent but complementary sub-studies on an A100 GPU on the Zaratan HPC cluster at UMD.

---

## The Three Sub-Studies

### 1. Triton Kernel Optimization (`kernals/`)

We wrote hand-tuned Triton kernels for two ops that show up in every layer of both models: **RMSNorm** and the routing **Softmax**. The idea was to fuse the two-pass variance + normalize computation for RMSNorm into a single kernel pass, and do the same for the row-wise softmax over expert gate logits.

In isolation, the kernels are genuinely fast:

| Op       | Config                  | Baseline   | Kernel     | Speedup  | Bandwidth  |
|----------|-------------------------|------------|------------|----------|------------|
| RMSNorm  | hidden=512              | 0.072 ms   | 0.013 ms   | **5.7x** | 501 GB/s   |
| RMSNorm  | hidden=2048             | 0.142 ms   | 0.019 ms   | **7.3x** | 1300 GB/s  |
| RMSNorm  | hidden=4096             | 0.303 ms   | 0.033 ms   | **9.2x** | 1521 GB/s  |
| Softmax  | OLMoE (64 experts)      | 0.017 ms   | 0.009 ms   | **2.0x** | 61 GB/s    |
| Softmax  | Mixtral (8 experts)     | 0.017 ms   | 0.008 ms   | **2.2x** | 9 GB/s     |

The catch is Amdahl's Law. RMSNorm only takes up **1.91% of OLMoE's forward pass time**, and Softmax is basically nothing. So even a 7.3x isolated speedup gives you a predicted E2E ceiling of just **1.015x**. The measured E2E confirmed this: OLMoE with Triton kernels runs at **0.985x** the baseline at seq=512 batch=4 because the kernel's launch overhead doesn't amortize at small batch sizes.

**Bottom line:** the kernels are correct and memory-efficient. The E2E ceiling is set by Amdahl, not kernel quality.

---

### 2. Routing Drift Under Quantization (`quantization/`)

The core question here: when you quantize a MoE model to INT8 or INT4, does the routing actually change? If tokens end up getting sent to completely different experts after quantization, the model's specialized knowledge is effectively scrambled, even if the numerical outputs look close on the surface.

We hooked the gate layer in OLMoE-1B-7B, ran the same prompts through FP16, INT8, and INT4, and compared expert selections using four metrics:

| Precision | Routing Similarity | Jaccard Drift | Overlap@k | Selection Shift |
|-----------|--------------------|---------------|-----------|-----------------|
| FP16      | 1.0000             | 0.0000        | 1.0000    | 0.0000          |
| INT8      | 0.9545             | 0.0455        | 0.9659    | 0.0341          |
| INT4      | 0.9333             | 0.0667        | 0.9497    | 0.0503          |

The routing is remarkably stable. At INT8, 95.5% of tokens still go to the same experts as FP16. At INT4, still 93.3%. The reason is that quantization shifts gate logit values slightly, but the top-k selection is dominated by large margin differences between experts, so small perturbations rarely flip the ranking. Degradation is also monotonic across all four metrics: FP16 > INT8 > INT4, no surprises.

We also ran a per-layer breakdown across all 16 layers to capture which layers drift the most under quantization. That's useful input for future mixed-precision schemes that could selectively protect the most routing-sensitive layers.

For an accuracy reference we ran `lm-eval` on the FP16 model:

| Benchmark  | Score  | Notes                                               |
|------------|--------|-----------------------------------------------------|
| MMLU       | 52.8%  | Slightly above chance, expected for 1B active params |
| HellaSwag  | 78.3%  | (acc_norm) Solid commonsense performance            |
| GSM8K      | 8.1%   | Matches the ~8% reported in the OLMoE paper         |

The main next step for this sub-study is closing the loop with INT8/INT4 accuracy evals to verify that low routing drift actually preserves downstream accuracy.

---

### 3. Why `torch.compile` Struggles with MoE (`Compiler/`)

`torch.compile` traces PyTorch code into a computation graph and fuses ops via TorchInductor. It works great on dense models. MoE routing breaks it because the routing logic is inherently data-dependent, so `torch.compile` can't trace through dynamic branches or dynamic shapes and falls back to eager mode at those points.

Both OLMoE and Mixtral produce **exactly 1 graph break**, both in the MoE routing layer. The compiled subgraphs cover Attention and FFN ops (which trace cleanly), but the routing kernel itself always runs in eager mode. That means only **50% of the graph** ever gets compiled for either model.

We swept all four compile modes on both models:

| Compile Mode     | OLMoE Speedup | OLMoE p50 (ms) | Mixtral Speedup | Mixtral p50 (ms) |
|------------------|---------------|----------------|-----------------|------------------|
| eager (baseline) | 1.000x        | 5.80           | 1.000x          | 5.10             |
| default          | 0.945x        | 6.14           | 0.890x          | 5.74             |
| reduce-overhead  | **1.032x**    | 5.62           | 0.999x          | 5.11             |
| max-autotune     | 1.032x        | 5.62           | **1.005x**      | 5.08             |

Two things stand out. First, `default` mode is actually *slower* than eager on both models because the compilation overhead outweighs the benefit when half the graph runs in eager anyway. Second, the best gains are modest: 3.2% for OLMoE (`reduce-overhead`) and 0.5% for Mixtral (`max-autotune`). The single graph break in the routing layer is a hard structural ceiling.

`torch.compile(dynamic=True)` looks like the most promising path to removing the break entirely, though we didn't test it in this study.

---

## Repo Structure

```
RoutingDrift/
│
├── kernals/                         # Sub-study 1: Custom Triton kernels
│   ├── rms_norm.py                  # Fused RMSNorm Triton kernel
│   ├── softmax.py                   # Row-wise Softmax Triton kernel
│   ├── patch_models.py              # Monkey-patches OLMoE/Mixtral to use custom ops
│   ├── validate_olmoe.py            # Numerical correctness tests for OLMoE
│   ├── validate_mixtral.py          # Numerical correctness tests for Mixtral
│   ├── benchmark.py                 # E2E latency sweep (seq_len x batch_size)
│   ├── profile_ops.py               # Isolated kernel profiling + Amdahl breakdown
│   ├── nsight_proxy.py              # Bandwidth / occupancy / roofline (no ncu needed)
│   ├── eval_accuracy.py             # lm-eval accuracy check on patched vs baseline
│   ├── results_table.py             # Reads CSVs -> summary table + 8 plots
│   ├── a100_results/olmoe/          # OLMoE A100 results (benchmark + profile CSVs)
│   └── results/
│       ├── olmoe/                   # OLMoE benchmark + isolated kernel CSVs + plots
│       └── mixtral/mixtral/         # Mixtral benchmark + isolated kernel CSVs + plots
│
├── quantization/                    # Sub-study 2: Routing drift under quantization
│   ├── run_experiment.py            # Pipeline: load -> hook router -> run -> compute drift
│   ├── drift.py                     # Routing metrics (RS, Jaccard, Overlap@k, Shift)
│   ├── routing_logger.py            # Gate layer hook to capture per-token expert indices
│   ├── model_loader.py              # Unified FP16 / INT8 / INT4 / GPTQ model loader
│   ├── harness_eval.py              # lm-eval integration (MMLU, GSM8K, HellaSwag)
│   └── results_olmoe_datasets/
│       ├── routing_drift_summary.csv    # Per-precision drift metrics (4 metrics x 3 precisions)
│       ├── routing_drift_layers.csv     # Per-layer drift breakdown (16 layers)
│       ├── routes_fp16/int8/int4.json   # Raw per-token expert selections
│       └── lm_eval/lm_eval_fp16.json   # FP16 accuracy baseline
│
├── Compiler/                        # Sub-study 3: torch.compile graph break analysis
│   ├── main.py                      # 5-phase pipeline orchestrator
│   ├── Graph_Break_Analyzer.py      # Wraps torch._dynamo.explain(); classifies breaks
│   ├── Benchmark.py                 # Compile mode sweep + latency measurement
│   ├── olmoe_retrieve.py            # Lightweight OLMoE stub with real routing logic
│   ├── mixtral_retrieve.py          # Lightweight Mixtral stub
│   ├── ir_inspector.py              # Inspects TorchInductor auto-generated Triton IR
│   └── outputs/
│       ├── metrics_summary.json     # Graph breaks, compile speedups, best modes
│       └── profiler_trace/          # Chrome trace timeline files (Zaratan A100 runs)
│
└── report/                          # Cross-study aggregation
    ├── generate_report.py           # Reads all CSVs/JSONs -> 11 comparison plots
    └── plots/                       # Generated figures (01-11)
```

---

## How to Run

**Prerequisites**
```bash
pip install -r requirements.txt
# For Mixtral GPTQ support:
pip install auto-gptq optimum
```

---

### Sub-study 1 — Kernel Optimization *(GPU required)*

All commands run from the repo root. Results land in `kernals/results/olmoe/` and `kernals/results/mixtral/mixtral/`.

**Step 1 — Validate kernels are numerically correct**

Run this before anything else. It tests RMSNorm and Softmax on random tensors and on a live model forward pass to confirm the kernels match PyTorch outputs within tolerance.

```bash
# OLMoE tests RMSNorm + Softmax + patched forward pass
python kernals/validate_olmoe.py

# Mixtral tests patched load and router module detection
python kernals/validate_mixtral.py
```

**Step 2 — E2E benchmark: baseline vs Triton kernels**

Sweeps all combinations of `seq_len in {128, 512, 1024}` and `batch_size in {1, 4}`. Measures latency p50/p90/p99, throughput (tokens/sec), peak VRAM, and speedup. Outputs `benchmark_<model>.csv`.

```bash
python kernals/benchmark.py --model OLMoE --out kernals/results/olmoe
python kernals/benchmark.py --model Mixtral --out kernals/results/mixtral/mixtral
```

**Step 3 — Isolated kernel profiling + Amdahl breakdown**

Benchmarks RMSNorm and Softmax in isolation across hidden sizes, then profiles a full model forward pass to measure what fraction of runtime each op takes. Outputs `profile_rmsnorm_isolated.csv`, `profile_softmax_isolated.csv`, `profile_model_ops_baseline.csv`, `profile_model_ops_kernel.csv`, and `profile_amdahl.csv`.

```bash
python kernals/profile_ops.py --model OLMoE --out kernals/results/olmoe
python kernals/profile_ops.py --model Mixtral --out kernals/results/mixtral/mixtral
```

**Step 4 — Nsight-proxy profiling (memory bandwidth + occupancy)**

Computes arithmetic intensity, achieved memory bandwidth (GB/s, % of A100 peak), compute throughput, estimated occupancy, and roofline region without needing `ncu`. Outputs `profile_nsight_proxy.csv`.

```bash
python kernals/nsight_proxy.py --out kernals/results/olmoe
```

**Step 5 — Accuracy validation (kernel correctness on real tasks)**

Confirms the patched model produces the same downstream accuracy as the baseline on GSM8K and MMLU. Runs lm-eval on baseline, kernels_only, int8, and int4 configs.

```bash
python kernals/eval_accuracy.py
```

**Step 6 — Generate results tables and plots**

Reads all CSVs produced above and generates 8 plots (speedup, latency vs seq_len, latency vs batch_size, roofline, bandwidth, latency percentiles, memory, throughput) plus a printed summary table. Plots land in `kernals/results/<model>/plots/`.

```bash
python kernals/results_table.py --model OLMoE --out kernals/results/olmoe
python kernals/results_table.py --model Mixtral --out kernals/results/mixtral/mixtral
```

---

### Sub-study 2 — Routing Drift *(GPU required)*

```bash
# Run drift experiment across FP16, INT8, INT4
python quantization/run_experiment.py --model OLMoE --precisions fp16 int8 int4

# Run lm-eval accuracy baseline (FP16)
python quantization/harness_eval.py --model OLMoE --tasks mmlu gsm8k hellaswag
```

---

### Sub-study 3 — Compiler Analysis *(CPU-friendly stubs available)*

```bash
python Compiler/main.py --model OLMoE
python Compiler/main.py --model Mixtral
```

---

### Cross-Study Report *(no GPU needed -- reads existing results)*

```bash
python report/generate_report.py --out report/plots
```

---

## Key Findings

1. **Triton kernels hit large isolated speedups but minimal E2E gain on OLMoE.** RMSNorm achieves 7.3x in microbenchmark but only occupies 1.91% of total runtime, giving an Amdahl ceiling of 1.015x. Measured E2E came out to 0.985x at seq=512.

2. **Triton kernels catastrophically regress on Mixtral-GPTQ.** 0.037x measured speedup (27x slower) because of packed INT4 memory layout incompatibility. The kernel is correct; the format is incompatible.

3. **Quantization barely moves the routing distribution.** INT8 gets 95.5% routing similarity to FP16; INT4 gets 93.3%. Expert selection is robust because the ranking margins between experts are large enough that small quantization perturbations rarely flip the top-k.

4. **MoE routing structurally limits `torch.compile`.** Both models produce exactly 1 graph break in the routing layer, keeping 50% of the graph in eager mode. Best compile-mode gain is 3.2% for OLMoE and 0.5% for Mixtral. `default` mode is slower than eager on both.

5. **INT8 quantization is the most deployable optimization overall.** Lowest routing drift (4.6%), no compilation instability, and it works on both model families. The ~7% drift at INT4 is still well within the stable routing regime.

---

## Team

| Person | Role |
|--------|------|
| Gokul  | Triton kernel engineering (RMSNorm + Softmax), HPC runs on Zaratan |
| Amogh  | Compiler: graph break analysis, `torch.compile` mode sweep, TorchInductor IR inspection |
| Giri   | Quantization: routing drift metrics, per-layer analysis, lm-eval accuracy baseline |

*MSML 605 · University of Maryland · Spring 2026*