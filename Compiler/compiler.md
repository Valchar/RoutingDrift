# compiler.md

This file provides guidance when working with code in this repository.

## What This Is

A `torch.compile` analysis framework that studies why Mixture-of-Experts (MoE) routing layers cause graph breaks in PyTorch's dynamic compilation system, and benchmarks compile modes across OLMoE and Mixtral models.

## Commands

```bash
pip install -r requirements.txt   # setup
python main.py                     # run full analysis pipeline
```

CUDA is needed for Triton kernels and `max-autotune` mode; graph break analysis and default/reduce-overhead modes run on CPU.

## Pipeline Architecture

`main.py` runs five sequential phases:

1. **Model Construction** — `olmoe_retrieve.build_olmoe()` / `mixtral_retrieve.build_mixtral()` build lightweight stubs with real MoE routing logic.

2. **Graph Break Analysis** (`Graph_Break_Analyzer.py`) — wraps `torch._dynamo.explain()`, parses break reasons, classifies by layer type and cause (topk, for-loop, index_add_, one_hot). Produces `ModelGraphBreakReport`.

3. **Compile Mode Sweep** (`Benchmark.py`) — runs `eager`, `default`, `reduce-overhead`, `max-autotune` with warmup + timed iterations; records p50/p90/p99 latency, throughput, speedup, VRAM overhead. Produces `CompileBenchmarkReport`.

4. **IR Inspection & Profiling** (`ir_inspector.py`, `profiler.py`) — optional; captures auto-generated Triton code from Inductor and compares it to hand-written kernels in `triton_kernels.py`; measures per-layer fusion ratios.

5. **Metrics & Visualization** (`test_results.py`, `Graph_Break_Viz.py`, `graph_compile_mode.py`) — aggregates results into JSON, writes best-mode recommendation, renders two PNG charts.

**Outputs:**
- `outputs/metrics_summary.json`
- `outputs/best_compile_mode.txt`
- `outputs/graph11_graph_breaks.png`
- `outputs/graph12_compile_modes.png`
- `outputs/profiler_trace/` (Chrome traces)

## Why MoE Routing Breaks

The four break sources in `OLMoESparseMoeBlock` / `MixtralSparseMoeBlock`:

| Op | Reason |
|---|---|
| `torch.topk` | Data-dependent indices → runtime guards |
| `for expert_idx in range(num_experts)` | Python loop with dynamic per-iteration mask |
| `if expert_mask.any()` | Data-dependent branch condition |
| `index_add_()` | In-place scatter with dynamic-length index tensor |

OLMoE: 4 breaks → 5 subgraphs → ~20% compiled. Mixtral: 3 breaks → 4 subgraphs.

## Configuration

All constants live in `config.py`:
- `OLMOE_CONFIG` / `MIXTRAL_CONFIG` — full model architecture params
- `OLMOE_ANALYSIS_CONFIG` / `MIXTRAL_ANALYSIS_CONFIG` — slim CPU configs for fast iteration
- `BenchmarkConfig` — batch_size, seq_len, warmup_iters, timed_iters
- `DEVICE` / `DTYPE` — auto-detected from CUDA availability

To run a single phase in isolation:

```python
from Graph_Break_Analyzer import explain_model
from olmoe_retrieve import build_olmoe
from config import OLMOE_ANALYSIS_CONFIG, DEVICE, DTYPE

model = build_olmoe(OLMOE_ANALYSIS_CONFIG, DEVICE, DTYPE)
report = explain_model(model, "OLMoE", device=DEVICE, dtype=DTYPE)
print(report.total_breaks, report.pct_compiled)
```

## Module Map

| File | Role |
|---|---|
| `main.py` | Orchestrates all 5 phases |
| `config.py` | All constants and model configs |
| `olmoe_retrieve.py` / `mixtral_retrieve.py` | Lightweight model stubs |
| `Graph_Break_Analyzer.py` | Break detection; key dataclasses at lines 24–72 |
| `Benchmark.py` | Compile sweep; key dataclasses at lines 24–64 |
| `test_results.py` | Unified metrics; `Metrics` dataclass at lines 22–39 |
| `ir_inspector.py` | Inductor IR capture and kernel comparison |
| `profiler.py` | Fusion ratio measurement |
| `triton_kernels.py` | Hand-written RMSNorm + Softmax Triton kernels |
| `Graph_Break_Viz.py` | graph11 chart |
| `graph_compile_mode.py` | graph12 chart |
| `analysis.py`, `models.py`, `metrics.py`, `kernels.py` | Re-export facades |
