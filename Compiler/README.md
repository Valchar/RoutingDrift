# compiler_analysis

Compiler-level analysis of graph breaks and compile performance for Mixture-of-Experts LLMs (OLMoE and Mixtral) using `torch.compile` and `torch._dynamo`.

## Project structure

```
compiler_analysis/
├── config.py                    # Model configs, compile modes, sweep params
├── models/
│   ├── olmoe_retrieve.py        # Lightweight OLMoE MoE routing layer
│   └── mixtral_retrieve.py      # Lightweight Mixtral MoE routing layer
├── analysis/
│   ├── Graph_Break_Analyzer.py  # torch._dynamo.explain() — Graph 11 data
│   ├── Benchmark.py             # 3-mode compile sweep — Graph 12 data
│   ├── ir_inspector.py          # TorchInductor IR for RMSNorm + Softmax
│   └── profiler.py              # torch.profiler op fusion breakdown
├── kernels/
│   └── triton_kernels.py        # Hand-written Triton kernels for comparison
├── metrics/
│   └── test_results.py          # Unified MetricsCollector
├── visualization/
│   ├── Graph_Break_Viz.py       # Graph 11: Graph Break Bar Chart
│   └── graph_compile_mode.py    # Graph 12: Compile Mode Comparison
├── main.py                      # Main entry point that runs everything
└── requirements.txt
```

## Requirements

```
torch>=2.3.0
transformers>=4.40.0
triton>=2.3.0
matplotlib>=3.8.0
numpy>=1.26.0
pandas>=2.2.0
tabulate>=0.9.0
rich>=13.7.0
```

## Setup

```bash
pip install -r requirements.txt
```

CUDA is recommended. The analysis configs fall back to CPU automatically, but Triton kernels and `max-autotune` mode require a CUDA device.

## Usage

```bash
python main.py
```

## What it does

1. Runs `torch._dynamo.explain()` on both models to find all graph breaks, which ops cause them, and how many subgraphs are generated
2. Sweeps all three `torch.compile` modes (`default`, `reduce-overhead`, `max-autotune`) and benchmarks latency, throughput, and VRAM
3. Inspects TorchInductor IR to see what Triton kernels are auto-generated for RMSNorm and Softmax, and compares against hand-written kernels
4. Uses `torch.profiler` to get a per-layer operator fusion breakdown

## Outputs

All outputs are saved to `outputs/`:

| File | Description |
|------|-------------|
| `graph11_graph_breaks.png` | Graph break counts per layer type and reason breakdown |
| `graph12_compile_modes.png` | Latency, throughput, and speedup per compile mode |
| `metrics_summary.json` | Full numeric results for both models |
| `best_compile_mode.txt` | Best compile mode per model for the config matrix |
| `inductor_ir_report.txt` | Auto-generated vs hand-written Triton kernel comparison |

## Graph break summary

All graph breaks in both models originate in the MoE routing layer. Attention, FFN, and RMSNorm compile cleanly.

| Model | Breaks | Subgraphs | Root causes |
|-------|--------|-----------|-------------|
| OLMoE | 4 | 5 | `topk`, `for`-loop over experts, `expert_mask.any()`, `index_add_` |
| Mixtral | 3 | 4 | `topk`, `F.one_hot`, `index_add_` |

## Why MoE routing breaks the graph

- **`torch.topk`** — returns data-dependent expert indices; Dynamo inserts runtime guards that fire on shape changes
- **Python `for` loop over experts** — each iteration applies a different dynamic mask; Inductor cannot fuse across loop body boundaries
- **`expert_mask.any()`** — branch condition depends on routing data; Dynamo cannot resolve it at trace time
- **`index_add_`** — in-place mutation with a variable-length index tensor; forces a new subgraph per loop iteration
