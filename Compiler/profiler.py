from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from rich.console import Console
from rich.table import Table

from config import (
    BENCH_CFG, DEVICE, DTYPE,
    PROFILER_TRACE_PATH,
)

console=Console()


@dataclass
class LayerFusionStat:
    layer_name: str
    layer_type: str
    python_ops: int
    cuda_kernels: int
    fusion_ratio: float
    fused_op_groups: List[str]
    unfused_ops: List[str]
    wall_time_us: float


@dataclass
class ProfilerReport:
    model_name: str
    total_cuda_kernels: int
    total_python_ops: int
    overall_fusion_ratio: float
    layer_stats: Dict[str, LayerFusionStat]
    top_unfused_ops: List[str]
    top_fused_groups: List[str]
    trace_path: str


class ProfilerAnalyzer:
    """
    Wraps torch.profiler and parses the trace for operator fusion info.

    Usage
    -----
    analyzer = ProfilerAnalyzer()
    report = analyzer.run(model, model_name="OLMoE")
    """

    LAYER_TYPE_MAP={
        "self_attn":  "attention",
        "attention":  "attention",
        "moe_block":  "moe_routing",
        "sparse_moe": "moe_routing",
        "gate":       "moe_routing",
        "expert":     "moe_routing",
        "mlp":        "ffn",
        "ffn":        "ffn",
        "norm":       "rmsnorm",
        "layernorm":  "rmsnorm",
        "rmsnorm":    "rmsnorm",
        "embed":      "embed",
        "lm_head":    "lm_head",
    }

    def __init__(self, device: str=DEVICE, dtype: torch.dtype=DTYPE):
        self.device=device
        self.dtype=dtype

    def run(
        self,
        model: nn.Module,
        model_name: str,
        verbose: bool=True,
    ) -> ProfilerReport:
        console.rule(f"[bold cyan]Profiler Fusion Analysis — {model_name}")

        input_ids=torch.randint(
            0, 1000,
            (BENCH_CFG.batch_size, BENCH_CFG.seq_len),
            device=self.device,
        )

        activities=[ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(ProfilerActivity.CUDA)

        layer_stats: Dict[str, LayerFusionStat]={}

        with profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=BENCH_CFG.profiler_wait,
                warmup=BENCH_CFG.profiler_warmup,
                active=BENCH_CFG.profiler_active,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                PROFILER_TRACE_PATH.replace(".json", ""),
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                for step in range(
                    BENCH_CFG.profiler_wait
                    + BENCH_CFG.profiler_warmup
                    + BENCH_CFG.profiler_active
                ):
                    # Annotate each call so the profiler trace is readable
                    with record_function(f"step_{step}"):
                        _=model(input_ids)
                    prof.step()

        report=self._parse_profiler_events(prof, model_name)

        try:
            prof.export_chrome_trace(PROFILER_TRACE_PATH)
            console.print(f"[dim]Chrome trace saved to {PROFILER_TRACE_PATH}[/dim]")
        except Exception:
            pass

        if verbose:
            self._print_report(report)

        return report

    def _parse_profiler_events(self, prof, model_name: str) -> ProfilerReport:
        key_avgs=prof.key_averages()

        total_cuda_kernels=0
        total_python_ops=0
        layer_stats: Dict[str, LayerFusionStat]={}

        layer_events: Dict[str, List]={}
        for event in key_avgs:
            name=event.key.lower()
            lt=self._classify_layer_type(name)
            if lt not in layer_events:
                layer_events[lt]=[]
            layer_events[lt].append(event)

        for layer_type, events in layer_events.items():
            python_ops=len(events)
            cuda_kernels=sum(
                getattr(e, "count", 1) for e in events
                if "cuda" in e.key.lower() or "kernel" in e.key.lower()
            ) or python_ops  # fallback: assume 1 kernel per op

            total_python_ops+=python_ops
            total_cuda_kernels+=cuda_kernels

            fusion_ratio=cuda_kernels / max(python_ops, 1)
            wall_time_us=sum(
                getattr(e, "self_cpu_time_total", 0) for e in events
            )

            fused_groups, unfused_ops=self._identify_fusion(events)

            layer_stats[layer_type]=LayerFusionStat(
                layer_name=layer_type,
                layer_type=layer_type,
                python_ops=python_ops,
                cuda_kernels=cuda_kernels,
                fusion_ratio=fusion_ratio,
                fused_op_groups=fused_groups,
                unfused_ops=unfused_ops,
                wall_time_us=wall_time_us,
            )

        overall_fusion_ratio=total_cuda_kernels / max(total_python_ops, 1)
        top_unfused=self._top_unfused_ops(layer_stats)
        top_fused=self._top_fused_groups(layer_stats)

        return ProfilerReport(
            model_name=model_name,
            total_cuda_kernels=total_cuda_kernels,
            total_python_ops=total_python_ops,
            overall_fusion_ratio=overall_fusion_ratio,
            layer_stats=layer_stats,
            top_unfused_ops=top_unfused,
            top_fused_groups=top_fused,
            trace_path=PROFILER_TRACE_PATH,
        )

    def _classify_layer_type(self, name: str) -> str:
        for key, lt in self.LAYER_TYPE_MAP.items():
            if key in name:
                return lt
        # Heuristic for CUDA kernels: map common op names to layer types
        if "gemm" in name or "linear" in name or "matmul" in name:
            return "ffn"
        if "softmax" in name or "attention" in name:
            return "attention"
        if "norm" in name:
            return "rmsnorm"
        return "other"

    def _identify_fusion(self, events) -> tuple[List[str], List[str]]:
        """
        Heuristic: ops that appear as children of a single CUDA kernel event
        are considered fused. All others are unfused.
        """
        fused_groups: List[str]=[]
        unfused_ops: List[str]=[]

        # Op groups known to fuse into a single kernel under inductor
        KNOWN_FUSED=[
            ("pow", "mean", "rsqrt", "mul"),  # RMSNorm
            ("exp", "sum", "div"),            # Softmax
            ("silu", "mul"),                  # SwiGLU gate
            ("add", "mul"),                   # residual + scale
        ]

        KNOWN_UNFUSED=[
            "index_add",  # in-place scatter with dynamic index
            "topk",       # data-dependent output values
            "scatter",    # dynamic index scatter
            "nonzero",    # output shape unknown at compile time
            "where",      # data-dependent shape
        ]

        op_names=[e.key for e in events]

        for group in KNOWN_FUSED:
            if all(any(op in name for name in op_names) for op in group):
                fused_groups.append(f"[{', '.join(group)}]")

        for op in KNOWN_UNFUSED:
            if any(op in name for name in op_names):
                unfused_ops.append(op)

        return fused_groups, unfused_ops

    def _top_unfused_ops(self, layer_stats: Dict[str, LayerFusionStat]) -> List[str]:
        ops=[]
        for stat in layer_stats.values():
            ops.extend(stat.unfused_ops)
        return list(dict.fromkeys(ops))  # deduplicate preserving order

    def _top_fused_groups(self, layer_stats: Dict[str, LayerFusionStat]) -> List[str]:
        groups=[]
        for stat in layer_stats.values():
            groups.extend(stat.fused_op_groups)
        return list(dict.fromkeys(groups))

    def _print_report(self, report: ProfilerReport) -> None:
        console.print(f"\n[bold green]{report.model_name} — Operator Fusion Report[/bold green]")
        console.print(f"  Total CUDA kernels : {report.total_cuda_kernels}")
        console.print(f"  Total Python ops   : {report.total_python_ops}")
        console.print(f"  Overall fusion ratio: {report.overall_fusion_ratio:.2f} "
                      f"(lower = more fused, ideal < 0.5)")

        table=Table(title="Per-Layer Fusion Breakdown", show_lines=True)
        table.add_column("Layer Type", style="cyan")
        table.add_column("Python Ops", justify="right")
        table.add_column("CUDA Kernels", justify="right")
        table.add_column("Fusion Ratio", justify="right")
        table.add_column("Fused Groups", style="green")
        table.add_column("Unfused Ops", style="red")
        table.add_column("Wall Time (µs)", justify="right")

        for lt, stat in sorted(report.layer_stats.items(),
                                key=lambda x: -x[1].wall_time_us):
            table.add_row(
                lt,
                str(stat.python_ops),
                str(stat.cuda_kernels),
                f"{stat.fusion_ratio:.2f}",
                ", ".join(stat.fused_op_groups[:2]) or "—",
                ", ".join(stat.unfused_ops[:3]) or "—",
                f"{stat.wall_time_us:,.0f}",
            )

        console.print(table)

        if report.top_unfused_ops:
            console.print(f"\n[red]Unfused ops (separate kernel launch each):[/red]")
            for op in report.top_unfused_ops:
                console.print(f"  • {op}")

        if report.top_fused_groups:
            console.print(f"\n[green]Fused op groups (single kernel):[/green]")
            for grp in report.top_fused_groups:
                console.print(f"  • {grp}")
