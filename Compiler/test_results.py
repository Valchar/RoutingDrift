
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from config import (
    BEST_MODE_PATH, COMPILE_MODES, LAYER_TYPES, METRICS_PATH, OUTPUT_DIR
)
from Graph_Break_Analyzer import ModelGraphBreakReport
from Benchmark import CompileBenchmarkReport
from dense_comparison import DenseComparisonResult

console=Console()


@dataclass
class Metrics:
    """
    Complete metrics record for one (model, run).
    This is exactly what Amogh hands to the rest of the team.
    """
    model_name: str

    total_graph_breaks: int
    total_subgraphs: int
    pct_compiled: float  # 0–100
    pct_eager: float     # 0–100
    breaks_per_layer: Dict[str, int]
    break_reasons: Dict[str, Dict[str, int]]
    ops_causing_breaks: Dict[str, List[str]]

    compile_results: Dict[str, Dict[str, Any]]
    best_compile_mode: str
    dense_comparison: Optional[DenseComparisonResult] = None

class MetricsCollector:
    """
    Aggregates ModelGraphBreakReport + CompileBenchmarkReport into Metrics.
    Persists results to JSON and writes the best_compile_mode recommendation.

    Usage
    -----
    collector = MetricsCollector()
    collector.add_graph_break_report(olmoe_report)
    collector.add_compile_report(olmoe_compile_report)
    collector.add_graph_break_report(mixtral_report)
    collector.add_compile_report(mixtral_compile_report)
    collector.finalize()          # writes JSON + best_mode files
    metrics = collector.get("OLMoE")
    """

    def __init__(self):
        self._graph_break_reports: Dict[str, ModelGraphBreakReport]={}
        self._compile_reports: Dict[str, CompileBenchmarkReport]={}
        self._metrics: Dict[str, Metrics]={}

    def add_graph_break_report(self, report: ModelGraphBreakReport) -> None:
        self._graph_break_reports[report.model_name]=report
        self._try_build(report.model_name)

    def add_compile_report(self, report: CompileBenchmarkReport) -> None:
        self._compile_reports[report.model_name]=report
        self._try_build(report.model_name)

    def add_dense_comparison(self, result: DenseComparisonResult) -> None:
        m=self._metrics.get(result.moe_name)
        if m:
            m.dense_comparison=result

    def _try_build(self, model_name: str) -> None:
        if model_name not in self._graph_break_reports:
            return
        if model_name not in self._compile_reports:
            return

        gb=self._graph_break_reports[model_name]
        cm=self._compile_reports[model_name]

        breaks_per_layer={
            lt: summary.break_count
            for lt, summary in gb.per_layer.items()
        }
        break_reasons={
            lt: summary.reason_counts
            for lt, summary in gb.per_layer.items()
        }
        ops_causing_breaks={
            lt: summary.ops_causing_breaks
            for lt, summary in gb.per_layer.items()
        }

        compile_results: Dict[str, Dict[str, Any]]={}
        for mode, result in cm.results.items():
            compile_results[mode]={
                "compile_time_s": result.compile_time_s,
                "p50_ms": result.p50_ms,
                "p90_ms": result.p90_ms,
                "p99_ms": result.p99_ms,
                "throughput_tps": result.throughput_tps,
                "speedup": result.speedup,
                "peak_vram_mb": result.peak_vram_mb,
                "compile_vram_overhead_mb": result.compile_vram_overhead_mb,
            }

        self._metrics[model_name]=Metrics(
            model_name=model_name,
            total_graph_breaks=gb.total_breaks,
            total_subgraphs=gb.total_subgraphs,
            pct_compiled=round(gb.pct_compiled * 100, 1),
            pct_eager=round(gb.pct_eager * 100, 1),
            breaks_per_layer=breaks_per_layer,
            break_reasons=break_reasons,
            ops_causing_breaks=ops_causing_breaks,
            compile_results=compile_results,
            best_compile_mode=cm.best_mode,
        )

    def finalize(self) -> None:
        """Write JSON metrics + best_mode file to disk."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        payload={}
        for model, m in self._metrics.items():
            entry={
                "graph_11": {
                    "total_graph_breaks": m.total_graph_breaks,
                    "total_subgraphs": m.total_subgraphs,
                    "pct_compiled": m.pct_compiled,
                    "pct_eager": m.pct_eager,
                    "breaks_per_layer": m.breaks_per_layer,
                    "break_reasons": m.break_reasons,
                    "ops_causing_breaks": m.ops_causing_breaks,
                },
                "graph_12": {
                    "best_compile_mode": m.best_compile_mode,
                    "compile_results": m.compile_results,
                },
            }
            if m.dense_comparison:
                entry["graph_14"]=m.dense_comparison.as_dict()
            payload[model]=entry

        Path(METRICS_PATH).write_text(json.dumps(payload, indent=2))
        console.print(f"[green]Metrics written to {METRICS_PATH}[/green]")

        # best_compile_mode per model is the key output for the team
        best_modes={
            model: m.best_compile_mode
            for model, m in self._metrics.items()
        }
        Path(BEST_MODE_PATH).write_text(json.dumps(best_modes, indent=2))
        console.print(f"[green]Best compile modes written to {BEST_MODE_PATH}[/green]")
        self._print_best_modes(best_modes)

    def get(self, model_name: str) -> Optional[Metrics]:
        return self._metrics.get(model_name)

    def all_metrics(self) -> Dict[str, Metrics]:
        return dict(self._metrics)

    def print_summary(self) -> None:
        """Print the full metrics table to console."""
        console.rule("[bold cyan]Amogh Full Metrics Summary")

        for model_name, m in self._metrics.items():
            console.print(f"\n[bold]Model: {model_name}[/bold]")

            table11=Table(title=f"Graph 11 — Graph Breaks", show_lines=True)
            table11.add_column("Layer Type", style="cyan")
            table11.add_column("Break Count", style="red", justify="right")
            table11.add_column("Top Reason", style="yellow")
            table11.add_column("Ops Causing", style="dim")

            for lt, count in sorted(m.breaks_per_layer.items(), key=lambda x: -x[1]):
                reasons=m.break_reasons.get(lt, {})
                top_r=max(reasons, key=reasons.get) if reasons else "—"
                ops=", ".join(m.ops_causing_breaks.get(lt, [])[:2]) or "—"
                table11.add_row(lt, str(count), top_r, ops)

            console.print(table11)
            console.print(
                f"  Total breaks: [red]{m.total_graph_breaks}[/red]  "
                f"Subgraphs: {m.total_subgraphs}  "
                f"Compiled: {m.pct_compiled:.1f}%  "
                f"Eager: {m.pct_eager:.1f}%"
            )

            table12=Table(title="Graph 12 — Compile Mode Results", show_lines=True)
            table12.add_column("Mode", style="cyan")
            table12.add_column("p50 (ms)", justify="right")
            table12.add_column("p90 (ms)", justify="right")
            table12.add_column("p99 (ms)", justify="right")
            table12.add_column("tok/s", justify="right")
            table12.add_column("Speedup", style="green", justify="right")
            table12.add_column("VRAM (MB)", justify="right")

            mode_order=["eager"] + COMPILE_MODES
            for mode in mode_order:
                if mode not in m.compile_results:
                    continue
                r=m.compile_results[mode]
                best_star=" ★" if mode == m.best_compile_mode else ""
                table12.add_row(
                    mode + best_star,
                    f"{r['p50_ms']:.2f}",
                    f"{r['p90_ms']:.2f}",
                    f"{r['p99_ms']:.2f}",
                    f"{r['throughput_tps']:,.0f}",
                    f"{r['speedup']:.2f}x",
                    f"{r['peak_vram_mb']:.0f}",
                )

            console.print(table12)

    def _print_best_modes(self, best_modes: Dict[str, str]) -> None:
        console.print("\n[bold]Best compile mode per model (→ config matrix):[/bold]")
        for model, mode in best_modes.items():
            console.print(f"  {model}: [magenta]{mode}[/magenta]")
