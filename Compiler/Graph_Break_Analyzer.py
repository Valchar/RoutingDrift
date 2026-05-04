from __future__ import annotations

import re
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch._dynamo
from rich.console import Console
from rich.table import Table

from config import (
    DEVICE, DTYPE, LAYER_TYPES, ModelConfig,
    OLMOE_CONFIG, MIXTRAL_CONFIG,
)

console=Console()


@dataclass
class GraphBreakRecord:
    """One graph-break event from torch._dynamo.explain()."""
    layer_name: str
    layer_type: str
    reason: str
    break_reason_category: str
    op_name: str
    subgraph_index: int


@dataclass
class LayerBreakSummary:
    """Aggregated graph-break stats for one coarse layer type."""
    layer_type: str
    break_count: int = 0
    subgraph_count: int = 0
    reason_counts: Dict[str, int] = field(default_factory=dict)
    ops_causing_breaks: List[str] = field(default_factory=list)


@dataclass
class ModelGraphBreakReport:
    """Full report for one model."""
    model_name: str
    total_breaks: int
    total_subgraphs: int
    pct_compiled: float  # 0.0 - 1.0
    pct_eager: float
    per_layer: Dict[str, LayerBreakSummary]
    raw_records: List[GraphBreakRecord]
    explanation_obj: object

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_breaks": self.total_breaks,
            "total_subgraphs": self.total_subgraphs,
            "pct_compiled": round(self.pct_compiled * 100, 1),
            "pct_eager": round(self.pct_eager * 100, 1),
            "per_layer": {
                k: {
                    "break_count": v.break_count,
                    "subgraph_count": v.subgraph_count,
                    "reason_counts": v.reason_counts,
                    "ops_causing_breaks": v.ops_causing_breaks,
                }
                for k, v in self.per_layer.items()
            },
        }


_BREAK_PATTERNS: List[Tuple[str, str]] = [
    (r"data.dependent", "data-dependent control flow"),
    (r"BreakReasonEnum\.UNSUPPORTED", "unsupported op"),
    (r"index_add", "dynamic in-place scatter"),
    (r"nonzero|torch\.where", "dynamic shape: nonzero/where"),
    (r"one_hot", "dynamic shape: one_hot"),
    (r"python.*for.*loop|for loop", "Python for-loop over experts"),
    (r"graph break in user code", "user code graph break"),
    (r"call_function.*ScatterAdd|scatter", "scatter op"),
    (r"builtin.*any|\.any\(\)", "data-dep branch: .any()"),
    (r"inplace mutation", "in-place mutation"),
    (r"unsupported.*builtin", "unsupported builtin"),
]


def _classify_break_reason(reason: str) -> str:
    for pattern, label in _BREAK_PATTERNS:
        if re.search(pattern, reason, re.IGNORECASE):
            return label
    return "other"


def _classify_layer_type(layer_name: str) -> str:
    name=layer_name.lower()
    if any(k in name for k in ("moe", "sparse_moe", "router", "gate", "expert")):
        return "moe_routing"
    if any(k in name for k in ("attn", "attention", "self_attn")):
        return "attention"
    if any(k in name for k in ("mlp", "ffn", "down_proj", "up_proj", "gate_proj", "w1", "w2", "w3")):
        return "ffn"
    if any(k in name for k in ("norm", "layernorm", "rmsnorm")):
        return "rmsnorm"
    if "embed" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    return "other"


class GraphBreakAnalyzer:
    """
    Wraps torch._dynamo.explain() and produces a ModelGraphBreakReport.

    Usage
    -----
    analyzer = GraphBreakAnalyzer()
    report   = analyzer.run(model, input_ids, model_name="OLMoE")
    """

    def __init__(self, device: str=DEVICE, dtype: torch.dtype=DTYPE):
        self.device=device
        self.dtype=dtype

    def run(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        model_name: str="Model",
        verbose: bool=True,
    ) -> ModelGraphBreakReport:
        """Run torch._dynamo.explain() and return a fully populated report."""
        console.rule(f"[bold cyan]Graph Break Analysis - {model_name}")

        torch._dynamo.reset()

        try:
            explanation=torch._dynamo.explain(model)(input_ids)
        except Exception as exc:
            console.print(f"[red]explain() raised: {exc}[/red]")
            console.print(traceback.format_exc())
            # Return an empty report so the pipeline doesn't crash
            return self._empty_report(model_name)

        records=self._parse_explanation(explanation)
        per_layer=self._aggregate_by_layer(records, explanation)
        total=explanation.break_reasons if hasattr(explanation, "break_reasons") else []
        n_breaks=len(total) if total else len(records)
        n_subgraphs=len(explanation.graphs) if hasattr(explanation, "graphs") else max(1, n_breaks + 1)

        # 1/(breaks+1) approximates the fraction of ops that fall in the first compiled subgraph
        pct_compiled=1.0 / (n_breaks + 1) if n_breaks > 0 else 1.0

        report=ModelGraphBreakReport(
            model_name=model_name,
            total_breaks=n_breaks,
            total_subgraphs=n_subgraphs,
            pct_compiled=pct_compiled,
            pct_eager=1.0 - pct_compiled,
            per_layer=per_layer,
            raw_records=records,
            explanation_obj=explanation,
        )

        if verbose:
            self._print_report(report)

        return report

    def _parse_explanation(self, explanation) -> List[GraphBreakRecord]:
        """
        Extract GraphBreakRecord objects from an ExplainOutput.

        ExplainOutput attributes (as of PyTorch 2.3):
          .break_reasons   - list of ExplainOutput.BreakReason
          .graphs          - list of compiled graph objects
          .ops_per_graph   - list of ops-per-subgraph counts
        """
        records: List[GraphBreakRecord]=[]

        break_reasons=getattr(explanation, "break_reasons", []) or []

        for subgraph_idx, br in enumerate(break_reasons):
            reason_str=str(getattr(br, "reason", str(br)))
            user_stack=getattr(br, "user_stack", [])

            op_name="unknown"
            layer_name="unknown"
            if user_stack:
                top_frame=user_stack[-1]
                op_name=getattr(top_frame, "line", "unknown").strip()
                filename=getattr(top_frame, "filename", "")
                layer_name=self._infer_layer_name(filename, op_name)

            layer_type=_classify_layer_type(layer_name)
            category=_classify_break_reason(reason_str)

            records.append(GraphBreakRecord(
                layer_name=layer_name,
                layer_type=layer_type,
                reason=reason_str,
                break_reason_category=category,
                op_name=op_name,
                subgraph_index=subgraph_idx,
            ))

        if not records:
            records=self._parse_from_str(str(explanation))

        return records

    def _infer_layer_name(self, filename: str, op_line: str) -> str:
        """Best-effort layer name from filename + op line."""
        if "moe" in filename.lower() or "moe" in op_line.lower():
            return "moe_block"
        if "attn" in filename.lower() or "attention" in op_line.lower():
            return "self_attn"
        if "norm" in op_line.lower():
            return "layernorm"
        if "mlp" in filename.lower() or "ffn" in op_line.lower():
            return "ffn"
        return "other"

    def _parse_from_str(self, explain_str: str) -> List[GraphBreakRecord]:
        """Fallback parser for older torch versions that return a plain string."""
        records=[]
        lines=explain_str.split("\n")
        subgraph_idx=0
        for line in lines:
            if "break" in line.lower() or "graph break" in line.lower():
                op_name=line.strip()
                layer_type=_classify_layer_type(line)
                category=_classify_break_reason(line)
                records.append(GraphBreakRecord(
                    layer_name="parsed_from_str",
                    layer_type=layer_type,
                    reason=line.strip(),
                    break_reason_category=category,
                    op_name=op_name,
                    subgraph_index=subgraph_idx,
                ))
                subgraph_idx+=1
        return records

    def _aggregate_by_layer(
        self, records: List[GraphBreakRecord], explanation
    ) -> Dict[str, LayerBreakSummary]:
        """Build per-layer-type summaries."""
        summaries: Dict[str, LayerBreakSummary]={
            lt: LayerBreakSummary(layer_type=lt) for lt in LAYER_TYPES
        }
        summaries["other"]=LayerBreakSummary(layer_type="other")

        for rec in records:
            lt=rec.layer_type if rec.layer_type in summaries else "other"
            s=summaries[lt]
            s.break_count+=1
            s.subgraph_count=rec.subgraph_index + 1
            s.reason_counts[rec.break_reason_category]=(
                s.reason_counts.get(rec.break_reason_category, 0) + 1
            )
            if rec.op_name not in s.ops_causing_breaks:
                s.ops_causing_breaks.append(rec.op_name)

        return {k: v for k, v in summaries.items() if v.break_count > 0}

    def _empty_report(self, model_name: str) -> ModelGraphBreakReport:
        return ModelGraphBreakReport(
            model_name=model_name,
            total_breaks=0,
            total_subgraphs=1,
            pct_compiled=1.0,
            pct_eager=0.0,
            per_layer={},
            raw_records=[],
            explanation_obj=None,
        )

    def _print_report(self, report: ModelGraphBreakReport) -> None:
        console.print(f"\n[bold green]{report.model_name} Graph Break Summary[/bold green]")
        console.print(f"  Total graph breaks : [bold red]{report.total_breaks}[/bold red]")
        console.print(f"  Total subgraphs    : {report.total_subgraphs}")
        console.print(f"  % compiled         : {report.pct_compiled*100:.1f}%")
        console.print(f"  % eager fallback   : {report.pct_eager*100:.1f}%")

        table=Table(title="Per-Layer Break Breakdown", show_lines=True)
        table.add_column("Layer Type", style="cyan")
        table.add_column("Break Count", style="red", justify="right")
        table.add_column("Subgraphs", justify="right")
        table.add_column("Top Reason", style="yellow")
        table.add_column("Ops Causing Break", style="dim")

        for lt, summary in sorted(report.per_layer.items(),
                                   key=lambda x: -x[1].break_count):
            top_reason=max(summary.reason_counts, key=summary.reason_counts.get) \
                         if summary.reason_counts else "-"
            ops=", ".join(summary.ops_causing_breaks[:3])
            table.add_row(
                lt,
                str(summary.break_count),
                str(summary.subgraph_count),
                top_reason,
                ops,
            )

        console.print(table)
        console.print(self._break_reason_explanation())

    def _break_reason_explanation(self) -> str:
        return textwrap.dedent("""
        [bold]Why MoE routing causes graph breaks:[/bold]
          [cyan]1. data-dependent control flow[/cyan]
             torch.topk returns expert indices whose values are unknown at
             trace time. Downstream Python if/for keyed on those values
             forces Dynamo to emit runtime guards -> break when shape changes.

          [cyan]2. dynamic in-place scatter (index_add_)[/cyan]
             index_add_() with a dynamically-sized index tensor produces a
             graph segment that cannot be statically sized. Inductor needs
             a fixed output shape to tile the CUDA kernel.

          [cyan]3. .nonzero() / torch.where() dispatch[/cyan]
             Returns a variadic-length tensor (number of rows = number of
             tokens routed to the expert). Shape is data-dependent; Dynamo
             cannot express this without a data-dependent guard.

          [cyan]4. Python for-loop over experts[/cyan]
             for expert_idx in range(N) is unrolled, but each iteration
             calls index_add_() with a different dynamic mask. Inductor
             cannot fuse across loop iterations with data-dependent indices.

          [cyan]5. in-place mutation inside traced region[/cyan]
             Dynamo assumes tensors are functionally immutable within a
             graph segment. index_add_() on final_hidden violates this,
             forcing a new segment for each mutation.
        """)

def explain_model(
    model: torch.nn.Module,
    model_name: str,
    batch_size: int=1,
    seq_len: int=128,
    device: str=DEVICE,
    dtype: torch.dtype=DTYPE,
) -> ModelGraphBreakReport:
    """Run graph-break analysis on `model` and return the report."""
    input_ids=torch.randint(0, 1000, (batch_size, seq_len), device=device)
    analyzer=GraphBreakAnalyzer(device=device, dtype=dtype)

    with torch.no_grad():
        report=analyzer.run(model, input_ids, model_name=model_name)

    return report
