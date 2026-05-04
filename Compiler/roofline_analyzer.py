from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import torch

from config import (
    BENCH_CFG, DEVICE, DTYPE, HARDWARE_SPECS,
    OUTPUT_DIR, ROOFLINE_DATA_PATH, ModelConfig,
)


@dataclass
class HardwareSpec:
    device_name: str
    peak_flops_tflops: float  # fp16/bf16 TFLOPs
    peak_bandwidth_gbps: float  # HBM bandwidth GB/s

    @property
    def ridge_point(self) -> float:
        """FLOPs/byte at the compute/memory boundary."""
        return (self.peak_flops_tflops * 1e12) / (self.peak_bandwidth_gbps * 1e9)


@dataclass
class OpRooflineRecord:
    op_name: str
    layer_type: str
    flops: float
    bytes_accessed: float
    arithmetic_intensity: float  # FLOPs / byte
    bound_type: str  # "compute" or "memory"


@dataclass
class RooflineReport:
    model_name: str
    hw: HardwareSpec
    records: List[OpRooflineRecord]=field(default_factory=list)
    ridge_point: float=0.0
    memory_bound_ops: int=0
    compute_bound_ops: int=0
    overall_bound: str="memory"

    def as_dict(self) -> dict:
        return asdict(self)


def _get_hw_spec(device: str) -> HardwareSpec:
    if device == "cpu" or not torch.cuda.is_available():
        s=HARDWARE_SPECS["CPU"]
        return HardwareSpec("CPU", s["peak_flops_tflops"], s["peak_bandwidth_gbps"])

    raw=torch.cuda.get_device_name(0).upper()
    for key, specs in HARDWARE_SPECS.items():
        if key == "CPU":
            continue
        if all(part in raw for part in key.upper().split()):
            return HardwareSpec(key, specs["peak_flops_tflops"], specs["peak_bandwidth_gbps"])

    # Unknown GPU — conservative A100-class estimate
    return HardwareSpec(raw, 100.0, 800.0)


def _dtype_bytes(dtype: torch.dtype) -> int:
    return {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int8: 1}.get(dtype, 4)


def _estimate_ops(
    cfg: ModelConfig,
    batch_size: int,
    seq_len: int,
    db: int,
    hw: HardwareSpec,
) -> List[OpRooflineRecord]:
    """
    Analytical FLOPs + bytes for one forward pass using standard transformer formulas.
    db = bytes per dtype element.
    """
    B, S, H=batch_size, seq_len, cfg.hidden_size
    E, K, I=cfg.num_experts, cfg.num_experts_per_tok, cfg.intermediate_size
    Nh, L=cfg.num_heads, cfg.num_layers
    head_dim=H // Nh

    def rec(name, layer_type, flops, nbytes):
        ai=flops / max(nbytes, 1)
        bound="compute" if ai >= hw.ridge_point else "memory"
        return OpRooflineRecord(name, layer_type, flops, nbytes, ai, bound)

    records=[]

    # Embedding: single table lookup at start of model (not per-layer)
    records.append(rec(
        "embed_lookup", "embed",
        B * S * H,
        (B * S * H + cfg.vocab_size * H) * db,
    ))

    # RMSNorm: 2 norms per layer (pre-attn + pre-ffn), ~2H FLOPs per token
    records.append(rec(
        "rmsnorm", "rmsnorm",
        B * S * 2 * H * 2 * L,
        B * S * H * db * 4 * L,
    ))

    # Attention: QKV proj + score computation + output proj
    qkv_f=3 * 2 * B * S * H * H
    qkv_b=(3 * H * H + B * S * 4 * H) * db
    score_f=2 * B * Nh * S * S * head_dim
    score_b=B * Nh * (2 * S * head_dim + S * S) * db
    out_f=2 * B * S * H * H
    out_b=(H * H + B * S * 2 * H) * db
    records.append(rec(
        "attention", "attention",
        (qkv_f + score_f + out_f) * L,
        (qkv_b + score_b + out_b) * L,
    ))

    # MoE routing: gate projection (H→E) + topk selection
    gate_f=2 * B * S * H * E
    gate_b=(H * E + B * S * (H + E)) * db
    topk_f=B * S * E          # O(E) comparisons per token
    topk_b=B * S * E * db * 2
    records.append(rec(
        "moe_routing", "moe_routing",
        (gate_f + topk_f) * L,
        (gate_b + topk_b) * L,
    ))

    # FFN: K active experts per token, each expert is a SwiGLU (3 matmuls: gate/up/down)
    ffn_f=B * S * K * 6 * H * I
    ffn_b=(K * 3 * H * I + B * S * K * (H + I * 2 + H)) * db
    records.append(rec("ffn_expert", "ffn", ffn_f * L, ffn_b * L))

    # LM head: single H→vocab projection at end of model
    records.append(rec(
        "lm_head", "lm_head",
        2 * B * S * H * cfg.vocab_size,
        (H * cfg.vocab_size + B * S * (H + cfg.vocab_size)) * db,
    ))

    return records


class RooflineAnalyzer:
    def __init__(self, device: str=DEVICE, dtype: torch.dtype=DTYPE):
        self.device=device
        self.dtype=dtype

    def run(self, cfg: ModelConfig, model_name: str) -> RooflineReport:
        hw=_get_hw_spec(self.device)
        db=_dtype_bytes(self.dtype)
        records=_estimate_ops(cfg, BENCH_CFG.batch_size, BENCH_CFG.seq_len, db, hw)
        mem=sum(1 for r in records if r.bound_type == "memory")
        comp=len(records) - mem
        return RooflineReport(
            model_name=model_name,
            hw=hw,
            records=records,
            ridge_point=hw.ridge_point,
            memory_bound_ops=mem,
            compute_bound_ops=comp,
            overall_bound="compute" if comp > mem else "memory",
        )

    @staticmethod
    def save(olmoe: Optional[RooflineReport], mixtral: Optional[RooflineReport]) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        payload={r.model_name: r.as_dict() for r in [olmoe, mixtral] if r}
        Path(ROOFLINE_DATA_PATH).write_text(json.dumps(payload, indent=2))
