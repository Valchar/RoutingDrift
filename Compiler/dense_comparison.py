from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import (
    BENCH_CFG, DENSE_COMPARISON_PATH, DEVICE, DTYPE, OUTPUT_DIR, ModelConfig
)


@dataclass
class DenseComparisonResult:
    """Timing comparison between an MoE model and its dense (no-routing) equivalent."""
    moe_name: str
    dense_name: str
    moe_p50_ms: float
    dense_p50_ms: float
    moe_throughput_tps: float
    dense_throughput_tps: float
    # dense_p50 / moe_p50: >1 means MoE is faster, <1 means dense is faster
    speedup_moe_vs_dense: float
    routing_overhead_ms: float  # moe_p50 - dense_p50; positive = routing costs time
    moe_params: int
    dense_params: int

    def as_dict(self) -> dict:
        return asdict(self)


class DenseComparisonAnalyzer:
    def __init__(self, device: str=DEVICE, dtype: torch.dtype=DTYPE):
        self.device=device
        self.dtype=dtype

    def run(
        self,
        moe_model: nn.Module,
        dense_model: nn.Module,
        moe_name: str,
        cfg: ModelConfig,
    ) -> DenseComparisonResult:
        moe_p50, moe_tps=self._bench(moe_model, cfg)
        dense_p50, dense_tps=self._bench(dense_model, cfg)
        return DenseComparisonResult(
            moe_name=moe_name,
            dense_name=f"Dense-{moe_name}",
            moe_p50_ms=moe_p50,
            dense_p50_ms=dense_p50,
            moe_throughput_tps=moe_tps,
            dense_throughput_tps=dense_tps,
            speedup_moe_vs_dense=dense_p50 / max(moe_p50, 1e-9),
            routing_overhead_ms=moe_p50 - dense_p50,
            moe_params=sum(p.numel() for p in moe_model.parameters()),
            dense_params=sum(p.numel() for p in dense_model.parameters()),
        )

    def _bench(self, model: nn.Module, cfg: ModelConfig) -> Tuple[float, float]:
        input_ids=torch.randint(
            0, cfg.vocab_size,
            (BENCH_CFG.batch_size, BENCH_CFG.seq_len),
            device=self.device,
        )
        seq_tokens=BENCH_CFG.batch_size * BENCH_CFG.seq_len
        latencies: List[float]=[]
        with torch.no_grad():
            for _ in range(BENCH_CFG.warmup_iters):
                _=model(input_ids)
            if self.device == "cuda":
                torch.cuda.synchronize()
            for _ in range(BENCH_CFG.timed_iters):
                t0=time.perf_counter()
                _=model(input_ids)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
        p50=float(np.percentile(latencies, 50))
        return p50, seq_tokens / (p50 / 1000.0)

    @staticmethod
    def save(results: List[DenseComparisonResult]) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        Path(DENSE_COMPARISON_PATH).write_text(
            json.dumps({r.moe_name: r.as_dict() for r in results}, indent=2)
        )
