from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class DenseRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float=1e-6):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.eps=eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DenseFFN(nn.Module):
    """SwiGLU FFN. intermediate_size = num_experts_per_tok × expert_intermediate_size
    so per-token FLOPs match the MoE model this shadows."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj=nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj=nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj=nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DenseAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=hidden_size // num_heads
        self.scale=math.sqrt(self.head_dim)
        self.q_proj=nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj=nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj=nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj=nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H=x.shape
        q=self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k=self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v=self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores=torch.matmul(q, k.transpose(-2, -1)) / self.scale
        out=torch.matmul(F.softmax(scores, dim=-1), v)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, H))


class DenseDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # intermediate = num_experts_per_tok × expert_intermediate so per-token FLOPs match MoE
        dense_intermediate=cfg.num_experts_per_tok * cfg.intermediate_size
        self.attn_norm=DenseRMSNorm(cfg.hidden_size)
        self.ffn_norm=DenseRMSNorm(cfg.hidden_size)
        self.attn=DenseAttention(cfg.hidden_size, cfg.num_heads)
        self.ffn=DenseFFN(cfg.hidden_size, dense_intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=x + self.attn(self.attn_norm(x))
        x=x + self.ffn(self.ffn_norm(x))
        return x


class DenseModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed=nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers=nn.ModuleList([DenseDecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm=DenseRMSNorm(cfg.hidden_size)
        self.lm_head=nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x=self.embed(input_ids)
        for layer in self.layers:
            x=layer(x)
        return self.lm_head(self.norm(x))


def build_dense(cfg: ModelConfig, device: str, dtype: torch.dtype) -> DenseModel:
    model=DenseModel(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model
