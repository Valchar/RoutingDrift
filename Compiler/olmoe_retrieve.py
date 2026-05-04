"""
models/olmoe_stub.py
--------------------
Lightweight OLMoE-style model with full MoE routing layer.

Faithfully reproduces the OLMoE routing logic (top-k with auxiliary load
balancing loss) so torch._dynamo.explain() hits the same graph breaks as the
real model — without requiring the 7B checkpoint to be downloaded.

References:
  - OLMoE paper: https://arxiv.org/abs/2409.02060
  - HuggingFace olmoe: modeling_olmoe.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig, OLMOE_CONFIG


class OLMoERMSNorm(nn.Module):
    """RMSNorm used by OLMoE (identical to LLaMA RMSNorm)."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon=eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Graph break risk: variance computation with .float() cast
        input_dtype=x.dtype
        x_fp32=x.to(torch.float32)
        variance=x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm=x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x_norm).to(input_dtype)


class OLMoEMLP(nn.Module):
    """Single expert FFN (SwiGLU activation)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj=nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj=nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj=nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OLMoESparseMoeBlock(nn.Module):
    """
    OLMoE sparse MoE block.

    Key routing logic that causes graph breaks:
    1. torch.topk          — data-dependent indexing
    2. F.softmax on router — fine (static), but follow-up gather is dynamic
    3. torch.zeros + scatter_add — dynamic shapes downstream
    4. Python-level for-loop over top-k experts — unrolled but breaks fuser
    5. Auxiliary load-balancing loss uses .mean() on dynamic expert counts
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts=cfg.num_experts
        self.num_experts_per_tok=cfg.num_experts_per_tok
        self.hidden_size=cfg.hidden_size
        self.gate=nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts=nn.ModuleList([OLMoEMLP(cfg) for _ in range(cfg.num_experts)])

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, hidden)
        Returns: (batch, seq_len, hidden), aux_loss scalar
        """
        orig_shape=x.shape
        x_flat=x.view(-1, self.hidden_size)  # (T, H) — T = B*S
        T=x_flat.shape[0]

        router_logits=self.gate(x_flat)  # (T, num_experts)

        # graph break 1: topk returns data-dependent indices
        routing_weights, selected_experts=torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )

        routing_weights=F.softmax(routing_weights, dim=-1)  # (T, top_k)

        # graph break 2: for-loop over experts with data-dependent mask
        final_hidden=torch.zeros_like(x_flat)  # (T, H)
        for expert_idx in range(self.num_experts):
            expert_mask=(selected_experts == expert_idx)  # (T, top_k) bool
            if expert_mask.any():
                token_indices, topk_positions=torch.where(expert_mask)
                token_weights=routing_weights[token_indices, topk_positions]  # (N,)
                token_weights=token_weights.unsqueeze(-1)  # (N, 1)
                expert_out=self.experts[expert_idx](x_flat[token_indices])  # (N, H)
                final_hidden.index_add_(0, token_indices, expert_out * token_weights)

        # graph break 3: .mean() on dynamic expert token counts (aux loss)
        router_probs=F.softmax(router_logits, dim=-1)
        tokens_per_expert=(selected_experts.unsqueeze(-1) == torch.arange(
            self.num_experts, device=x.device
        )).float().sum(0).sum(0)
        aux_loss=(router_probs.mean(0) * tokens_per_expert / T).sum()

        return final_hidden.view(orig_shape), aux_loss


class OLMoEAttention(nn.Module):
    """Grouped-query attention (OLMoE uses GQA)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads=cfg.num_heads
        self.head_dim=cfg.hidden_size // cfg.num_heads
        self.q_proj=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor=None):
        B, S, H=x.shape
        q=self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k=self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v=self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn=F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn=attn.transpose(1, 2).contiguous().view(B, S, H)
        return self.o_proj(attn)


class OLMoEDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn=OLMoEAttention(cfg)
        self.moe_block=OLMoESparseMoeBlock(cfg)
        self.input_layernorm=OLMoERMSNorm(cfg.hidden_size)
        self.post_attn_layernorm=OLMoERMSNorm(cfg.hidden_size)
        self.post_ff_layernorm=OLMoERMSNorm(cfg.hidden_size)

    def forward(self, x: torch.Tensor):
        residual=x
        x=self.input_layernorm(x)
        x=self.self_attn(x)
        x=residual + x

        residual=x
        x=self.post_attn_layernorm(x)
        x, aux_loss=self.moe_block(x)
        x=residual + x
        x=self.post_ff_layernorm(x)

        return x, aux_loss


class OLMoEModel(nn.Module):
    """
    Trimmed OLMoE model for compiler analysis.
    Includes embed + N decoder layers + lm_head.
    """
    def __init__(self, cfg: ModelConfig=OLMOE_CONFIG):
        super().__init__()
        self.cfg=cfg
        self.embed=nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers=nn.ModuleList(
            [OLMoEDecoderLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm=OLMoERMSNorm(cfg.hidden_size)
        self.lm_head=nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        x=self.embed(input_ids)  # (B, S, H)
        total_aux_loss=0.0
        for layer in self.layers:
            x, aux_loss=layer(x)
            total_aux_loss=total_aux_loss + aux_loss
        x=self.norm(x)
        logits=self.lm_head(x)
        return logits, total_aux_loss


def build_olmoe(
    cfg: ModelConfig=OLMOE_CONFIG,
    device: str="cpu",
    dtype: torch.dtype=torch.float32,
) -> OLMoEModel:
    model=OLMoEModel(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model
