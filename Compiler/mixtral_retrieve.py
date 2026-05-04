import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig, MIXTRAL_CONFIG


class MixtralRMSNorm(nn.Module):
    """Mixtral RMSNorm — identical signature to LLaMA."""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.eps=eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same dtype upcast as OLMoE — should see same graph-break pattern
        x_fp32=x.float()
        rms=torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(x.dtype) * self.weight


class MixtralBLockSparseTop2MLP(nn.Module):
    """Single Mixtral expert (SiLU gating, no SwiGLU)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1=nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2=nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.w3=nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MixtralSparseMoeBlock(nn.Module):
    """
    Mixtral sparse MoE block — top-2 routing, no aux loss.

    Graph break catalogue (different from OLMoE):
    1. torch.topk — dynamic indices, data-dependent scatter
    2. one_hot encoding on selected experts — dynamic shapes
    3. einsum over dynamically selected expert weights — unsupported reshape
    4. Expert capacity enforcement (if used) — data-dependent branches
    5. .nonzero() for token→expert dispatch — fully data-dependent
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts=cfg.num_experts
        self.num_experts_per_tok=cfg.num_experts_per_tok  # 2 for Mixtral
        self.hidden_dim=cfg.hidden_size
        self.gate=nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts=nn.ModuleList(
            [MixtralBLockSparseTop2MLP(cfg) for _ in range(cfg.num_experts)]
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, hidden)
        Returns: (batch, seq_len, hidden)
        """
        batch, seq_len, hidden=x.shape
        x_flat=x.view(-1, hidden)
        T=x_flat.shape[0]

        router_logits=self.gate(x_flat)

        # Graph break 1: topk returns data-dependent indices
        routing_weights, selected_experts=torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )  # (T, 2)

        routing_weights=F.softmax(routing_weights, dim=-1, dtype=torch.float32)
        routing_weights=routing_weights.to(x.dtype)

        # graph break 2: one_hot output shape depends on num_classes at runtime
        expert_mask=torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, T)

        final_hidden=torch.zeros(
            (T, hidden), dtype=x.dtype, device=x.device
        )

        # graph break 3: for-loop over experts with torch.where dispatch
        for expert_idx in range(self.num_experts):
            expert=self.experts[expert_idx]
            idx=torch.where(expert_mask[expert_idx])
            if len(idx[0]) == 0:
                continue
            top_k_slot, token_idx=idx
            weights=routing_weights[token_idx, top_k_slot]
            current_state=x_flat[token_idx]  # (N, H)
            current_hidden_states=expert(current_state) * weights.unsqueeze(-1)
            # Graph break 4: index_add_ with variable-length token_idx
            final_hidden.index_add_(0, token_idx, current_hidden_states)

        return final_hidden.view(batch, seq_len, hidden)


class MixtralAttention(nn.Module):
    """Sliding-window attention (simplified — fixed window for analysis)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads=cfg.num_heads
        self.head_dim=cfg.hidden_size // cfg.num_heads
        self.q_proj=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H=x.shape
        q=self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k=self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v=self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out=F.scaled_dot_product_attention(q, k, v)
        out=out.transpose(1, 2).contiguous().view(B, S, H)
        return self.o_proj(out)


class MixtralDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn=MixtralAttention(cfg)
        self.block_sparse_moe=MixtralSparseMoeBlock(cfg)
        self.input_layernorm=MixtralRMSNorm(cfg.hidden_size)
        self.post_attention_layernorm=MixtralRMSNorm(cfg.hidden_size)

    def forward(self, x: torch.Tensor):
        residual=x
        x=self.input_layernorm(x)
        x=self.self_attn(x)
        x=residual + x

        residual=x
        x=self.post_attention_layernorm(x)
        x=self.block_sparse_moe(x)
        x=residual + x

        return x


class MixtralModel(nn.Module):
    """Trimmed Mixtral model for compiler analysis."""
    def __init__(self, cfg: ModelConfig=MIXTRAL_CONFIG):
        super().__init__()
        self.cfg=cfg
        self.embed=nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers=nn.ModuleList(
            [MixtralDecoderLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm=MixtralRMSNorm(cfg.hidden_size)
        self.lm_head=nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        x=self.embed(input_ids)
        for layer in self.layers:
            x=layer(x)
        x=self.norm(x)
        return self.lm_head(x)


def build_mixtral(
    cfg: ModelConfig=MIXTRAL_CONFIG,
    device: str="cpu",
    dtype: torch.dtype=torch.float32,
) -> MixtralModel:
    model=MixtralModel(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model
