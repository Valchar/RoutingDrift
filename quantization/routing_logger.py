"""
routing_logger.py

Utilities to hook into MoE router/gate layers and log top-k selected experts per token.

Main pieces:
    RoutingLogger
    find_router_modules(model)
    collect_routes(model, tokenizer, prompts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from model_loader import get_model_device


@dataclass
class RoutingRecord:
    """Stores router selections for one forward hook call."""

    module_name: str
    topk_indices: torch.Tensor
    shape: Tuple[int, ...]


@dataclass
class RoutingLogger:
    """
    Forward-hook logger for MoE router/gate layers.

    It captures router logits, applies torch.topk, and stores expert indices.
    """

    top_k: int = 2
    target_module_names: Optional[Sequence[str]] = None
    records: List[RoutingRecord] = field(default_factory=list)
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def _is_target_router(self, module_name: str) -> bool:
        lower = module_name.lower()

        if self.target_module_names:
            return any(target.lower() in lower for target in self.target_module_names)

        # Default broad search. Inspect printed module names and narrow if needed.
        router_keywords = [
            "block_sparse_moe.gate",  # Mixtral common pattern
            "router",
            "gate",
        ]
        return any(keyword in lower for keyword in router_keywords)

    def _extract_router_logits(self, output) -> torch.Tensor | None:
        """
        Router modules may return tensor directly or tuple/list.
        This function tries to grab the tensor containing expert scores.
        """
        if torch.is_tensor(output):
            return output

        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item

        # Some outputs may be objects with logits/router_logits attributes.
        for attr in ["router_logits", "logits"]:
            if hasattr(output, attr):
                value = getattr(output, attr)
                if torch.is_tensor(value):
                    return value

        return None

    def _make_hook(self, module_name: str):
        def hook_fn(module, inputs, output):
            router_logits = self._extract_router_logits(output)
            if router_logits is None:
                return

            # Router logits should normally end with num_experts dimension.
            # top_k selected expert ids are taken along the final dimension.
            if router_logits.shape[-1] < self.top_k:
                return

            with torch.no_grad():
                _, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
                topk_indices = topk_indices.detach().cpu()

            self.records.append(
                RoutingRecord(
                    module_name=module_name,
                    topk_indices=topk_indices,
                    shape=tuple(topk_indices.shape),
                )
            )

        return hook_fn

    def attach(self, model, verbose: bool = True) -> None:
        """Attach hooks to matching router/gate modules."""
        self.remove()
        self.clear()

        for name, module in model.named_modules():
            if self._is_target_router(name):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
                if verbose:
                    print(f"[RoutingLogger] Attached hook to: {name}")

        if verbose and not self.handles:
            print("[RoutingLogger] WARNING: No router/gate modules found. Run find_router_modules(model).")

    def clear(self) -> None:
        self.records.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def get_routes_by_module(self) -> Dict[str, List[torch.Tensor]]:
        routes: Dict[str, List[torch.Tensor]] = {}
        for record in self.records:
            routes.setdefault(record.module_name, []).append(record.topk_indices)
        return routes


@torch.no_grad()
def collect_routes(
    model,
    tokenizer,
    prompts: Sequence[str],
    top_k: int = 2,
    target_module_names: Optional[Sequence[str]] = None,
    max_length: int = 256,
    verbose: bool = True,
) -> Dict[str, List[torch.Tensor]]:
    """
    Run fixed prompts through the model and collect top-k expert selections.

    Args:
        model, tokenizer:
            Loaded from load_model.
        prompts:
            Fixed prompt set. Use the same prompts for FP16/INT8/INT4.
        top_k:
            Number of experts selected per token. Mixtral commonly uses top_k=2.
        target_module_names:
            Optional list of router module name substrings to hook.
            Example for Mixtral: ["block_sparse_moe.gate"]
        max_length:
            Tokenizer truncation length.
        verbose:
            Print hook information.

    Returns:
        Dictionary: module_name -> list of top-k tensors from each forward pass.
    """

    logger = RoutingLogger(top_k=top_k, target_module_names=target_module_names)
    logger.attach(model, verbose=verbose)
    logger.clear()

    device = get_model_device(model)

    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"[collect_routes] Prompt {i + 1}/{len(prompts)}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        _ = model(**inputs)

    routes = logger.get_routes_by_module()
    logger.remove()
    return routes


def find_router_modules(model) -> List[Tuple[str, str]]:
    """
    Print and return candidate router/gate modules.
    Run this once if hooks do not attach correctly.
    """
    candidates = []
    keywords = ["router", "gate", "moe", "expert"]

    for name, module in model.named_modules():
        lower = name.lower()
        if any(keyword in lower for keyword in keywords):
            candidates.append((name, module.__class__.__name__))

    print("\nCandidate router/MoE modules:")
    for name, class_name in candidates:
        print(f"  {name:80s} {class_name}")

    return candidates
