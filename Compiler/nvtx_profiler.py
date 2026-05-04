from __future__ import annotations

import os
from typing import Any, List

import torch
import torch.nn as nn

from config import (
    BENCH_CFG, DEVICE, DTYPE,
    NSYS_REPORT_PATH, NVTX_CHROME_TRACE_PATH, OUTPUT_DIR,
)

_CUDA=torch.cuda.is_available()


class NVTXHookManager:
    """
    Hooks are removed on __exit__ so the model is unmodified afterwards.
    On CPU, the hooks are installed but NVTX calls are skipped.
    """

    def __init__(self, model: nn.Module, model_name: str):
        self.model=model
        self.model_name=model_name
        self._handles: List[Any]=[]

    def register(self) -> None:
        for name, module in self.model.named_modules():
            if not name:
                continue
            label=f"{self.model_name}/{name}"
            self._handles.append(
                module.register_forward_pre_hook(
                    lambda m, inp, _l=label: torch.cuda.nvtx.range_push(_l) if _CUDA else None
                )
            )
            self._handles.append(
                module.register_forward_hook(
                    lambda m, inp, out: torch.cuda.nvtx.range_pop() if _CUDA else None
                )
            )

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *_):
        self.remove()


def profile_with_chrome_trace(
    model: nn.Module,
    model_name: str,
    device: str=DEVICE,
    dtype: torch.dtype=DTYPE,
    output_path: str=NVTX_CHROME_TRACE_PATH,
) -> str:
    """On CUDA the NVTX ranges from NVTXHookManager are also visible in Nsight Systems."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_ids=torch.randint(
        0, 1000,
        (BENCH_CFG.batch_size, BENCH_CFG.seq_len),
        device=device,
    )

    activities=[torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with NVTXHookManager(model, model_name):
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with torch.no_grad():
                for _ in range(3):
                    with torch.profiler.record_function(f"{model_name}_forward"):
                        _=model(input_ids)

    prof.export_chrome_trace(output_path)
    return output_path


def generate_nsys_command(
    script_path: str="main.py",
    report_path: str=NSYS_REPORT_PATH,
) -> str:
    return (
        f"nsys profile "
        f"--trace=cuda,nvtx,osrt "
        f"--output={report_path} "
        f"--force-overwrite=true "
        f"python {script_path}"
    )
