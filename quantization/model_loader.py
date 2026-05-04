"""
model_loader.py

Single quantization loader for MoE models such as Mixtral and OLMoE.
Supports FP16, INT8, and INT4 loading using Hugging Face Transformers + bitsandbytes.

Main function:
    load_model(model_name, precision)

Example:
    model, tokenizer = load_model("mistralai/Mixtral-8x7B-v0.1", "int4")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SUPPORTED_PRECISIONS = {"fp16", "int8", "int4", "gptq"}


def _read_local_quantization_config(model_name: str) -> Mapping[str, Any]:
    """Return quantization_config from a local HF config.json, if present."""
    config_path = Path(model_name) / "config.json"
    if not config_path.is_file():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    quantization_config = config.get("quantization_config")
    return quantization_config if isinstance(quantization_config, Mapping) else {}


def _is_gptq_checkpoint(model_name: str) -> bool:
    quantization_config = _read_local_quantization_config(model_name)
    quant_method = str(quantization_config.get("quant_method", "")).lower()
    return quant_method == "gptq" or "gptq" in model_name.lower()


def _validate_quantization_source(model_name: str, precision: str) -> None:
    if precision != "gptq" and _is_gptq_checkpoint(model_name):
        raise ValueError(
            "The selected precision uses bitsandbytes and requires the original dense checkpoint, "
            "but the model path appears to be a pre-quantized GPTQ checkpoint. Use --precisions gptq "
            "for this checkpoint, or use a dense model path for fp16/int8/int4."
        )


def load_model(
    model_name: str,
    precision: str = "fp16",
    device_map: Any = "auto",
    trust_remote_code: bool = True,
    offload_folder: str = "offload_cache",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load OLMoE/Mixtral-style causal language models in FP16, INT8, or INT4.

    Args:
        model_name:
            Hugging Face model id or local model path.
            Example: "mistralai/Mixtral-8x7B-v0.1"
        precision:
            One of: "fp16", "int8", "int4".
        device_map:
            Usually "auto". Lets Accelerate place the model on available GPU/CPU.
        trust_remote_code:
            Some models require custom modeling code from Hugging Face.
        offload_folder:
            Folder used by Accelerate when layers must be offloaded to disk.

    Returns:
        model, tokenizer
    """

    precision = precision.lower().strip()
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(SUPPORTED_PRECISIONS)}, got: {precision}")
    _validate_quantization_source(model_name, precision)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    # Some decoder-only models do not define pad_token by default.
    # For batching/padding, using eos_token as pad_token is common for inference.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if precision == "gptq":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    elif precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    elif precision in {"int8", "int4"}:
        effective_device_map = device_map
        if device_map == "auto" and torch.cuda.is_available():
            effective_device_map = {"": 0}

        use_cpu_disk_offload = effective_device_map == "auto"
        if use_cpu_disk_offload:
            (Path(offload_folder) / precision).mkdir(parents=True, exist_ok=True)

        if precision == "int8":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=use_cpu_disk_offload,
            )
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        kwargs = {
            "quantization_config": quant_config,
            "device_map": effective_device_map,
            "trust_remote_code": trust_remote_code,
        }
        if use_cpu_disk_offload:
            kwargs["offload_folder"] = str(Path(offload_folder) / precision)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model.eval()
    return model, tokenizer


def get_model_device(model) -> torch.device:
    """
    Return a safe input device for models loaded with or without device_map='auto'.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
