"""
harness_eval.py

lm-evaluation-harness integration for MMLU/GSM8K/HellaSwag.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any


SUPPORTED_EVAL_TASKS = ("mmlu", "gsm8k", "hellaswag")
PRIMARY_METRIC_CANDIDATES = (
    "acc_norm",
    "acc",
    "exact_match",
    "em",
)


def build_hf_model_args(
    model_name: str,
    precision: str,
    device: str = "cuda",
    trust_remote_code: bool = True,
) -> str:
    """
    Build lm-eval HF model_args string for a precision setting.
    """
    parts = [
        f"pretrained={model_name}",
        f"device={device}",
        "clean_up_tokenization_spaces=False",
    ]
    if trust_remote_code:
        parts.append("trust_remote_code=True")

    precision = precision.lower().strip()
    if precision == "gptq":
        pass
    elif precision == "fp16":
        parts.append("dtype=float16")
    elif precision == "int8":
        parts.append("load_in_8bit=True")
    elif precision == "int4":
        parts.extend(
            [
                "load_in_4bit=True",
                "bnb_4bit_compute_dtype=float16",
                "bnb_4bit_quant_type=nf4",
                "bnb_4bit_use_double_quant=True",
            ]
        )
    else:
        raise ValueError(f"Unsupported precision for lm-eval: {precision}")

    return ",".join(parts)


def _patch_lm_eval_git_hash(lm_eval_module) -> None:
    """
    Avoid noisy git stderr in environments where the run directory isn't a git repo.
    """
    try:
        evaluator = lm_eval_module.evaluator
    except Exception:
        return

    if hasattr(evaluator, "get_git_commit_hash"):
        evaluator.get_git_commit_hash = lambda: "unknown"


def run_lm_eval(
    model_name: str,
    precision: str,
    tasks: list[str],
    output_path: str | Path,
    num_fewshot: int = 5,
    batch_size: str = "auto",
    limit: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Run lm-evaluation-harness via Python API and persist full JSON output.
    """
    try:
        import lm_eval
    except ImportError as exc:
        raise RuntimeError(
            "lm-evaluation-harness is not installed. Install with: pip install lm-eval"
        ) from exc

    _patch_lm_eval_git_hash(lm_eval)

    precision = precision.lower().strip()

    if precision in {"int8", "int4"}:
        # For some remote-code models (including OLMoE variants), passing
        # load_in_8bit/load_in_4bit via lm-eval model_args can leak through
        # to model __init__. Load quantized model ourselves and pass LM object.
        try:
            from lm_eval.models.huggingface import HFLM
        except ImportError as exc:
            raise RuntimeError(
                "Your lm-eval version does not expose lm_eval.models.huggingface.HFLM. "
                "Upgrade lm-eval to a recent version to evaluate int8/int4 variants safely."
            ) from exc
        from model_loader import load_model

        model, tokenizer = load_model(
            model_name=model_name,
            precision=precision,
            device_map="auto",
            trust_remote_code=True,
        )
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            batch_size=batch_size,
            device=device,
            clean_up_tokenization_spaces=False,
        )
        try:
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=limit,
            )
        finally:
            del lm
            del model
            del tokenizer
            gc.collect()
    else:
        model_args = build_hf_model_args(
            model_name=model_name,
            precision=precision,
            device=device,
        )
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
        )

    sanitized = _sanitize_for_json(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)
    return results


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively remove non-JSON-serializable objects from nested dicts/lists."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def _extract_primary_metric(task_metrics: dict[str, Any]) -> tuple[str, float] | None:
    for metric_name in PRIMARY_METRIC_CANDIDATES:
        value = task_metrics.get(metric_name)
        if isinstance(value, (int, float)):
            return metric_name, float(value)
    return None


def extract_task_accuracies(
    results: dict[str, Any],
    tasks: list[str],
) -> dict[str, dict[str, float | str]]:
    """
    Parse lm-eval result JSON into task -> {"accuracy": float, "metric": str}.
    """
    parsed: dict[str, dict[str, float | str]] = {}
    task_results = results.get("results", {})
    if not isinstance(task_results, dict):
        return parsed

    for task in tasks:
        if task in task_results and isinstance(task_results[task], dict):
            metric = _extract_primary_metric(task_results[task])
            if metric is not None:
                parsed[task] = {"accuracy": metric[1], "metric": metric[0]}
                continue

        # Group fallback: average matching subtasks, common for some harness task groups.
        subtask_scores: list[float] = []
        subtask_metric_name = ""
        prefix = f"{task}_"
        for subtask_name, metrics in task_results.items():
            if not isinstance(subtask_name, str) or not subtask_name.startswith(prefix):
                continue
            if not isinstance(metrics, dict):
                continue
            metric = _extract_primary_metric(metrics)
            if metric is None:
                continue
            subtask_metric_name = metric[0]
            subtask_scores.append(metric[1])

        if subtask_scores:
            parsed[task] = {
                "accuracy": sum(subtask_scores) / len(subtask_scores),
                "metric": subtask_metric_name or "avg_subtasks",
            }

    return parsed
