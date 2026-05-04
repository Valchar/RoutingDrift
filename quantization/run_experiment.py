"""
run_experiment.py

End-to-end experiment:
    1. Load model in FP16, INT8, INT4, or GPTQ (+ optional compiler modes)
    2. Hook router/gate layer
    3. Log top-k expert indices per token
    4. Compute routing drift vs the selected baseline
    5. (Optional) Run lm-evaluation-harness on MMLU/GSM8K/HellaSwag
    6. Compute Pearson/Spearman correlation (routing drift vs accuracy drop)
    7. Save JSON/CSV/Markdown summaries + layer drift heatmaps

Example:
    python run_experiment.py --model_name mistralai/Mixtral-8x7B-v0.1 --top_k 2

For OLMoE, use the Hugging Face model id used by your team.
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch

from analysis_utils import (
    build_drift_accuracy_rows,
    plot_layer_heatmap,
    save_rows_csv,
    summarize_correlations,
)
from drift import build_layerwise_rows, summarize_research_metrics
from harness_eval import SUPPORTED_EVAL_TASKS, extract_task_accuracies, run_lm_eval
from io_utils import save_prompts_txt, save_routes_json, save_summary_csv, save_summary_md
from model_loader import load_model
from routing_logger import collect_routes, find_router_modules


DEFAULT_PROMPTS = [
    "Explain quantization in machine learning using simple terms.",
    "What is the difference between a compiler and an interpreter?",
    "Write a short Python function to reverse a list.",
    "Explain mixture of experts models in two sentences.",
    "Summarize the benefits and risks of using AI in hiring.",
]

SUPPORTED_COMPILER_MODES = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}


def free_memory():
    """Free CPU/GPU memory between precision runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _sanitize_name_for_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _build_variant_name(precision: str, compiler_mode: str) -> str:
    if compiler_mode == "eager":
        return precision
    return f"{precision}+compile:{compiler_mode}"


def _normalize_task_names(task_args: List[str]) -> List[str]:
    tasks: List[str] = []
    for task_arg in task_args:
        for task in task_arg.split(","):
            task = task.strip()
            if task:
                tasks.append(task)
    return tasks


def _apply_compiler_mode(model, compiler_mode: str):
    if compiler_mode == "eager":
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError("Requested compiler modes but torch.compile is not available in this PyTorch version.")

    print(f"[Compiler] Applying torch.compile(mode='{compiler_mode}')")
    return torch.compile(model, mode=compiler_mode)


def run_for_precision(
    model_name: str,
    precision: str,
    compiler_mode: str,
    prompts: List[str],
    top_k: int,
    target_module_names: Optional[List[str]],
    max_length: int,
    output_dir: Path,
    inspect_modules: bool = False,
):
    variant_name = _build_variant_name(precision, compiler_mode)
    print(f"\n========== Loading variant: {variant_name} ==========")
    model, tokenizer = load_model(model_name=model_name, precision=precision)
    model = _apply_compiler_mode(model, compiler_mode)

    if inspect_modules:
        find_router_modules(model)

    print(f"\n========== Collecting routes for {variant_name} ==========")
    routes = collect_routes(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        top_k=top_k,
        target_module_names=target_module_names,
        max_length=max_length,
        verbose=True,
    )

    output_path = output_dir / f"routes_{_sanitize_name_for_filename(variant_name)}.json"
    save_routes_json(routes, output_path)
    print(f"[Saved] {output_path}")

    del model
    del tokenizer
    free_memory()

    return routes


def _run_lm_eval_matrix(
    model_name: str,
    variants_for_eval: List[str],
    variant_to_precision: Dict[str, str],
    tasks: List[str],
    output_dir: Path,
    num_fewshot: int,
    batch_size: str,
    limit: Optional[int],
    device: str,
) -> List[dict]:
    eval_rows: List[dict] = []
    eval_output_dir = output_dir / "lm_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants_for_eval:
        precision = variant_to_precision[variant]
        output_path = eval_output_dir / f"lm_eval_{_sanitize_name_for_filename(variant)}.json"
        print(f"\n========== Running lm-eval for {variant} ({','.join(tasks)}) ==========")
        try:
            result = run_lm_eval(
                model_name=model_name,
                precision=precision,
                tasks=tasks,
                output_path=output_path,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=limit,
                device=device,
            )
        except Exception as exc:
            print(f"[lm-eval WARNING] Skipping variant '{variant}' due to error: {exc}")
            continue
        parsed = extract_task_accuracies(result, tasks)
        for task in tasks:
            task_info = parsed.get(task)
            if not task_info:
                continue
            eval_rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "accuracy": float(task_info["accuracy"]),
                    "metric": str(task_info["metric"]),
                }
            )

    return eval_rows


def main():
    parser = argparse.ArgumentParser(description="Quantization routing drift experiment for MoE models.")
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        type=str,
        required=True,
        help="Hugging Face model id or local path.",
    )
    parser.add_argument("--top_k", "--top-k", dest="top_k", type=int, default=2, help="Number of selected experts to log.")
    parser.add_argument(
        "--max_length",
        "--max-length",
        dest="max_length",
        type=int,
        default=256,
        help="Prompt truncation length.",
    )
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="results", help="Directory for outputs.")
    parser.add_argument(
        "--target_module",
        "--target-module",
        action="append",
        default=None,
        help=(
            "Router module name substring to hook. Can be used multiple times. "
            "For Mixtral try: --target_module block_sparse_moe.gate"
        ),
    )
    parser.add_argument(
        "--inspect_modules",
        "--inspect-routers",
        action="store_true",
        help="Print candidate router/MoE modules before collecting routes.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp16", "int8", "int4"],
        choices=["fp16", "int8", "int4", "gptq"],
        help="Precisions to run. Use gptq for an already GPTQ-quantized checkpoint.",
    )
    parser.add_argument(
        "--compiler_modes",
        nargs="+",
        default=[],
        choices=sorted(SUPPORTED_COMPILER_MODES),
        help="Optional torch.compile modes to evaluate as additional routing drift variants.",
    )
    parser.add_argument(
        "--compiler_precision",
        type=str,
        default="fp16",
        choices=["fp16", "int8", "int4", "gptq"],
        help="Precision used for compiler-mode drift variants.",
    )
    parser.add_argument(
        "--run_lm_eval",
        "--run-lm-eval",
        action="store_true",
        help="Run lm-evaluation-harness tasks (MMLU/GSM8K/HellaSwag by default).",
    )
    parser.add_argument(
        "--lm_eval_tasks",
        "--lm-eval-tasks",
        nargs="+",
        default=list(SUPPORTED_EVAL_TASKS),
        help="lm-eval tasks to run. Accepts either spaces or commas, e.g. mmlu gsm8k or mmlu,gsm8k.",
    )
    parser.add_argument(
        "--lm_eval_num_fewshot",
        "--lm-eval-num-fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples per lm-eval task.",
    )
    parser.add_argument(
        "--lm_eval_batch_size",
        "--lm-eval-batch-size",
        type=str,
        default="auto",
        help="lm-eval batch size.",
    )
    parser.add_argument(
        "--lm_eval_limit",
        "--lm-eval-limit",
        type=int,
        default=None,
        help="Optional lm-eval sample limit for quick smoke tests.",
    )
    parser.add_argument(
        "--lm_eval_device",
        "--lm-eval-device",
        type=str,
        default="cuda",
        help="lm-eval backend device (e.g. cuda/cpu).",
    )
    parser.add_argument(
        "--skip_heatmaps",
        "--skip-heatmaps",
        action="store_true",
        help="Skip layer drift heatmap generation.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_prompts_txt(DEFAULT_PROMPTS, output_dir / "prompts_used.txt")
    print(f"[Saved] {output_dir / 'prompts_used.txt'}")

    all_routes: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    variant_to_precision: Dict[str, str] = {}
    for precision in args.precisions:
        variant_name = _build_variant_name(precision, "eager")
        routes = run_for_precision(
            model_name=args.model_name,
            precision=precision,
            compiler_mode="eager",
            prompts=DEFAULT_PROMPTS,
            top_k=args.top_k,
            target_module_names=args.target_module,
            max_length=args.max_length,
            output_dir=output_dir,
            inspect_modules=args.inspect_modules,
        )
        all_routes[variant_name] = routes
        variant_to_precision[variant_name] = precision

    for compiler_mode in args.compiler_modes:
        variant_name = _build_variant_name(args.compiler_precision, compiler_mode)
        routes = run_for_precision(
            model_name=args.model_name,
            precision=args.compiler_precision,
            compiler_mode=compiler_mode,
            prompts=DEFAULT_PROMPTS,
            top_k=args.top_k,
            target_module_names=args.target_module,
            max_length=args.max_length,
            output_dir=output_dir,
            inspect_modules=False,
        )
        all_routes[variant_name] = routes
        variant_to_precision[variant_name] = args.compiler_precision

    baseline_variant = _build_variant_name(args.precisions[0], "eager")
    if baseline_variant not in all_routes:
        print(f"[Warning] Baseline variant {baseline_variant!r} was not run, so drift cannot be computed.")
        return

    baseline_routes = all_routes[baseline_variant]
    summary_rows = [
        {
            "variant": baseline_variant,
            "precision": variant_to_precision[baseline_variant],
            "compiler_mode": "eager",
            "variant_type": "baseline",
            "routing_similarity_rs": 1.0,
            "jaccard_drift": 0.0,
            "overlap_at_k": 1.0,
            "selection_shift": 0.0,
        }
    ]

    layer_rows: List[dict] = []
    for variant, routes in all_routes.items():
        if variant == baseline_variant:
            continue

        metrics = summarize_research_metrics(
            baseline_routes=baseline_routes,
            quantized_routes=routes,
        )

        compiler_mode = variant.split("+compile:", 1)[1] if "+compile:" in variant else "eager"

        row = {
            "variant": variant,
            "precision": variant_to_precision[variant],
            "compiler_mode": compiler_mode,
            "variant_type": "compiler" if compiler_mode != "eager" else "quantization",
            "routing_similarity_rs": round(metrics["routing_similarity_rs"], 6),
            "jaccard_drift": round(metrics["jaccard_drift"], 6),
            "overlap_at_k": round(metrics["overlap_at_k"], 6),
            "selection_shift": round(metrics["selection_shift"], 6),
        }
        summary_rows.append(row)

        print(f"\nResearch Metrics {variant} vs {baseline_variant}")
        print(f"  Routing similarity RS : {metrics['routing_similarity_rs']:.4f}")
        print(f"  Jaccard routing drift : {metrics['jaccard_drift']:.4f}")
        print(f"  Overlap@k             : {metrics['overlap_at_k']:.4f}")
        print(f"  Selection shift       : {metrics['selection_shift']:.4f}")

        layer_rows.extend(
            build_layerwise_rows(
                baseline_routes=baseline_routes,
                quantized_routes=routes,
                variant=variant,
            )
        )

    summary_path = output_dir / "routing_drift_summary.csv"
    save_summary_csv(summary_rows, summary_path)
    print(f"\n[Saved] {summary_path}")

    layer_summary_path = output_dir / "routing_drift_layers.csv"
    save_rows_csv(layer_rows, layer_summary_path)
    print(f"[Saved] {layer_summary_path}")

    if not args.skip_heatmaps:
        heatmap_jaccard_path = output_dir / "routing_drift_heatmap_jaccard.png"
        plot_layer_heatmap(
            layer_rows=layer_rows,
            metric_key="jaccard_drift",
            output_path=heatmap_jaccard_path,
            title="Routing Drift Heatmap by Layer (Jaccard Drift)",
        )
        print(f"[Saved] {heatmap_jaccard_path}")

        heatmap_shift_path = output_dir / "routing_drift_heatmap_selection_shift.png"
        plot_layer_heatmap(
            layer_rows=layer_rows,
            metric_key="selection_shift",
            output_path=heatmap_shift_path,
            title="Routing Drift Heatmap by Layer (Selection Shift)",
        )
        print(f"[Saved] {heatmap_shift_path}")

    summary_md_path = output_dir / "summary.md"
    save_summary_md(
        model_name=args.model_name,
        prompts_count=len(DEFAULT_PROMPTS),
        top_k=args.top_k,
        rows=summary_rows,
        output_path=summary_md_path,
    )
    print(f"[Saved] {summary_md_path}")

    if args.run_lm_eval:
        tasks = _normalize_task_names(args.lm_eval_tasks)
        if not tasks:
            raise ValueError("No lm-eval tasks provided.")

        eval_variants = [
            row["variant"]
            for row in summary_rows
            if row["compiler_mode"] == "eager"
        ]
        eval_rows = _run_lm_eval_matrix(
            model_name=args.model_name,
            variants_for_eval=eval_variants,
            variant_to_precision=variant_to_precision,
            tasks=tasks,
            output_dir=output_dir,
            num_fewshot=args.lm_eval_num_fewshot,
            batch_size=args.lm_eval_batch_size,
            limit=args.lm_eval_limit,
            device=args.lm_eval_device,
        )
        eval_csv_path = output_dir / "lm_eval_scores.csv"
        save_rows_csv(eval_rows, eval_csv_path)
        print(f"[Saved] {eval_csv_path}")

        drift_accuracy_rows = build_drift_accuracy_rows(
            drift_rows=summary_rows,
            eval_rows=eval_rows,
            baseline_variant=baseline_variant,
            drift_metric_key="jaccard_drift",
        )
        drift_accuracy_path = output_dir / "drift_vs_accuracy_drop.csv"
        save_rows_csv(drift_accuracy_rows, drift_accuracy_path)
        print(f"[Saved] {drift_accuracy_path}")

        correlation_rows = summarize_correlations(drift_accuracy_rows)
        correlation_path = output_dir / "drift_accuracy_correlations.csv"
        save_rows_csv(correlation_rows, correlation_path)
        print(f"[Saved] {correlation_path}")


if __name__ == "__main__":
    main()
