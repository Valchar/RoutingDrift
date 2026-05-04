from __future__ import annotations

import gc
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.benchmark as benchmark
from rich.console import Console
from rich.table import Table

from config import (
    BENCH_CFG, BenchmarkConfig,
    COMPILE_MODES,
    DEVICE, DTYPE,
)

console=Console()


@dataclass
class CompileModeResult:
    model_name:       str
    mode:             str            
    compile_time_s:   float          
    latencies_ms:     List[float]
    p50_ms:           float
    p90_ms:           float
    p99_ms:           float
    throughput_tps:   float          
    speedup:          float          
    peak_vram_mb:     float          
    compile_vram_overhead_mb: float  


@dataclass
class CompileBenchmarkReport:
    model_name:   str
    results:      Dict[str, CompileModeResult]   
    best_mode:    str                           
    eager_result: CompileModeResult

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "best_mode":  self.best_mode,
            "results": {
                mode: {
                    "compile_time_s":  r.compile_time_s,
                    "p50_ms":          round(r.p50_ms, 3),
                    "p90_ms":          round(r.p90_ms, 3),
                    "p99_ms":          round(r.p99_ms, 3),
                    "throughput_tps":  round(r.throughput_tps, 1),
                    "speedup":         round(r.speedup, 3),
                    "peak_vram_mb":    round(r.peak_vram_mb, 1),
                    "compile_vram_overhead_mb": round(r.compile_vram_overhead_mb, 1),
                }
                for mode, r in self.results.items()
            },
        }



class CompileBenchmarker:
    """
    Benchmarks a model under eager + all 3 compile modes.

    Usage
    -----
    benchmarker = CompileBenchmarker()
    report = benchmarker.run(model, model_name="OLMoE")
    print(report.best_mode)   # feed into config matrix
    """

    def __init__(
        self,
        cfg: BenchmarkConfig = BENCH_CFG,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
    ):
        self.cfg    = cfg
        self.device = device
        self.dtype  = dtype



    def run(
        self,
        model: torch.nn.Module,
        model_name: str,
        verbose: bool = True,
    ) -> CompileBenchmarkReport:
        console.rule(f"[bold cyan]Compile Mode Sweep — {model_name}")

        input_ids = torch.randint(
            0, 1000,
            (self.cfg.batch_size, self.cfg.seq_len),
            device=self.device,
        )
        seq_tokens = self.cfg.batch_size * self.cfg.seq_len

        console.print("[yellow]▶ Running eager baseline…[/yellow]")
        eager_result = self._benchmark_mode(
            model, input_ids, "eager", seq_tokens, model_name
        )

        results: Dict[str, CompileModeResult] = {"eager": eager_result}

        # Inductor's CPU backend on Windows shells out to MSVC (cl.exe). If it
        # isn't on PATH there's no point attempting the compile modes — every
        # one will fail with the same RuntimeError. Skip them up-front with a
        # single readable message and record N/A results so the graphs still
        # render.
        skip_compile = (
            sys.platform == "win32"
            and self.device == "cpu"
            and shutil.which("cl") is None
        )
        if skip_compile:
            console.print(
                "[yellow]▶ Skipping torch.compile modes: MSVC cl.exe not on "
                "PATH (Windows CPU build needs it for Inductor codegen).[/yellow]"
            )
            for mode in COMPILE_MODES:
                results[mode] = self._failed_result(model_name, mode, eager_result)

        for mode in [] if skip_compile else COMPILE_MODES:
            console.print(f"[yellow]▶ Compiling with mode='{mode}'…[/yellow]")

            torch._dynamo.reset()

           
            t0 = time.perf_counter()
            try:
                compiled_model = torch.compile(model, mode=mode)
            
                with torch.no_grad():
                    _ = compiled_model(input_ids)
            except Exception as exc:
                console.print(f"[red]  torch.compile(mode={mode}) failed: {exc}[/red]")
                results[mode] = self._failed_result(model_name, mode, eager_result)
                continue
            compile_time = time.perf_counter() - t0

            result = self._benchmark_mode(
                compiled_model, input_ids, mode, seq_tokens, model_name,
                compile_time_s=compile_time,
                eager_p50=eager_result.p50_ms,
                eager_vram=eager_result.peak_vram_mb,
            )
            results[mode] = result
            console.print(
                f"  [green]✓[/green] mode={mode}  "
                f"compile={compile_time:.1f}s  "
                f"p50={result.p50_ms:.2f}ms  "
                f"speedup={result.speedup:.2f}x"
            )

        best_mode = self._pick_best_mode(results)
        report    = CompileBenchmarkReport(
            model_name   = model_name,
            results      = results,
            best_mode    = best_mode,
            eager_result = eager_result,
        )

        if verbose:
            self._print_report(report)

        return report


    def _benchmark_mode(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        mode: str,
        seq_tokens: int,
        model_name: str,
        compile_time_s: float = 0.0,
        eager_p50: float = None,
        eager_vram: float = 0.0,
    ) -> CompileModeResult:
        """Time `model(input_ids)` for warmup + timed iterations."""

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            for _ in range(self.cfg.warmup_iters):
                _ = model(input_ids)

        if self.device == "cuda":
            torch.cuda.synchronize()

 
        latencies_ms: List[float] = []
        with torch.no_grad():
            for _ in range(self.cfg.timed_iters):
                if self.device == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event   = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = model(input_ids)
                    end_event.record()
                    torch.cuda.synchronize()
                    latencies_ms.append(start_event.elapsed_time(end_event))
                else:
                    t0 = time.perf_counter()
                    _ = model(input_ids)
                    latencies_ms.append((time.perf_counter() - t0) * 1000)

        arr     = np.array(latencies_ms)
        p50     = float(np.percentile(arr, 50))
        p90     = float(np.percentile(arr, 90))
        p99     = float(np.percentile(arr, 99))
        avg_ms  = p50  # use median as representative latency

        # Throughput: tokens per second
        throughput = (seq_tokens / (avg_ms / 1000.0))

        # Speedup vs eager
        speedup = (eager_p50 / p50) if eager_p50 else 1.0

        # VRAM
        peak_vram = 0.0
        if self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        vram_overhead = max(0.0, peak_vram - eager_vram)

        return CompileModeResult(
            model_name               = model_name,
            mode                     = mode,
            compile_time_s           = compile_time_s,
            latencies_ms             = latencies_ms,
            p50_ms                   = p50,
            p90_ms                   = p90,
            p99_ms                   = p99,
            throughput_tps           = throughput,
            speedup                  = speedup,
            peak_vram_mb             = peak_vram,
            compile_vram_overhead_mb = vram_overhead,
        )

    def _failed_result(
        self,
        model_name: str,
        mode: str,
        eager: CompileModeResult,
    ) -> CompileModeResult:
        return CompileModeResult(
            model_name               = model_name,
            mode                     = mode,
            compile_time_s           = -1.0,
            latencies_ms             = [],
            p50_ms                   = float("inf"),
            p90_ms                   = float("inf"),
            p99_ms                   = float("inf"),
            throughput_tps           = 0.0,
            speedup                  = 0.0,
            peak_vram_mb             = 0.0,
            compile_vram_overhead_mb = 0.0,
        )


    def _pick_best_mode(self, results: Dict[str, CompileModeResult]) -> str:
        """
        Pick the compiled mode with the lowest p50 latency.
        Falls back to 'default' if all compiled modes failed.
        """
        compiled = {
            mode: r for mode, r in results.items()
            if mode != "eager" and r.p50_ms < float("inf")
        }
        if not compiled:
            return "default"
        return min(compiled, key=lambda m: compiled[m].p50_ms)

  

    def _print_report(self, report: CompileBenchmarkReport) -> None:
        console.print(f"\n[bold green]{report.model_name} — Compile Mode Comparison[/bold green]")
        console.print(f"  Best mode: [bold magenta]{report.best_mode}[/bold magenta]\n")

        table = Table(title="Latency & Throughput by Mode", show_lines=True)
        table.add_column("Mode",          style="cyan")
        table.add_column("Compile (s)",   justify="right")
        table.add_column("p50 (ms)",      justify="right")
        table.add_column("p90 (ms)",      justify="right")
        table.add_column("p99 (ms)",      justify="right")
        table.add_column("tok/s",         justify="right")
        table.add_column("Speedup",       justify="right", style="green")
        table.add_column("VRAM (MB)",     justify="right")
        table.add_column("VRAM overhead", justify="right")

        mode_order = ["eager"] + COMPILE_MODES
        for mode in mode_order:
            if mode not in report.results:
                continue
            r = report.results[mode]
            best_marker = " ★" if mode == report.best_mode else ""
            table.add_row(
                mode + best_marker,
                "—" if r.compile_time_s <= 0 else f"{r.compile_time_s:.1f}",
                f"{r.p50_ms:.2f}",
                f"{r.p90_ms:.2f}",
                f"{r.p99_ms:.2f}",
                f"{r.throughput_tps:,.0f}",
                f"{r.speedup:.2f}x",
                f"{r.peak_vram_mb:.0f}",
                f"+{r.compile_vram_overhead_mb:.0f}",
            )

        console.print(table)


def run_compile_sweep(
    model: torch.nn.Module,
    model_name: str,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> CompileBenchmarkReport:
    """Run the full 3-mode compile sweep and return the report."""
    benchmarker = CompileBenchmarker(device=device, dtype=dtype)
    with torch.no_grad():
        report = benchmarker.run(model, model_name)
    return report