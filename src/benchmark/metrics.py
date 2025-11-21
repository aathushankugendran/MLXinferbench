# src/benchmark/metrics.py

"""
Metric utilities for benchmarking.

Used by benchmark runners (e.g., MLX, PyTorch, vLLM backends) to compute:
  * latency in ms
  * throughput in tokens/sec
  * normalized metric rows for CSV logging
"""

from __future__ import annotations

from typing import Dict, Any


def compute_latency_ms(start_seconds: float, end_seconds: float) -> float:
    """
    Convert a start/end timestamp pair (in seconds) into latency in milliseconds.
    """
    return (end_seconds - start_seconds) * 1000.0


def compute_throughput(tokens_generated: int, latency_seconds: float) -> float:
    """
    Compute tokens per second given:
      - tokens_generated: number of new tokens produced
      - latency_seconds: wall-clock time for generation
    """
    if latency_seconds <= 0:
        return 0.0
    return tokens_generated / latency_seconds


def build_metric_row(
    backend: str,
    model_name: str,
    context_length: int,
    generate_length: int,
    trial_index: int,
    precision: str,
    run_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize a single run result into a flat dict suitable for CSV logging.

    Args:
        backend: e.g. "mlx"
        model_name: e.g. "gpt2"
        context_length: size of the input prompt (approx tokens/chars)
        generate_length: requested number of tokens to generate
        trial_index: 1-based trial index
        precision: e.g. "fp32", "fp16"
        run_result: dict returned from the backend inference, expected keys:
            - "latency_ms"
            - "latency_seconds"
            - "tokens_generated"
            - "tokens_per_second"
            - "generated_text" (not strictly needed for metrics, but can be logged)

    Returns:
        A flat dict (string -> scalar) for CSV.
    """
    return {
        "backend": backend,
        "model": model_name,
        "precision": precision,
        "context_length": context_length,
        "generate_length": generate_length,
        "trial": trial_index,
        "latency_ms": run_result.get("latency_ms"),
        "latency_seconds": run_result.get("latency_seconds"),
        "tokens_generated": run_result.get("tokens_generated"),
        "tokens_per_second": run_result.get("tokens_per_second"),
        # Optional: helpful for debugging
        "generated_text": run_result.get("generated_text", ""),
    }
