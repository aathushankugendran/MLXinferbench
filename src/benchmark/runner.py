# src/benchmark/runner.py
"""
Simple benchmarking harness for LLM inference.

This script is meant to be run from the repo root, for example:

    python -m src.benchmark.runner \
        --model gpt2 \
        --context 128 256 \
        --generate 32 \
        --precision fp32 \
        --trials 5 \
        --output data/benchmarks.csv

High-level responsibilities:
  * Parse CLI arguments (model, context lengths, generate lengths, etc.).
  * Build simple dummy prompts that roughly scale with context length.
  * Load an MLX-backed model + tokenizer ONCE per run.
  * Run multiple timed inference trials for each configuration.
  * Compute per-trial metrics (latency, tokens/sec).
  * Append all results to a CSV file for later analysis.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict, Any

from mlx_lm import generate  # type: ignore # MLX generation helper
from src.backends.mlx_backend.mlx_models import load_model_and_tokenizer
from src.benchmark.metrics import build_metric_row
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    """
    Define and parse the command-line interface for the benchmark runner.

    Returns:
        argparse.Namespace containing all parsed CLI arguments.
    """
    # Create a parser with a short description for --help output.
    parser = argparse.ArgumentParser(description="LLM benchmarking harness")

    # --model: simple string identifier for the model we want to load.
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (as understood by the backend, e.g. 'gpt2').",
    )

    # --context: one or more prompt lengths to sweep over.
    # Using nargs="+" allows passing multiple values like: --context 128 256 512
    parser.add_argument(
        "--context",
        type=int,
        nargs="+",
        default=[128],
        help="One or more context sizes (approx prompt lengths).",
    )

    # --generate: one or more generation lengths to sweep over.
    parser.add_argument(
        "--generate",
        type=int,
        nargs="+",
        default=[32],
        help="One or more generation lengths (max tokens).",
    )

    # --precision: currently a tag for logging; later can control actual dtypes.
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision tag to log (does not yet change backend behavior).",
    )

    # --trials: how many times we repeat each (context, generate) combo.
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per (context, generate) configuration.",
    )

    # --output: where to write the CSV file with benchmark results.
    parser.add_argument(
        "--output",
        type=str,
        default="data/benchmarks.csv",
        help="Path to CSV file where metrics will be appended.",
    )

    # Actually parse the arguments from sys.argv and return them.
    return parser.parse_args()


def make_dummy_prompt(context_length: int) -> str:
    """
    Create a very simple dummy prompt that roughly scales with `context_length`.

    The exact token count is not critical here; we just need something that
    grows in length so we can test how latency changes with context size.

    Args:
        context_length: Approximate length target for the prompt (in characters).

    Returns:
        A string prompt composed of repeated words.
    """
    # Base word we repeat. It doesn't matter semantically; we're just filling space.
    base_word = "hello "

    # Compute how many times we need to repeat the base word to get close to
    # the desired context length (in characters). We ensure at least 1 repeat.
    repeats = max(context_length // len(base_word), 1)

    # Build the prompt by repeating "hello " and trimming any trailing whitespace.
    prompt = (base_word * repeats).strip()
    return prompt


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure that the parent directory of `path` exists.

    This is used before writing the CSV file so that `data/` or any nested
    directory is automatically created if it doesn't already exist.
    """
    parent = path.parent
    if not parent.exists():
        # Create all missing parent directories (e.g., "data/benchmarks/").
        parent.mkdir(parents=True, exist_ok=True)


def write_rows_to_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Append a list of metric rows to a CSV file.

    If the file does not exist yet, this function also writes a header row
    based on the keys of the first row in `rows`.

    Args:
        csv_path: Path where the CSV file should be written.
        rows: List of dictionaries, all with the same keys, one per trial.
    """
    # If there are no rows to write, do nothing and return early.
    if not rows:
        return

    # Make sure the directory for the CSV file exists.
    ensure_parent_dir(csv_path)

    # All rows should share the same keys; we use the first row to get the header.
    fieldnames = list(rows[0].keys())

    # Check if the file already exists so we know whether to write a header.
    file_exists = csv_path.exists()

    # Open the file in append mode so benchmarks can accumulate over time.
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # If the file is new, write the header row once.
        if not file_exists:
            writer.writeheader()

        # Write each row as a CSV record.
        for row in rows:
            writer.writerow(row)


def run_inference_with_model(
    model,
    tokenizer,
    prompt: str,
    model_name: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Run a single inference using a preloaded model + tokenizer and measure latency.

    This function mirrors the behavior of `run_single_inference` from the MLX
    backend, but operates directly on an already-loaded model + tokenizer so we
    avoid paying the model load cost every trial.

    Args:
        model: Loaded MLX model instance returned by `load_model_and_tokenizer`.
        tokenizer: Corresponding tokenizer instance.
        prompt: Input text prompt to feed into the model.
        model_name: String name of the model (mostly for logging/metadata).
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A dictionary with the following keys (compatible with `build_metric_row`):
            - "generated_text"
            - "latency_seconds"
            - "latency_ms"
            - "tokens_generated"
            - "tokens_per_second"
    """
    # Record the wall-clock time just before calling `generate`.
    start = time.perf_counter()

    # Use MLX's high-level `generate` helper to produce new tokens.
    generated_text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,  # we handle printing/logging ourselves
    )

    # Record time immediately after generation completes.
    end = time.perf_counter()

    # Convert the elapsed seconds into both seconds and milliseconds.
    latency_seconds = end - start
    latency_ms = latency_seconds * 1000.0

    # To get a rough token count, we compare the length of the encoded
    # prompt vs. the encoded full output (prompt + completion).
    prompt_ids = tokenizer.encode(prompt)
    output_ids = tokenizer.encode(generated_text)
    tokens_generated = len(output_ids) - len(prompt_ids)

    # Guard against pathological cases where the model returns nothing new.
    # We clamp at 1 to avoid division-by-zero in throughput.
    tokens_generated = max(tokens_generated, 1)

    # Compute throughput as tokens per second.
    tokens_per_second = (
        tokens_generated / latency_seconds if latency_seconds > 0 else 0.0
    )

    return {
        "generated_text": generated_text,
        "latency_seconds": latency_seconds,
        "latency_ms": latency_ms,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
    }


def main() -> None:
    """
    Entry point for the benchmark runner.

    Orchestrates:
      * parsing CLI arguments
      * setting up logging
      * loading the MLX model + tokenizer once
      * running timed inference trials over all requested configs
      * writing metrics to a CSV file
    """
    # Parse all CLI arguments from the user.
    args = parse_args()

    # Create a logger instance for this module. All logs from this runner
    # will be tagged with the name "benchmark.runner".
    logger = setup_logger("benchmark.runner", log_level="INFO")

    # Hard-coded backend name for now. In the future, this could be a CLI flag
    # (e.g., "mlx", "torch", "vllm", "tensorrt").
    backend_name = "mlx"

    # Convert the output path string into a Path object for convenience.
    csv_path = Path(args.output)

    # We'll accumulate all trial results into this list, then flush them
    # to disk once at the end.
    all_rows: List[Dict[str, Any]] = []

    # Log the high-level configuration for traceability.
    logger.info(f"Backend:          {backend_name}")
    logger.info(f"Model:            {args.model}")
    logger.info(f"Context sizes:    {args.context}")
    logger.info(f"Generate sizes:   {args.generate}")
    logger.info(f"Trials per config:{args.trials}")
    logger.info(f"Output CSV:       {csv_path}")

    # Load the model + tokenizer exactly once, outside the trial loops.
    # This ensures our measured latency only reflects inference, not load time.
    logger.info(f"Loading model '{args.model}' via MLX backend...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    logger.info("Model loaded. Starting benchmarks...")

    # Outer loop over different context lengths (prompt sizes).
    for context_length in args.context:
        # Build a dummy prompt for this specific context length.
        prompt = make_dummy_prompt(context_length)

        # Inner loop over different generation lengths (how many tokens to sample).
        for generate_length in args.generate:
            logger.info(
                f"Configuration: context={context_length}, "
                f"generate={generate_length}"
            )

            # Repeat the same configuration `args.trials` times to get
            # a distribution of latency / throughput measurements.
            for trial in range(1, args.trials + 1):
                logger.info(f"  Trial {trial}/{args.trials}...")

                # Perform a single timed inference using the preloaded model.
                run_result = run_inference_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    model_name=args.model,
                    max_tokens=generate_length,
                )

                # Convert the raw run_result dict into a standardized metric row
                # that includes backend, model, config, and trial index.
                row = build_metric_row(
                    backend=backend_name,
                    model_name=args.model,
                    context_length=context_length,
                    generate_length=generate_length,
                    trial_index=trial,
                    precision=args.precision,
                    run_result=run_result,
                )
                all_rows.append(row)

                # Log a short summary of this trial to the console.
                logger.info(
                    f"    latency={row['latency_ms']:.2f} ms, "
                    f"tokens/s={row['tokens_per_second']:.2f}"
                )

    # Once all trials across all configurations are complete,
    # append the metrics to the CSV file.
    write_rows_to_csv(csv_path, all_rows)
    logger.info(f"Wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    # Standard Python entry-point guard so this file can be imported
    # as a module without immediately running the benchmark.
    main()
