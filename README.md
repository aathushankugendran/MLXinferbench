
# MLXinferbench

Lightweight benchmarking harness for LLM inference backends. MLXinferbench
provides small, focused tools to run deterministic inference trials, measure
latency and throughput, and collect results for comparison and visualization.

This repository contains a simple MLX-backed runner (GPT-2 by default),
utilities for loading MLX models, metric computation and CSV logging, and
helpers for generating repeatable dummy prompts.

**Status:** small research/utility project — examples and helpers for MLX
inference experiments.

**Contents (high level)**
- `src/backends/mlx_backend/` — MLX-specific loading and a minimal inference
	runner (`mlx_inference.py`).
- `src/benchmark/` — benchmarking harness and metric helpers (`runner.py`,
	`metrics.py`, `profiler.py`).
- `src/utils/` — small utilities (`logging.py`, `memory.py`).
- `data/benchmarks.csv` — example output CSV (ignored via `.gitignore`).

## Quickstart

Prerequisites

- Python 3.10+ recommended (see `requirements.txt` for pinned deps).
- A working installation of the `mlx-lm`/`mlx` ecosystem to load MLX-format
	models. The example uses `mlx-community/gpt2-base-mlx` via the Hugging Face
	hub.

Install dependencies (recommended in a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the small MLX inference demo (from repository root):

```bash
python -m src.backends.mlx_backend.mlx_inference
```

This loads the default `gpt2` mapping, runs a single generation, prints the
generated text and timing metrics.

Run the benchmark harness (example):

```bash
python -m src.benchmark.runner \
	--model gpt2 \
	--context 128 \
	--generate 32 \
	--trials 1 \
	--output data/benchmarks.csv
```

The runner will load the specified model once and perform the requested
trials, appending results to the CSV file.

## How it works (short)

- `src/backends/mlx_backend/mlx_models.py` maps short model names (e.g.
	`gpt2`) to Hugging Face repo ids in MLX format and uses `mlx_lm.load` to
	obtain a `(model, tokenizer)` pair.
- `src/backends/mlx_backend/mlx_inference.py` provides `run_single_inference()`
	that calls `mlx_lm.generate()` and measures latency and tokens/sec.
- `src/benchmark/runner.py` offers a CLI to sweep context/generate lengths and
	record per-trial metrics via `src/benchmark/metrics.py`.

## Output & Logging

- CSV output: `data/benchmarks.csv` — rows include `backend, model, precision,
	context_length, generate_length, trial, latency_ms, tokens_per_second, ...`.
- Logging: use `src/utils/logging.setup_logger()` to configure console/file
	logs for longer runs.

## Extending / Adding Models

- Add new model mappings to `MODEL_ID_MAP` in
	`src/backends/mlx_backend/mlx_models.py`.
- For other backends (PyTorch, vLLM, etc.) implement a backend module that
	exposes an interface similar to `run_single_inference()` so the runner can
	reuse `build_metric_row()` and CSV logging.

## Development tips

- Use `git add -p` to make granular commits for each logical change.
- The project ignores `data/` in `.gitignore`; sample CSVs may be present but
	are not committed.

## Contributing

PRs welcome. Keep changes small and include focused commit messages that
explain the motivation and any trade-offs.

## License

See `LICENSE` in the repository root.

