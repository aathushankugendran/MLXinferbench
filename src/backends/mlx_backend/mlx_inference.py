# src/backends/mlx_backend/mlx_inference.py

"""
Minimal MLX inference benchmark.

Run with:
    python -m src.backends.mlx_backend.mlx_inference

What it does:
    * loads a GPT-2 model in MLX format
    * runs generation on a dummy prompt
    * measures end-to-end generation latency
    * prints latency and tokens/sec
"""

import time
from typing import Dict, Any

from mlx_lm import generate
from .mlx_models import load_model_and_tokenizer


def run_single_inference(
    prompt: str,
    model_name: str = "gpt2",
    max_tokens: int = 32,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single inference pass using mlx-lm's generate() and measure latency.

    Args:
        prompt: Text prompt to feed into the model.
        model_name: Simple model name (e.g. 'gpt2').
        max_tokens: How many tokens to generate.
        verbose: Whether to print the generated text.

    Returns:
        A dict with:
            - "generated_text"
            - "latency_seconds"
            - "latency_ms"
            - "tokens_generated"
            - "tokens_per_second"
    """
    # 1) Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # 2) Prepare prompt (for chat-style models you might use apply_chat_template;
    #    for GPT-2 we just use raw text).
    prompt_text = prompt

    # 3) Time the generation
    start = time.perf_counter()
    generated_text = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        verbose=False,  # we handle printing ourselves
    )
    end = time.perf_counter()

    latency_seconds = end - start
    latency_ms = latency_seconds * 1000.0

    # Rough token count: length of generated output tokens only.
    # For more precise measurement you could use tokenizer on the output.
    tokens_generated = len(tokenizer.encode(generated_text)) - len(
        tokenizer.encode(prompt_text)
    )
    tokens_generated = max(tokens_generated, 1)  # avoid div by zero

    tokens_per_second = tokens_generated / latency_seconds

    if verbose:
        print("\n=== Inference Result ===")
        print(f"Model:           {model_name}")
        print(f"Prompt:          {repr(prompt_text)}")
        print(f"Generated text:\n{generated_text}")
        print("\n=== Timing ===")
        print(f"Latency:         {latency_ms:.2f} ms")
        print(f"Tokens generated:{tokens_generated}")
        print(f"Tokens / second: {tokens_per_second:.2f}")

    return {
        "generated_text": generated_text,
        "latency_seconds": latency_seconds,
        "latency_ms": latency_ms,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
    }


if __name__ == "__main__":
    # Quick sanity test
    default_prompt = "In a surprising discovery, researchers found that"
    _ = run_single_inference(
        prompt=default_prompt,
        model_name="gpt2",
        max_tokens=32,
        verbose=True,
    )
