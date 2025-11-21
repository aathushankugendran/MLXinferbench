# src/backends/mlx_backend/mlx_models.py

"""
Utilities for loading MLX LLMs (e.g., GPT-2) and their tokenizers.

This uses the `mlx-lm` package, which provides:
    from mlx_lm import load, generate
"""

from typing import Tuple

from mlx_lm import load  # HF + MLX integration


# Map simple model names to Hugging Face model IDs in MLX format.
# For now we reliably support base GPT-2 via mlx-community.
# You can extend this dict with more models as you convert/add them.
MODEL_ID_MAP = {
    # GPT-2 family (fully available in MLX format)
    "gpt2": "mlx-community/gpt2-base-mlx",             # 124M
    #"gpt2-medium": "mlx-community/gpt2-medium-mlx",    # 355M
    #"gpt2-large": "mlx-community/gpt2-large-mlx",      # 774M
    #"gpt2-xl": "mlx-community/gpt2-xl-mlx",            # 1.5B

    # Optional LLaMA-style small models
    #"llama-1b": "mlx-community/llama-1b-mlx",
    #"llama-3b": "mlx-community/llama-3b-mlx",
    # 7B is possible but risky on 16GB
    ##"llama-7b": "mlx-community/llama-7b-mlx",
}



def get_hf_repo_id(model_name: str) -> str:
    """
    Translate a simple model name (e.g. 'gpt2') into a Hugging Face repo id.

    Raises:
        KeyError: if the model name is not known.
    """
    if model_name not in MODEL_ID_MAP:
        available = ", ".join(MODEL_ID_MAP.keys())
        raise KeyError(
            f"Unknown model_name='{model_name}'. "
            f"Known models: {available}. "
            "To add more, edit MODEL_ID_MAP in mlx_models.py."
        )
    return MODEL_ID_MAP[model_name]


def load_model_and_tokenizer(model_name: str = "gpt2"):
    """
    Load an MLX model + tokenizer via mlx-lm.

    Args:
        model_name: Simple name like 'gpt2'.

    Returns:
        model: MLX model object
        tokenizer: tokenizer compatible with the model
    """
    hf_repo_id = get_hf_repo_id(model_name)
    print(f"[mlx_models] Loading '{model_name}' from Hugging Face repo '{hf_repo_id}'...")
    model, tokenizer = load(hf_repo_id)
    print("[mlx_models] Model loaded.")
    return model, tokenizer