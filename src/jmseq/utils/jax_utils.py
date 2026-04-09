"""
JAX utility helpers for the jmseq package.
"""

import jax
import jax.numpy as jnp
import numpy as np


def safe_positive(x: jnp.ndarray, min_val: float = 1e-6) -> jnp.ndarray:
    """Clamp array to be strictly positive."""
    return jnp.clip(x, a_min=min_val)


def stable_log(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Log with a small floor to avoid -inf."""
    return jnp.log(jnp.clip(x, a_min=eps))


def pad_to_length(arr: np.ndarray, target_len: int, pad_value=0.0) -> np.ndarray:
    """Pad a 1-D numpy array to target_len with pad_value."""
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.concatenate([arr, np.full(target_len - len(arr), pad_value, dtype=arr.dtype)])


def vmap_over_individuals(fn):
    """Decorator: vmap a function over its first argument (individual axis)."""
    return jax.vmap(fn)


def count_params(params: dict) -> int:
    """Count total number of scalar parameters in a NumPyro params dict."""
    total = 0
    for v in params.values():
        arr = np.asarray(v)
        total += arr.size
    return total
