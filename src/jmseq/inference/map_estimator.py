"""
MAP and SVI estimators wrapping NumPyro models.
"""

from typing import Any, Callable, Optional, Type

import jax
import jax.numpy as jnp
import numpyro
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoNormal


def fit_map(
    model: Callable,
    model_args: tuple = (),
    model_kwargs: Optional[dict] = None,
    n_steps: int = 10_000,
    lr: float = 0.01,
    rng_key: Optional[Any] = None,
) -> tuple[dict, Any]:
    """MAP estimation via AutoDelta guide + SVI with ClippedAdam.

    Returns (median_params, losses) where median_params is a dict keyed by
    latent site name (constrained space), and losses is an (n_steps,) array.

    Parameters
    ----------
    model        : NumPyro model callable.
    model_args   : positional arguments forwarded to model.
    model_kwargs : keyword arguments forwarded to model.
    n_steps      : number of SVI steps.
    lr           : initial learning rate for ClippedAdam.
    rng_key      : JAX PRNGKey; defaults to key(0).
    """
    if model_kwargs is None:
        model_kwargs = {}
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    guide = AutoDelta(model)
    optimizer = optim.ClippedAdam(step_size=lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Use a Python loop rather than lax.scan so that only the inner model
    # (Kalman filter scan) gets JIT-compiled — avoiding a huge nested trace
    # that compounds JIT compilation time for large datasets.
    svi_state = svi.init(rng_key, *model_args, **model_kwargs)
    update_jit = jax.jit(svi.update)
    losses = []
    for _ in range(n_steps):
        svi_state, loss = update_jit(svi_state, *model_args, **model_kwargs)
        losses.append(float(loss))
    losses_arr = jnp.array(losses)

    params = svi.get_params(svi_state)
    median = guide.median(params)

    # Run Predictive to collect deterministic sites (e.g. beta = R_ast_inv @ theta)
    predictive = Predictive(model, guide=guide, params=params, num_samples=1)
    pred = predictive(rng_key, *model_args, **model_kwargs)
    # Merge: deterministic sites from predictive, latent sites from median
    combined = {k: v[0] for k, v in pred.items() if k not in median}
    combined.update(dict(median))
    return combined, losses_arr


def fit_svi(
    model: Callable,
    model_args: tuple = (),
    model_kwargs: Optional[dict] = None,
    guide_cls: Type = AutoNormal,
    n_steps: int = 20_000,
    lr: float = 0.01,
    rng_key: Optional[Any] = None,
) -> tuple:
    """Variational Bayes via AutoNormal (or custom) guide + SVI.

    Returns (guide, params, losses) for downstream posterior sampling via
    ``guide.sample_posterior(rng_key, params)``.

    Parameters
    ----------
    model        : NumPyro model callable.
    model_args   : positional arguments forwarded to model.
    model_kwargs : keyword arguments forwarded to model.
    guide_cls    : AutoGuide class (default AutoNormal).
    n_steps      : number of SVI steps.
    lr           : initial learning rate.
    rng_key      : JAX PRNGKey; defaults to key(0).
    """
    if model_kwargs is None:
        model_kwargs = {}
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    guide = guide_cls(model)
    optimizer = optim.ClippedAdam(step_size=lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    svi_state = svi.init(rng_key, *model_args, **model_kwargs)
    update_jit = jax.jit(svi.update)
    losses = []
    for _ in range(n_steps):
        svi_state, loss = update_jit(svi_state, *model_args, **model_kwargs)
        losses.append(float(loss))

    params = svi.get_params(svi_state)
    return guide, params, jnp.array(losses)
