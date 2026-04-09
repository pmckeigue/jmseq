"""Phase 1 checkpoint tests for Poisson GLM with QR decomposition."""

import jax.numpy as jnp
import numpy as np
import pytest

from jmseq.models.poisson_glm import poisson_glm_model, qr_decompose
from jmseq.inference.map_estimator import fit_map


# ---------------------------------------------------------------------------
# QR decomposition tests
# ---------------------------------------------------------------------------

def test_qr_output_shapes(poisson_glm_data):
    X = poisson_glm_data["X"]
    N, P = X.shape
    Q_ast, R_ast, R_ast_inv = qr_decompose(X)
    assert Q_ast.shape == (N, P)
    assert R_ast.shape == (P, P)
    assert R_ast_inv.shape == (P, P)


def test_qr_matches_numpy(poisson_glm_data):
    """Q_ast and R_ast reproduce X up to sign convention."""
    X = poisson_glm_data["X"]
    N = X.shape[0]
    Q_ast, R_ast, _ = qr_decompose(X)
    scale = np.sqrt(N - 1)
    # Q_ast / scale is orthonormal; (Q_ast / scale) @ R_ast * scale should equal X
    X_recovered = (Q_ast / scale) @ (R_ast * scale)
    np.testing.assert_allclose(np.array(X_recovered), np.array(X), atol=1e-5)


def test_r_ast_inverse(poisson_glm_data):
    """R_ast_inv @ R_ast == I to machine precision."""
    X = poisson_glm_data["X"]
    _, R_ast, R_ast_inv = qr_decompose(X)
    P = R_ast.shape[0]
    product = R_ast_inv @ R_ast
    np.testing.assert_allclose(np.array(product), np.eye(P), atol=1e-5)


# ---------------------------------------------------------------------------
# MAP estimation test
# ---------------------------------------------------------------------------

def test_map_recovers_true_beta(poisson_glm_data):
    """MAP beta should be within 0.1 of true_beta (inf-norm)."""
    X = poisson_glm_data["X"]
    y = poisson_glm_data["y"]
    log_tobs = poisson_glm_data["log_tobs"]
    true_beta = poisson_glm_data["true_beta"]

    Q_ast, _, R_ast_inv = qr_decompose(X)

    params, losses = fit_map(
        poisson_glm_model,
        model_args=(Q_ast, y, log_tobs),
        model_kwargs={"R_ast_inv": R_ast_inv},
        n_steps=5_000,
        lr=0.05,
    )

    assert "beta" in params, "beta deterministic site missing from MAP output"
    beta_est = np.array(params["beta"])
    err = np.max(np.abs(beta_est - np.array(true_beta)))
    assert err < 0.1, f"MAP beta error {err:.4f} exceeds tolerance 0.1; beta_est={beta_est}"


def test_losses_decrease(poisson_glm_data):
    """ELBO loss should decrease over training."""
    X = poisson_glm_data["X"]
    y = poisson_glm_data["y"]
    log_tobs = poisson_glm_data["log_tobs"]

    Q_ast, _, R_ast_inv = qr_decompose(X)

    _, losses = fit_map(
        poisson_glm_model,
        model_args=(Q_ast, y, log_tobs),
        model_kwargs={"R_ast_inv": R_ast_inv},
        n_steps=2_000,
        lr=0.05,
    )
    # Average of last 100 steps should be lower than average of first 100
    assert float(losses[:100].mean()) > float(losses[-100:].mean()), \
        "Loss did not decrease over training"
