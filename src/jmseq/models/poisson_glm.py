"""
Poisson GLM with QR decomposition — NumPyro port of poissonglm.stan.

Stan model recap:
  Q* = qr_thin_Q(X)[:, 1:P] * sqrt(N-1)
  R* = qr_thin_R(X)[1:P, ] / sqrt(N-1)
  eta_i = log(tobs_i) + beta0 + (Q* theta)_i
  y_i ~ Poisson(exp(eta_i))
  beta = R*^{-1} theta          (generated quantity)
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def qr_decompose(X):
    """Thin QR decomposition scaled by sqrt(n-1).

    Parameters
    ----------
    X : array (N, P)  — design matrix WITHOUT intercept column.

    Returns
    -------
    Q_ast       : (N, P)   Q scaled by sqrt(N-1)
    R_ast       : (P, P)   R scaled by 1/sqrt(N-1)
    R_ast_inv   : (P, P)   inverse of R_ast
    """
    N = X.shape[0]
    scale = jnp.sqrt(jnp.array(N - 1, dtype=X.dtype))
    Q, R = jnp.linalg.qr(X, mode="reduced")
    # Absorb sign convention: force positive diagonal of R (matches Stan)
    signs = jnp.sign(jnp.diag(R))
    Q = Q * signs[None, :]
    R = R * signs[:, None]
    Q_ast = Q * scale
    R_ast = R / scale
    R_ast_inv = jnp.linalg.inv(R_ast)
    return Q_ast, R_ast, R_ast_inv


def poisson_glm_model(Q_ast, y, log_tobs, R_ast_inv=None):
    """NumPyro Poisson GLM model in the QR-rotated parameter space.

    Parameters
    ----------
    Q_ast    : (N, P)  QR-rotated design matrix (from qr_decompose).
    y        : (N,)    integer event counts (0 or 1 per person-time interval).
    log_tobs : (N,)    log interval length (offset).
    R_ast_inv: (P, P)  if provided, register beta = R_ast_inv @ theta as
                       a deterministic site for easy retrieval.
    """
    P = Q_ast.shape[1]
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, 10.0))
    theta = numpyro.sample("theta", dist.Normal(jnp.zeros(P), 2.5 * jnp.ones(P)))
    log_mu = log_tobs + beta0 + Q_ast @ theta
    numpyro.sample("y", dist.Poisson(jnp.exp(log_mu)), obs=y)
    if R_ast_inv is not None:
        numpyro.deterministic("beta", R_ast_inv @ theta)
