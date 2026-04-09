"""Phase 2A checkpoint tests for Kalman filter and Van Loan covariance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from jmseq.kalman.covariance import van_loan, discrete_drift
from jmseq.kalman.filter import run_filter, batched_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numpy_kalman_scalar(a, g, b, r, m0, P0, Y, dts, mask):
    """Reference NumPy Kalman filter for scalar AR(1) system."""
    T = len(Y)
    means = np.zeros(T)
    covs = np.zeros(T)
    m, P, ll = float(m0), float(P0), 0.0
    for t in range(T):
        dt = float(dts[t])
        # Discrete transition
        A_d = np.exp(a * dt)
        # Van Loan Q_d (scalar): Q_d = G^2 * (exp(2a dt) - 1) / (2a)  when a != 0
        if abs(a) > 1e-9:
            Q_d = g**2 * (np.exp(2 * a * dt) - 1) / (2 * a)
        else:
            Q_d = g**2 * dt
        b_d = (A_d - 1) * b / a if abs(a) > 1e-9 else b * dt
        # Predict
        m_pred = A_d * m + b_d
        P_pred = A_d**2 * P + Q_d
        if mask[t]:
            # Update
            innov = Y[t] - m_pred
            S = P_pred + r
            K = P_pred / S
            m = m_pred + K * innov
            P = (1 - K) * P_pred * (1 - K) + K * r * K
            ll += -0.5 * (np.log(2 * np.pi) + np.log(S) + innov**2 / S)
        else:
            m, P = m_pred, P_pred
        means[t] = m
        covs[t] = P
    return ll, means, covs


# ---------------------------------------------------------------------------
# Van Loan tests
# ---------------------------------------------------------------------------

def test_van_loan_scalar_known():
    """Scalar case: A_d and Q_d match closed-form."""
    a = -0.5
    g = 0.3
    dt = 1.0
    A = jnp.array([[a]])
    G = jnp.array([[g]])

    A_d, Q_d = van_loan(A, G, dt)

    expected_A_d = np.exp(a * dt)
    expected_Q_d = g**2 * (np.exp(2 * a * dt) - 1) / (2 * a)

    np.testing.assert_allclose(float(A_d[0, 0]), expected_A_d, rtol=1e-5)
    np.testing.assert_allclose(float(Q_d[0, 0]), expected_Q_d, rtol=1e-5)


def test_van_loan_near_zero_drift():
    """Near A=0 the method should not overflow or produce NaN."""
    A = jnp.array([[1e-8, 0.0], [0.0, -1e-8]])
    G = jnp.array([[0.2, 0.0], [0.05, 0.1]])
    dt = 0.25

    A_d, Q_d = van_loan(A, G, dt)
    assert not jnp.any(jnp.isnan(Q_d)), "Q_d contains NaN near zero drift"
    assert not jnp.any(jnp.isnan(A_d)), "A_d contains NaN near zero drift"
    # Q_d should be approximately G G^T * dt for small A
    expected_approx = G @ G.T * dt
    np.testing.assert_allclose(np.array(Q_d), expected_approx, atol=1e-4)


def test_van_loan_symmetry():
    """Q_d must be symmetric."""
    A = jnp.array([[-0.3, 0.1], [-0.05, -0.4]])
    G = jnp.array([[0.2, 0.0], [0.1, 0.15]])
    _, Q_d = van_loan(A, G, 0.5)
    np.testing.assert_allclose(np.array(Q_d), np.array(Q_d).T, atol=1e-6)


# ---------------------------------------------------------------------------
# Kalman filter tests
# ---------------------------------------------------------------------------

def _make_scalar_params(a=-0.5, g=0.3, b_val=0.1, r=0.2, m0=1.0, P0=1.0, T=20):
    """Build JAX scalar Kalman filter inputs."""
    A = jnp.array([[a]])
    G = jnp.array([[g]])
    b = jnp.array([b_val])
    R = jnp.array([[r]])
    m0_ = jnp.array([m0])
    P0_ = jnp.array([[P0]])
    rng = np.random.default_rng(0)
    dts = jnp.array(rng.uniform(0.1, 0.5, T))
    Y_np = rng.standard_normal((T, 1)).astype(np.float32)
    Y = jnp.array(Y_np)
    mask = jnp.ones(T, dtype=bool)
    return A, G, b, R, m0_, P0_, Y, dts, mask, a, g, b_val, r, m0, P0, Y_np[:, 0], np.array(dts)


def test_filter_scalar_matches_numpy():
    """JAX filter filtered means and log-lik match NumPy reference for scalar AR(1)."""
    A, G, b, R, m0_, P0_, Y, dts, mask, a, g, b_val, r, m0, P0, Y_np, dts_np = \
        _make_scalar_params()
    ll_ref, means_ref, _ = numpy_kalman_scalar(a, g, b_val, r, m0, P0, Y_np, dts_np, np.ones(20))

    ll, means, _ = run_filter(A, G, b, R, m0_, P0_, Y, dts, mask)

    np.testing.assert_allclose(float(ll), ll_ref, rtol=1e-4,
                               err_msg="Log-likelihood mismatch vs NumPy reference")
    np.testing.assert_allclose(np.array(means[:, 0]), means_ref, rtol=1e-4,
                               err_msg="Filtered means mismatch vs NumPy reference")


def test_filter_all_missing():
    """With all obs_mask=False, filtered means equal predicted means (no updates)."""
    A, G, b, R, m0_, P0_, Y, dts, _, a, g, b_val, r, m0, P0, Y_np, dts_np = \
        _make_scalar_params()
    mask_false = jnp.zeros(20, dtype=bool)
    mask_true  = jnp.ones(20, dtype=bool)

    ll_miss, means_miss, covs_miss = run_filter(A, G, b, R, m0_, P0_, Y, dts, mask_false)
    ll_obs,  means_obs,  covs_obs  = run_filter(A, G, b, R, m0_, P0_, Y, dts, mask_true)

    # Log-likelihood should be zero for all-missing
    np.testing.assert_allclose(float(ll_miss), 0.0, atol=1e-6,
                               err_msg="LL should be 0 when all obs masked")
    # Means should differ (missing doesn't update)
    assert not np.allclose(np.array(means_miss), np.array(means_obs)), \
        "Means should differ between all-missing and all-observed"


def test_filter_ll_matches_scipy():
    """Step-by-step log-likelihood matches scipy multivariate_normal.logpdf."""
    p = 2
    T = 5
    A = jnp.array([[-0.3, 0.0], [0.0, -0.4]])
    G = jnp.array([[0.2, 0.0], [0.1, 0.15]])
    b = jnp.array([0.05, -0.02])
    R = jnp.diag(jnp.array([0.1, 0.15]))
    rng = np.random.default_rng(7)
    m0_ = jnp.array(rng.standard_normal(p).astype(np.float32))
    P0_ = jnp.eye(p)
    Y = jnp.array(rng.standard_normal((T, p)).astype(np.float32))
    dts = jnp.full(T, 0.25)
    mask = jnp.ones(T, dtype=bool)

    ll_jax, _, _ = run_filter(A, G, b, R, m0_, P0_, Y, dts, mask)

    # Compute reference step-by-step with scipy
    from jmseq.kalman.covariance import van_loan, discrete_drift
    m = np.array(m0_)
    P = np.array(P0_)
    ll_ref = 0.0
    for t in range(T):
        A_d_j, Q_d_j = van_loan(A, G, 0.25)
        b_d_j = discrete_drift(A, b, 0.25)
        A_d = np.array(A_d_j); Q_d = np.array(Q_d_j); b_d = np.array(b_d_j)
        m_pred = A_d @ m + b_d
        P_pred = A_d @ P @ A_d.T + Q_d
        S = P_pred + np.array(R)
        ll_ref += multivariate_normal.logpdf(np.array(Y[t]), mean=m_pred, cov=S)
        innov = np.array(Y[t]) - m_pred
        K = np.linalg.solve(S.T, P_pred.T).T
        m = m_pred + K @ innov
        IK = np.eye(p) - K
        P = IK @ P_pred @ IK.T + K @ np.array(R) @ K.T
        P = 0.5 * (P + P.T)

    np.testing.assert_allclose(float(ll_jax), ll_ref, rtol=1e-4,
                               err_msg="LL mismatch vs scipy step-by-step")


# ---------------------------------------------------------------------------
# Batched filter test
# ---------------------------------------------------------------------------

def test_batched_filter_sums_ll():
    """batched_filter log-lik equals sum of per-individual run_filter log-liks."""
    N, T, p = 5, 10, 2
    A = jnp.array([[-0.3, 0.05], [0.0, -0.4]])
    G = jnp.array([[0.2, 0.0], [0.1, 0.15]])
    b = jnp.array([0.05, -0.02])
    R = jnp.diag(jnp.array([0.1, 0.15]))
    P0 = jnp.eye(p)

    rng = np.random.default_rng(42)
    m0_batch = jnp.array(rng.standard_normal((N, p)).astype(np.float32))
    Y_batch  = jnp.array(rng.standard_normal((N, T, p)).astype(np.float32))
    dt_batch = jnp.full((N, T), 0.25)
    mask_batch = jnp.ones((N, T), dtype=bool)

    ll_batched, _, _ = batched_filter(A, G, b, R, m0_batch, P0, Y_batch, dt_batch, mask_batch)

    ll_sum = 0.0
    for i in range(N):
        ll_i, _, _ = run_filter(A, G, b, R, m0_batch[i], P0, Y_batch[i], dt_batch[i], mask_batch[i])
        ll_sum += float(ll_i)

    np.testing.assert_allclose(float(ll_batched), ll_sum, rtol=1e-5,
                               err_msg="Batched LL doesn't match sum of individual LLs")
