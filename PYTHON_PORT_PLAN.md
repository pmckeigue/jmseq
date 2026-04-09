# Python Port Plan for jmseq

## Overview

Port the jmseq R package to Python, eliminating Stan, rstan, and ctsem. Start by replacing Stan models with NumPyro; then replace ctsem with a JAX Kalman filter + NumPyro LGSSM.

---

## Proposed Package Structure

```
jmseq/
├── pyproject.toml                  # Build system config
├── PYTHON_PORT_PLAN.md             # This file
├── tests/
│   ├── conftest.py                 # Shared fixtures: synthetic data generators
│   ├── test_poisson_glm.py         # Phase 1 tests
│   ├── test_kalman.py              # Phase 2 tests (filter correctness)
│   ├── test_lgssm.py               # Phase 2 tests (parameter estimation)
│   └── test_pipeline.py            # Phase 3 integration tests
└── src/
    └── jmseq/
        ├── __init__.py             # Public API surface
        ├── models/
        │   ├── __init__.py
        │   ├── poisson_glm.py      # NumPyro Poisson GLM with QR decomposition
        │   ├── lgssm.py            # NumPyro LGSSM model definitions (5 variants)
        │   └── model_config.py     # Dataclass: which params are free/fixed per variant
        ├── inference/
        │   ├── __init__.py
        │   ├── map_estimator.py    # MAP via jaxopt.LBFGSB wrapping JAX log-joint
        │   └── svi_estimator.py    # NumPyro SVI with AutoNormal / AutoDelta guide
        ├── kalman/
        │   ├── __init__.py
        │   ├── filter.py           # Core Kalman filter using jax.lax.scan
        │   ├── smoother.py         # RTS smoother (optional, for diagnostics)
        │   └── covariance.py       # Van Loan method, matrix exponential utilities
        ├── data/
        │   ├── __init__.py
        │   ├── transforms.py       # split_surv_long, interval splitting logic
        │   ├── splits.py           # trainsplit_surv, trainsplit_long (CV folds)
        │   └── schema.py           # Expected column names, dtype contracts
        ├── pipeline/
        │   ├── __init__.py
        │   ├── train.py            # fit_lgssm_fold + run_kalman_fold + fit_poisson_fold
        │   └── predict.py          # predict_testdata, tabulate_predictions
        └── utils/
            ├── __init__.py
            ├── jax_utils.py        # vmap helpers, safe_expm, numerical guards
            └── metrics.py          # C-statistic, log-score, calibration tables
```

---

## Key Dependencies

| Package | Version | Role |
|---|---|---|
| `jax` | >=0.4.25 | Array backend, autodiff, `lax.scan` |
| `jaxlib` | matching jax | XLA kernels |
| `numpyro` | >=0.15 | Probabilistic models, SVI, MCMC |
| `jaxopt` | >=0.8 | L-BFGS-B with JAX gradients (for MAP) |
| `optax` | >=0.2 | Gradient-based optimisers |
| `numpy` | >=1.26 | Interop, data preparation |
| `pandas` | >=2.1 | Data management (replaces data.table) |
| `scipy` | >=1.12 | Reference expm, fallback optimisers |
| `lifelines` | >=0.29 | C-statistic, survival utilities |

---

## Phase 1 — Poisson GLM (eliminates poissonglm.stan + rstan)

### What is being replaced
- `stan/poissonglm.stan`: Poisson GLM with QR decomposition
- `rstan::optimizing()` call in `fit.poissontsplit()` (MAP mode)
- `rstan::vb()` call in `fit.poissontsplit()` (variational Bayes mode)

### Mathematical specification
The Stan model:
- Thin QR decomposition: `X = Q* R*` where `Q* = Q * sqrt(n-1)`, `R* = R / sqrt(n-1)`
- Linear predictor: `η_i = log(tobs_i) + beta0 + (Q* θ)_i`
- Likelihood: `y_i ~ Poisson(exp(η_i))`
- No explicit priors (Stan uniform → use weak Normal priors in NumPyro)
- Generated quantity: `beta = R*_inverse @ theta`

### Functions to implement

**`src/jmseq/models/poisson_glm.py`**

- `qr_decompose(X)` → `(Q_ast, R_ast, R_ast_inverse)`  
  Uses `jnp.linalg.qr`, scales by `sqrt(n-1)`. Done outside the NumPyro model.

- `poisson_glm_model(Q_ast, y, log_tobs, R_ast_inverse=None)`  
  NumPyro model. Samples `beta0 ~ Normal(0, 10)`, `theta ~ Normal(0, 2.5)`.  
  Computes `eta`, observes `y ~ Poisson(exp(eta))`.  
  Adds `numpyro.deterministic("beta", R_ast_inverse @ theta)` when provided.

**`src/jmseq/inference/map_estimator.py`**

- `fit_map(model, args, kwargs)` → `dict` of MAP parameter estimates  
  Wrap `SVI` with `AutoDelta` guide + `optax.adam` or `jaxopt.LBFGSB`.

- `fit_svi(model, args, guide_cls=AutoNormal, n_steps=20000)` → `SVIRunResult`  
  Approximate-posterior path matching `rstan::vb`.

### NumPyro notes
- Use `AutoDelta` guide + `SVI` with `ClippedAdam` for MAP (standard NumPyro pattern).
- `theta` is on a more isotropic scale than `beta` (QR advantage); `Normal(0, 2.5)` on `theta` is preferable.
- QR decomposition is done once on training `X`; for prediction apply `X_test @ R_ast_inverse.T` (do not re-decompose).

### Phase 1 checkpoint tests
- Fit on synthetic `y ~ Poisson(exp(log_tobs + 0.5 + X @ true_beta))`; assert `|beta_est - true_beta|_inf < 0.05`.
- Compare QR output against `numpy.linalg.qr` reference.
- Verify `R_ast_inverse @ R_ast = I` to machine precision.
- Assert `AutoDelta` MAP and `jaxopt.LBFGSB` MAP agree to 3 significant figures.

---

## Phase 2 — LGSSM + Kalman Filter (eliminates ctsem + Stan)

This has two sub-tasks that can be developed in parallel: **(A)** Kalman filter for given parameters; **(B)** LGSSM parameter estimation.

### 2A. Kalman Filter (`src/jmseq/kalman/`)

#### State space notation
State dimension `p` (typically 2 biomarkers). Observation dimension `d = p` (LAMBDA = identity).

Continuous-time SDE: `dX(t) = (A X(t) + b) dt + G dW(t)`

Discrete-time (interval Δt):
- `x_{t+1} = A_d x_t + b_d + w_t`,  `w_t ~ N(0, Q_d)`
- `y_t = x_t + v_t`,  `v_t ~ N(0, R)`

where `A_d(Δt) = expm(A * Δt)`, `b_d` and `Q_d` from Van Loan method.

#### Van Loan method for Q_d (`covariance.py`)

Avoids any matrix inverse; stable near A=0:

1. Form `M = Δt * [[-A, G G^T], [0, A^T]]` (2p × 2p)
2. Compute `expm(M)`
3. `F = expm(M)[p:, p:]^T` (equals `expm(A Δt)`)
4. `Q_d = F @ expm(M)[:p, p:]`

Use `jax.scipy.linalg.expm`. Fully differentiable via JAX's custom VJP.

#### Kalman filter (`filter.py`)

Use `jax.lax.scan` over time steps. Carry: `(m, P, log_likelihood)`.

Per-step function:
```
Predict:
  A_d, Q_d = van_loan(A, G, dt)
  m_pred = A_d @ m + b_d
  P_pred = A_d @ P @ A_d.T + Q_d

Update (gated by obs_mask):
  innov = y - m_pred
  S = P_pred + R
  K = P_pred @ solve(S, I)
  m_upd = m_pred + K @ innov                   (Joseph form)
  P_upd = (I - K) @ P_pred @ (I - K).T + K @ R @ K.T

  m_out = where(obs_mask, m_upd, m_pred)
  P_out = where(obs_mask, P_upd, P_pred)
  ll += where(obs_mask, gaussian_log_prob(innov, S), 0)
```

**Missing observations**: `obs_mask` boolean per step (or per-dimension for partial missingness). `jnp.where` gates both update and log-likelihood — no branching inside scan.

**Batching**: `jax.vmap(run_filter, in_axes=0)` over individuals. Pad all to `max_T` with `obs_mask=False` for padding steps.

**Numerical stability**: Symmetrise `P_out = 0.5 * (P_out + P_out.T)` each step. Use Joseph form.

#### Phase 2A checkpoint tests
- Scalar AR(1) with known params: compare filtered means to NumPy reference recursion.
- All-missing sequence: verify filter equals prediction throughout.
- Log-likelihood matches `scipy.stats.multivariate_normal.logpdf` step-by-step.
- Test `van_loan` against reference for near-zero `A`.
- Profile `vmap` over 100 individuals vs Python loop.

### 2B. LGSSM Model Variants (`src/jmseq/models/`)

#### Model config (`model_config.py`)

```python
@dataclass
class LGSSMConfig:
    name: str
    free_drift: bool      # A matrix (DRIFT): True=estimated, False=zero
    free_diffusion: bool  # G matrix (DIFFUSION): True=estimated, False=zero
    free_cint: bool       # b vector (CINT/slope): True=estimated, False=zero
    p: int                # state dimension
    d: int                # observation dimension
    n_tipred: int         # number of time-invariant predictors
```

Five variants:

| Name | free_drift | free_diffusion | free_cint |
|---|---|---|---|
| `model_lmm` | False | False | True |
| `model_nolmm` | True | True | False |
| `model_lmmdiff` | False | True | True |
| `model_lmmdrift` | True | False | True |
| `model_lmmdriftdiff` | True | True | True |

#### Parameters to sample (`lgssm.py`)

- `A`: if `free_drift` — parameterise diagonal as `-exp(log_neg_A_diag)` (negative = stable); off-diagonal free. If fixed: `A = zeros(p, p)`.
- `G`: if `free_diffusion` — lower-triangular Cholesky `L_G` with `constraint=lower_cholesky`. If fixed: `G = zeros(p, p)`.
- `b` (CINT): if `free_cint` — `Normal(0, 1)` vector of length `p`.
- `R`: diagonal `exp(log_sigma_obs)^2`, always free.
- `mu0`: initial state mean, length `p`.
- `T0` (`p × n_tipred`): TI predictor effect on initial state: `m0_i = mu0 + T0 @ ti_pred_i`.
- `L_P0`: Cholesky of initial state covariance.

#### Log-likelihood as NumPyro factor

The model calls the JAX Kalman filter to compute total log-likelihood over all individuals, then:
```python
total_ll = batched_kalman_filter(A, G, b, R, m0_batch, P0, Y, dt_seqs, obs_masks)
numpyro.factor("obs", total_ll)
```

This is the standard NumPyro pattern for models with custom likelihood computations.

#### MAP estimation (`inference/map_estimator.py`)

Matching R's `optimise=TRUE`:
- Use `numpyro.infer.autoguide.AutoDelta` + `SVI` + `ClippedAdam`, or
- Use `jax.value_and_grad` on potential energy + `jaxopt.LBFGSB` directly.
- NumPyro's `constraints` system handles parameter transformations automatically.

#### Phase 2B checkpoint tests
- Generate synthetic data from `model_lmm`; verify MAP `b` and `R` within 2 SE of truth.
- All five configs construct without error; fixed-zero params not in free parameter count.
- `A` eigenvalues have negative real parts after optimisation.
- Verify `T0` effect: individuals with different TI predictors have different initial state means.

---

## Phase 3 — Data Pipeline and Integration

### Data transforms (`src/jmseq/data/transforms.py`)

- `split_surv_long(surv_df, long_df, max_interval, ...) → (surv_intervals_df, long_intervals_df)`  
  Replicates R `split.SurvLong()`. Per individual: compute follow-up intervals, split those longer than `max_interval`, insert NA-biomarker rows for split intervals.  
  Use `pandas.groupby` + `apply` (correctness over vectorisation).

- `long_to_arrays(long_df, id_col, time_col, biomarker_cols, max_T) → (Y, dt_seqs, obs_masks)`  
  Converts longitudinal DataFrame to padded JAX arrays `(N, max_T, d)`, `(N, max_T)`, `(N, max_T)`.

### Pipeline orchestration

**`src/jmseq/pipeline/train.py`**

- `fit_lgssm_fold(long_df, config: LGSSMConfig) → LGSSMResult`  
  Equivalent to `ctstanfit_fold()`. Prepares arrays, fits LGSSM MAP.

- `run_kalman_fold(long_df, lgssm_result: LGSSMResult) → pd.DataFrame`  
  Equivalent to `kalmanwide()`. Returns DataFrame with `[id, tstart, state_1, ..., state_p]`.

- `fit_poisson_fold(surv_df, kalman_df, timeinvar_cols, biomarker_cols) → PoissonResult`  
  Equivalent to `fit_poissontsplit()`. Joins, assembles `X`, runs MAP Poisson GLM.

**`src/jmseq/pipeline/predict.py`**

- `predict_testdata(test_surv_df, kalman_df, poisson_result, landmark_time) → pd.DataFrame`  
  Replicates `predict_testdata()`.

- `tabulate_predictions(pred_df) → dict`  
  C-statistic (`lifelines.statistics.concordance_index`), log-score, observed vs predicted.

### Phase 3 checkpoint tests
- `test_split_surv_long`: 3-individual hand-crafted dataset, assert exact output match.
- `test_long_to_arrays`: verify padding, `dt`, `obs_mask`.
- End-to-end smoke test: 50 simulated individuals, full pipeline, C-statistic > 0.55.
- Cross-validation fold test: fit on train, predict on test, verify output shape.

---

## Implementation Order (Recommended)

1. `kalman/covariance.py` — Van Loan, tested in isolation
2. `kalman/filter.py` — scalar case, then multivariate + missing, then batched vmap
3. `models/poisson_glm.py` + `inference/map_estimator.py` — Phase 1 complete
4. `models/model_config.py` + `models/lgssm.py` — LGSSM definitions
5. `inference/map_estimator.py` applied to LGSSM — Phase 2 complete
6. `data/transforms.py` + `data/splits.py` — Phase 3 data layer
7. `pipeline/train.py` + `pipeline/predict.py` — Phase 3 complete
8. End-to-end validation on PBC dataset

Phases 1 and 2A can be developed in parallel (no dependencies between them).

---

## Numerical Considerations

- **Matrix exponential**: use `jax.scipy.linalg.expm` (Padé approximant, differentiable via JAX custom VJP). Do not use Taylor expansion or `matrix_power`.
- **Van Loan overflow**: for large `Δt` with large-magnitude `A`, `expm` of the auxiliary matrix can overflow. Clamp `Δt` to a maximum (e.g. 20 years) in `split_surv_long`.
- **Joseph form**: explicitly symmetrise `P_out = 0.5 * (P_out + P_out.T)` each Kalman step to prevent numerical drift.
- **Poisson QR prediction**: use `X_test @ R_ast_inverse.T` (not re-QR-decomposing) to stay in the same column space as training.
- **Gradient flow through expm**: if optimisation of `A` stalls, log-transform the eigenvalues: `A_diag = -exp(log_neg_a)` keeps gradients bounded.
- **Padding in batched filter**: group individuals by sequence length to reduce wasted compute from padding.
