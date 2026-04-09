# jmseq

Joint modelling of longitudinal biomarker trajectories and time-to-event outcomes.

`jmseq` fits a continuous-time linear Gaussian state-space model (LGSSM) to repeated
biomarker measurements using a Kalman filter, then links the filtered state trajectories
to a Poisson hazard model via a piecewise-constant intensity representation.  Parameter
estimation uses MAP inference (NumPyro SVI with an `AutoDelta` guide).

This is a Python port of the [jmseq R package](https://github.com/pmckeigue/jmseq),
replacing Stan + ctsem with [NumPyro](https://num.pyro.ai) + [JAX](https://github.com/google/jax).
A vignette using the Mayo Clinic primary biliary cirrhosis (PBC) dataset is available
[here](https://htmlpreview.github.io/?https://github.com/pmckeigue/jmseq/blob/main/jmseq/vignettes/xvalid.html).

---

## Getting started

### Installation

```bash
pip install -e ".[dev]"   # from the repo root
```

Dependencies: `jax`, `numpyro`, `optax`, `numpy`, `pandas`, `scipy`, `scikit-learn`.

### Minimal example

```python
import pandas as pd
from jmseq import make_config, split_surv_long, cross_validate

# Load data (one row per individual for survival; one row per visit for longitudinal)
dataSurv = pd.read_csv("pbc_surv.csv")   # columns: id, Time.cens, event, sex, agebaseline, trt_binary
dataLong = pd.read_csv("pbc_long.csv")   # columns: id, Time, logBili, albumin

biomarkers     = ["logBili", "albumin"]
timeinvar_surv = ["sex", "agebaseline", "trt_binary"]

# 1. Split follow-up into intervals of at most 0.25 years
from jmseq import split_surv_long
split = split_surv_long(
    dataSurv[["id", "Time.cens", "event"] + timeinvar_surv],
    dataLong[["id", "Time"] + biomarkers],
    max_interval=0.25,
    timeinvar_surv=timeinvar_surv,
    biomarkers=biomarkers,
)
dataSurv_split = split[["id", "Time", "tstop", "event"] + timeinvar_surv]
dataLong_split = split[["id", "Time", "tstop"] + biomarkers]

# 2. Choose a model variant and run 4-fold cross-validation
config  = make_config("model_lmmdriftdiff", p=2)
results = cross_validate(
    dataSurv_split, dataLong_split,
    configs=[config],
    biomarker_cols=biomarkers,
    timeinvar_surv=timeinvar_surv,
    landmark_time=5.0,
    n_folds=4,
)

res = results["model_lmmdriftdiff"]
print(f"C-statistic: {res.c_statistic:.3f}")
print(f"Log-score:   {res.log_score:.1f}")
```

---

## Model variants

Five LGSSM variants are pre-defined, matching the ctsem models in the R package:

| Name | Drift A | Diffusion G | Intercept b | Description |
|---|---|---|---|---|
| `model_lmm` | fixed 0 | fixed 0 | free | Linear mixed model (random slope only) |
| `model_nolmm` | free | free | fixed 0 | Autoregressive + diffusion, no slope |
| `model_lmmdiff` | fixed 0 | free | free | LMM + Wiener diffusion |
| `model_lmmdrift` | free | fixed 0 | free | LMM + autoregressive drift |
| `model_lmmdriftdiff` | free | free | free | Full SDE model |

Select a variant with `make_config(name, p)` where `p` is the number of biomarkers.

---

## Public API

### Configuration

| Symbol | Description |
|---|---|
| `make_config(name, p, n_tipred=0)` | Return `LGSSMConfig` for the named variant |
| `LGSSMConfig` | Frozen dataclass: `name`, `free_drift`, `free_diffusion`, `free_cint`, `p`, `n_tipred` |
| `VARIANTS` | Dict of the five pre-defined `LGSSMConfig` objects (p=2) |

### Data preparation

| Symbol | Description |
|---|---|
| `split_surv_long(dataSurv, dataLong, max_interval, timeinvar_surv, biomarkers)` | Split follow-up intervals; returns combined DataFrame |
| `trainsplit_surv(ids_test, dataSurv, landmark_time)` | Restrict survival data to training fold |
| `trainsplit_long(ids_test, dataLong, landmark_time, biomarker_cols)` | Restrict longitudinal data; blank post-landmark observations for test individuals |
| `long_to_arrays(long_df, id_col, time_col, biomarker_cols)` | Convert longitudinal DataFrame to padded JAX arrays `(Y, dt_batch, mask_batch, ids, times_list)` |
| `validate_surv(df, timeinvar_surv)` | Raise `ValueError` if required survival columns are missing |
| `validate_long(df, biomarker_cols)` | Raise `ValueError` if required longitudinal columns are missing |

### Training

| Symbol | Description |
|---|---|
| `fit_lgssm_fold(long_df, config, biomarker_cols, n_steps, lr)` → `LGSSMResult` | Fit LGSSM MAP to training longitudinal data |
| `run_kalman_fold(long_df, lgssm_result)` → `pd.DataFrame` | Run Kalman filter; return filtered state means for all individuals |
| `fit_poisson_fold(dataSurv, kalman_df, timeinvar_surv, biomarker_cols, n_steps, lr)` → `PoissonResult` | Fit Poisson GLM MAP using Kalman-filtered states |
| `LGSSMResult` | Dataclass: `config`, `params`, `biomarker_cols`, `id_col`, `time_col` |
| `PoissonResult` | Dataclass: `beta0`, `beta`, `feature_names`, `Q_ast`, `R_ast_inv` |

### Prediction

| Symbol | Description |
|---|---|
| `predict_testdata(dataSurv, kalman_df, poisson_result, timeinvar_surv, biomarker_cols, landmark_time)` → `pd.DataFrame` | Predict event probabilities for test-fold individuals |
| `tabulate_predictions(pred_df)` → `dict` | Summary stats: Observed, Predicted, Person-years, Log score, C-statistic |

### Cross-validation

| Symbol | Description |
|---|---|
| `cross_validate(dataSurv, dataLong, configs, biomarker_cols, timeinvar_surv, ...)` → `dict[str, CVResult]` | Run k-fold cross-validation for one or more model variants |
| `CVResult` | Dataclass: `model_name`, `fold_stats`, `fold_pred`, `pooled`, `fold_times`; properties `c_statistic`, `log_score` |

### Metrics

| Symbol | Description |
|---|---|
| `log_score(events, p_event)` → `float` | Sum of per-interval Bernoulli log-probabilities |
| `c_statistic(events, p_event)` → `float` | ROC AUC over all person-time intervals |
| `calibration_table(events, p_event, n_groups=10)` → `pd.DataFrame` | Observed vs expected events by decile of predicted risk |

---

## Statistical model

**Longitudinal submodel** — continuous-time SDE discretised over observed intervals:

```
dX(t) = (A X(t) + b) dt + G dW(t)
y_t   = x_t + v_t,   v_t ~ N(0, R)
```

Discrete-time transition matrices `A_d(Δt)` and process noise covariance `Q_d(Δt)`
are computed via the Van Loan method (matrix exponential of a 2p × 2p auxiliary
matrix).  The Kalman filter is run with `jax.lax.scan` over time steps and
`jax.vmap` over individuals.

**Survival submodel** — piecewise-constant Poisson hazard:

```
log λ_i(t) = β₀ + β_t · t + β_x · x_i(t) + β_z · z_i
```

where `x_i(t)` are Kalman-filtered biomarker states and `z_i` are time-invariant
covariates.  The model is fitted as a Poisson GLM with log(tobs) offset using
QR decomposition for numerical stability.

**Performance metrics** are computed over all person-time intervals from the
landmark time to exit.  Both the log-score and the C-statistic treat each interval
as an independent observation with a binary outcome.
