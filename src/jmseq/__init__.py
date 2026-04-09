"""
jmseq — Joint Models for Longitudinal and Survival data.

Python port of the jmseq R package, replacing Stan + ctsem with
NumPyro + JAX.

Quick-start
-----------
>>> from jmseq import make_config, split_surv_long, cross_validate
>>> config = make_config("model_lmm", p=2)
>>> split = split_surv_long(dataSurv, dataLong, max_interval=0.25,
...                         timeinvar_surv=["sex"], biomarkers=["logBili"])
>>> results = cross_validate(dataSurv_split, dataLong_split,
...                          configs=[config], biomarker_cols=["logBili"],
...                          timeinvar_surv=["sex"])
"""

# Model configuration
from jmseq.models.model_config import LGSSMConfig, make_config, VARIANTS

# Data transforms
from jmseq.data.transforms import split_surv_long, long_to_arrays
from jmseq.data.splits import trainsplit_surv, trainsplit_long
from jmseq.data.schema import validate_surv, validate_long

# Training pipeline
from jmseq.pipeline.train import (
    fit_lgssm_fold,
    run_kalman_fold,
    fit_poisson_fold,
    LGSSMResult,
    PoissonResult,
)

# Prediction
from jmseq.pipeline.predict import predict_testdata, tabulate_predictions

# Cross-validation
from jmseq.pipeline.cv import cross_validate, CVResult

# Metrics
from jmseq.utils.metrics import log_score, c_statistic, calibration_table

__all__ = [
    # config
    "LGSSMConfig",
    "make_config",
    "VARIANTS",
    # data
    "split_surv_long",
    "long_to_arrays",
    "trainsplit_surv",
    "trainsplit_long",
    "validate_surv",
    "validate_long",
    # training
    "fit_lgssm_fold",
    "run_kalman_fold",
    "fit_poisson_fold",
    "LGSSMResult",
    "PoissonResult",
    # prediction
    "predict_testdata",
    "tabulate_predictions",
    # CV
    "cross_validate",
    "CVResult",
    # metrics
    "log_score",
    "c_statistic",
    "calibration_table",
]
