"""
Cross-validation runner for the jmseq pipeline.

cross_validate — k-fold CV with configurable models, landmark time, and fold seeds.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jmseq.models.model_config import LGSSMConfig, make_config
from jmseq.data.splits import trainsplit_surv, trainsplit_long
from jmseq.pipeline.train import fit_lgssm_fold, run_kalman_fold, fit_poisson_fold
from jmseq.pipeline.predict import predict_testdata, tabulate_predictions


@dataclass
class CVResult:
    """Output of cross_validate()."""
    model_name: str
    fold_stats: list[dict]         # one dict per fold (Observed, Predicted, etc.)
    fold_pred:  list[pd.DataFrame] # per-fold prediction DataFrames
    pooled:     dict               # tabulate_predictions on concatenated folds
    fold_times: list[float]        # wall time per fold (seconds)

    @property
    def c_statistic(self) -> float:
        return self.pooled["C-statistic"]

    @property
    def log_score(self) -> float:
        return self.pooled["Log score"]


def cross_validate(
    dataSurv: pd.DataFrame,
    dataLong: pd.DataFrame,
    biomarker_cols: list[str],
    timeinvar_surv: list[str],
    configs: list[LGSSMConfig] | dict[str, LGSSMConfig] | None = None,
    model_names: list[str] | None = None,
    p: int = 2,
    landmark_time: float = 5.0,
    n_folds: int = 4,
    random_seed: int = 1234,
    id_col: str = "id",
    lgssm_steps: int = 3_000,
    lgssm_lr: float = 0.02,
    poisson_steps: int = 3_000,
    poisson_lr: float = 0.05,
    verbose: bool = True,
) -> dict[str, CVResult]:
    """Run k-fold cross-validation for one or more LGSSM + Poisson GLM models.

    Port of the loop in the R vignette (jmseq.Rmd).

    Parameters
    ----------
    dataSurv        : interval-split survival data [id, Time, tstop, event, *timeinvar].
    dataLong        : interval-split longitudinal data [id, Time, *biomarkers].
    biomarker_cols  : biomarker column names.
    timeinvar_surv  : time-invariant covariate names.
    configs         : LGSSMConfig objects — list or {name: config} dict.
                      Mutually exclusive with model_names.
    model_names     : list of model name strings (e.g. ["model_lmm", "model_lmmdriftdiff"]).
                      Preferred when calling from R via reticulate to avoid passing Python
                      objects across the language boundary.  Mutually exclusive with configs.
    p               : state dimension (number of biomarkers).  Used only with model_names.
    landmark_time   : start of prediction window (years).
    n_folds         : number of CV folds.
    random_seed     : for fold assignment.
    id_col          : individual ID column name.
    lgssm_steps     : SVI steps for LGSSM MAP.
    lgssm_lr        : learning rate for LGSSM MAP.
    poisson_steps   : SVI steps for Poisson GLM MAP.
    poisson_lr      : learning rate for Poisson GLM MAP.
    verbose         : print fold-by-fold progress.

    Returns
    -------
    Dict mapping model name → CVResult.
    """
    # Normalise configs to {name: config}
    if model_names is not None:
        configs = {name: make_config(name, p=p) for name in model_names}
    elif isinstance(configs, list):
        configs = {c.name: c for c in configs}
    elif configs is None:
        raise ValueError("Either configs or model_names must be provided")

    # Individuals at risk after landmark: keep those whose last tstop > landmark_time.
    # Use Time.cens column when present (raw survival data); otherwise derive from
    # max tstop across all interval rows for each individual (split survival data).
    if "Time.cens" in dataSurv.columns:
        first_row = dataSurv.drop_duplicates(subset=[id_col])
        ids_at_risk = first_row.loc[
            first_row["Time.cens"] > landmark_time, id_col
        ].to_numpy()
    else:
        max_tstop   = dataSurv.groupby(id_col)["tstop"].max()
        ids_at_risk = max_tstop.index[max_tstop > landmark_time].to_numpy()

    rng = np.random.default_rng(random_seed)
    ids_perm = rng.permutation(ids_at_risk)
    folds    = np.array_split(ids_perm, n_folds)

    if verbose:
        print(f"{n_folds}-fold CV, {len(ids_at_risk)} individuals at risk after landmark {landmark_time}")

    results = {name: CVResult(name, [], [], {}, []) for name in configs}
    all_pred = {name: [] for name in configs}

    for fold_idx, ids_test in enumerate(folds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx+1}/{n_folds}  (test n={len(ids_test)})")
            print('='*60)

        train_surv = trainsplit_surv(ids_test, dataSurv, landmark_time)
        train_long = trainsplit_long(ids_test, dataLong, landmark_time, biomarker_cols)
        train_ids  = set(train_surv[id_col].unique())

        for model_name, config in configs.items():
            if verbose:
                print(f"\n  Model: {model_name}", flush=True)

            t_fold = time.time()

            # Fit LGSSM
            t0 = time.time()
            lgssm_result = fit_lgssm_fold(
                train_long, config,
                id_col=id_col,
                biomarker_cols=biomarker_cols,
                n_steps=lgssm_steps,
                lr=lgssm_lr,
            )
            if verbose:
                print(f"    LGSSM MAP: {time.time()-t0:.1f}s", flush=True)

            # Kalman filter (all individuals)
            t0 = time.time()
            kalman_df = run_kalman_fold(dataLong, lgssm_result)
            if verbose:
                print(f"    Kalman filter: {time.time()-t0:.1f}s  ({len(kalman_df)} rows)", flush=True)

            # Poisson GLM on training individuals
            t0 = time.time()
            poisson_result = fit_poisson_fold(
                train_surv,
                kalman_df[kalman_df[id_col].isin(train_ids)],
                timeinvar_surv,
                biomarker_cols,
                id_col=id_col,
                n_steps=poisson_steps,
                lr=poisson_lr,
            )
            if verbose:
                print(f"    Poisson GLM: {time.time()-t0:.1f}s", flush=True)

            # Predict on test fold (post-landmark intervals only)
            kal_test = kalman_df[
                kalman_df[id_col].isin(ids_test) & (kalman_df["tstart"] > landmark_time)
            ]
            pred_df = predict_testdata(
                dataSurv, kal_test, poisson_result,
                timeinvar_surv, biomarker_cols,
                landmark_time=landmark_time,
                id_col=id_col,
            )
            stats = tabulate_predictions(pred_df)

            if verbose:
                print(
                    f"    Observed={stats['Observed']}  Predicted={stats['Predicted']:.2f}"
                    f"  Pyrs={stats['Person-years']:.1f}"
                    f"  LL={stats['Log score']:.2f}"
                    f"  C={stats['C-statistic']:.4f}",
                    flush=True,
                )

            results[model_name].fold_stats.append(stats)
            results[model_name].fold_pred.append(pred_df)
            results[model_name].fold_times.append(time.time() - t_fold)
            all_pred[model_name].append(pred_df)

    # Pooled summary
    for model_name in configs:
        combined = pd.concat(all_pred[model_name], ignore_index=True)
        results[model_name].pooled = tabulate_predictions(combined)

    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY — pooled cross-validated performance")
        print('='*60)
        for model_name, res in results.items():
            s = res.pooled
            print(f"\n{model_name}:")
            print(f"  Observed events:   {s['Observed']}")
            print(f"  Predicted events:  {s['Predicted']:.2f}")
            print(f"  Person-years:      {s['Person-years']:.1f}")
            print(f"  Log-score:         {s['Log score']:.4f}")
            print(f"  C-statistic:       {s['C-statistic']:.4f}")

    return results
