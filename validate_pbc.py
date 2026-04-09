"""
Validation of the Python jmseq pipeline on the PBC dataset.

Mirrors the R vignette jmseq.Rmd:
  - 4-fold cross-validation
  - Two models: model_lmm and model_lmmdriftdiff
  - Landmark time 5 years, follow-up to 15 years
  - Predictive performance: C-statistic, log-score, observed/predicted events

Usage:
  python3 -u validate_pbc.py [--folds N] [--models model_lmm,model_lmmdriftdiff]
"""

import sys, time, argparse
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from jmseq.data.transforms import split_surv_long
from jmseq.data.splits import trainsplit_surv, trainsplit_long
from jmseq.models.model_config import make_config
from jmseq.pipeline.train import fit_lgssm_fold, run_kalman_fold, fit_poisson_fold
from jmseq.pipeline.predict import predict_testdata, tabulate_predictions

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--folds",  type=int, default=4)
parser.add_argument("--models", type=str, default="model_lmm,model_lmmdriftdiff")
parser.add_argument("--lgssm_steps",   type=int, default=3_000)
parser.add_argument("--poisson_steps", type=int, default=3_000)
args = parser.parse_args()

model_names = args.models.split(",")
nfolds      = args.folds

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
surv_raw = pd.read_csv("pbc_surv.csv")
long_raw = pd.read_csv("pbc_long.csv")

biomarkers     = ["logBili", "albumin"]
timeinvar_surv = ["sex", "agebaseline", "trt_binary"]

print(f"dataSurv: {len(surv_raw)} individuals,  events: {surv_raw['event'].sum()}")
print(f"dataLong: {len(long_raw)} observations")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
print("\nSplitting intervals (max 0.25 yr)...", flush=True)
t0 = time.time()

surv_input = surv_raw[["id", "Time.cens", "event"] + timeinvar_surv].copy()
long_input = long_raw[["id", "Time"] + biomarkers].copy()

split = split_surv_long(
    surv_input, long_input,
    max_interval=0.25,
    timeinvar_surv=timeinvar_surv,
    biomarkers=biomarkers,
)
print(f"  Done in {time.time()-t0:.1f}s: {len(split)} rows", flush=True)

dataSurv = split[["id", "Time", "tstop", "event"] + timeinvar_surv].copy()
dataLong = split[["id", "Time", "tstop"] + biomarkers].copy()

print(f"  max_T per individual: {dataLong.groupby('id').size().max()}", flush=True)
print(f"  mean obs per individual: {dataLong.groupby('id').size().mean():.1f}", flush=True)

# ---------------------------------------------------------------------------
# Cross-validation setup
# ---------------------------------------------------------------------------
landmark_time = 5.0

np.random.seed(1234)
ids_at_risk  = surv_raw.loc[surv_raw["Time.cens"] > landmark_time, "id"].unique()
ids_permuted = np.random.permutation(ids_at_risk)
fold_labels  = np.array_split(ids_permuted, nfolds)

print(f"\n{nfolds}-fold CV, {len(ids_at_risk)} individuals at risk after landmark {landmark_time}",
      flush=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
models = {name: make_config(name, p=2) for name in model_names}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_pred = {name: [] for name in model_names}

for fold_idx in range(nfolds):
    ids_test   = fold_labels[fold_idx]
    train_surv = trainsplit_surv(ids_test, dataSurv, landmark_time)
    train_long = trainsplit_long(ids_test, dataLong, landmark_time, biomarkers)

    print(f"\n{'='*60}", flush=True)
    print(f"Fold {fold_idx+1}/{nfolds}  (test n={len(ids_test)})", flush=True)
    print('='*60, flush=True)

    for model_name, config in models.items():
        print(f"\n  Model: {model_name}", flush=True)

        # --- Fit LGSSM ---
        t0 = time.time()
        lgssm_result = fit_lgssm_fold(
            train_long, config,
            biomarker_cols=biomarkers,
            n_steps=args.lgssm_steps,
            lr=0.02,
        )
        elapsed = time.time() - t0
        print(f"    LGSSM MAP: {elapsed:.1f}s", flush=True)

        p = lgssm_result.params
        print(f"    b:          {np.array(p.get('b', np.zeros(2)))}", flush=True)
        print(f"    sigma_obs:  {np.exp(np.array(p['log_sigma_obs']))}", flush=True)
        if "log_neg_a_diag" in p:
            print(f"    A diag:     {-np.exp(np.array(p['log_neg_a_diag']))}", flush=True)

        # --- Kalman filter ---
        t0 = time.time()
        kalman_df = run_kalman_fold(dataLong, lgssm_result)
        print(f"    Kalman filter: {time.time()-t0:.1f}s  ({len(kalman_df)} rows)", flush=True)

        # --- Poisson GLM ---
        t0 = time.time()
        train_ids = set(train_surv["id"].unique())
        poisson_result = fit_poisson_fold(
            train_surv,
            kalman_df[kalman_df["id"].isin(train_ids)],
            timeinvar_surv,
            biomarkers,
            n_steps=args.poisson_steps,
            lr=0.05,
        )
        print(f"    Poisson GLM: {time.time()-t0:.1f}s", flush=True)
        for fname, bval in zip(["beta0"] + poisson_result.feature_names,
                               [poisson_result.beta0] + list(poisson_result.beta)):
            print(f"      {fname:22s}: {bval:+.4f}", flush=True)

        # --- Predict on test fold ---
        kal_test = kalman_df[
            kalman_df["id"].isin(ids_test) & (kalman_df["tstart"] > landmark_time)
        ]
        pred_df = predict_testdata(
            dataSurv, kal_test, poisson_result,
            timeinvar_surv, biomarkers,
            landmark_time=landmark_time,
        )
        stats = tabulate_predictions(pred_df)
        print(f"    Observed={stats['Observed']}  Predicted={stats['Predicted']:.2f}"
              f"  Pyrs={stats['Person-years']:.1f}"
              f"  LL={stats['Log score']:.2f}"
              f"  C={stats['C-statistic']:.4f}", flush=True)

        all_pred[model_name].append(pred_df)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}", flush=True)
print("SUMMARY — pooled cross-validated performance", flush=True)
print('='*60, flush=True)
for model_name in model_names:
    combined = pd.concat(all_pred[model_name], ignore_index=True)
    s = tabulate_predictions(combined)
    print(f"\n{model_name}:", flush=True)
    print(f"  Observed events:   {s['Observed']}", flush=True)
    print(f"  Predicted events:  {s['Predicted']:.2f}", flush=True)
    print(f"  Person-years:      {s['Person-years']:.1f}", flush=True)
    print(f"  Log-score:         {s['Log score']:.4f}", flush=True)
    print(f"  C-statistic:       {s['C-statistic']:.4f}", flush=True)
