"""Phase 3 tests: data transforms, train/test splits, and pipeline smoke test."""

import functools

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jmseq.data.transforms import split_surv_long, long_to_arrays
from jmseq.data.splits import trainsplit_surv, trainsplit_long
from jmseq.models.model_config import make_config
from jmseq.pipeline.train import fit_lgssm_fold, run_kalman_fold, fit_poisson_fold
from jmseq.pipeline.predict import predict_testdata, tabulate_predictions


# ---------------------------------------------------------------------------
# Fixtures: hand-crafted 3-individual dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset():
    """Three individuals, two biomarkers, irregular intervals."""
    surv_df = pd.DataFrame({
        "id":      [1, 2, 3],
        "Time.cens": [3.0, 2.5, 4.0],
        "event":   [1, 0, 1],
        "age":     [50, 60, 55],
    })
    long_df = pd.DataFrame({
        "id":    [1, 1, 1,   2, 2,     3, 3, 3, 3],
        "Time":  [0.0, 1.0, 2.5,  0.0, 1.5,  0.0, 1.0, 2.0, 3.0],
        "bili":  [1.0, 1.5, 2.0,  0.8, 1.0,  1.2, 1.4, 1.6, 1.8],
        "alb":   [4.0, 3.8, 3.5,  4.2, 4.0,  3.9, 3.7, 3.5, 3.3],
    })
    return surv_df, long_df


# ---------------------------------------------------------------------------
# test_split_surv_long
# ---------------------------------------------------------------------------

def test_split_surv_long_no_split(tiny_dataset):
    """With max_interval=10, no intervals should be split."""
    surv_df, long_df = tiny_dataset
    result = split_surv_long(
        surv_df, long_df, max_interval=10.0,
        timeinvar_surv=["age"], biomarkers=["bili", "alb"]
    )
    # Should have same number of rows as long_df (9 rows)
    assert len(result) == len(long_df), f"Expected 9 rows, got {len(result)}"
    assert set(result.columns) >= {"id", "Time", "tstop", "event", "age", "bili", "alb"}


def test_split_surv_long_splits_correctly(tiny_dataset):
    """With max_interval=0.5, long intervals get split; inserted rows have NaN biomarkers."""
    surv_df, long_df = tiny_dataset
    result = split_surv_long(
        surv_df, long_df, max_interval=0.5,
        timeinvar_surv=["age"], biomarkers=["bili", "alb"]
    )
    # All intervals should be <= 0.5 + small tolerance
    tobs = result["tstop"] - result["Time"]
    assert (tobs <= 0.55).all(), f"Interval longer than max_interval: {tobs.max():.3f}"
    # Total rows > 9 because some intervals are split
    assert len(result) > len(long_df), "Expected more rows after splitting"


def test_split_surv_long_event_only_on_last(tiny_dataset):
    """Event=1 should appear only on the last interval per individual."""
    surv_df, long_df = tiny_dataset
    result = split_surv_long(
        surv_df, long_df, max_interval=0.5,
        timeinvar_surv=["age"], biomarkers=["bili", "alb"]
    )
    for id_val, grp in result.groupby("id"):
        # At most one row should have event=1
        assert grp["event"].sum() <= 1, f"id={id_val} has multiple event=1 rows"
        if grp["event"].sum() == 1:
            # That row should be the last one (highest tstop)
            assert grp.loc[grp["event"] == 1, "tstop"].values[0] == grp["tstop"].max()


def test_split_surv_long_inserted_rows_are_nan(tiny_dataset):
    """Inserted (split) rows should have NaN biomarkers; original rows should not."""
    surv_df, long_df = tiny_dataset
    result = split_surv_long(
        surv_df, long_df, max_interval=0.5,
        timeinvar_surv=["age"], biomarkers=["bili", "alb"]
    )
    # Extra rows (beyond original 9) must have NaN biomarkers
    n_extra = len(result) - len(long_df)
    if n_extra > 0:
        # The extra rows are the inserted sub-intervals; they have NaN biomarkers
        # Check indirectly: number of NaN rows equals number of extra rows
        nan_rows = result[result["bili"].isna()]
        assert len(nan_rows) == n_extra, (
            f"Expected {n_extra} NaN rows, got {len(nan_rows)}"
        )


# ---------------------------------------------------------------------------
# test_long_to_arrays
# ---------------------------------------------------------------------------

def test_long_to_arrays_shapes(tiny_dataset):
    """Y, dt_batch, mask_batch should have correct shapes."""
    surv_df, long_df = tiny_dataset
    result = split_surv_long(
        surv_df, long_df, max_interval=10.0,
        timeinvar_surv=["age"], biomarkers=["bili", "alb"]
    )
    Y, dt, mask, ids, times_list = long_to_arrays(
        result, "id", "Time", ["bili", "alb"]
    )
    N = result["id"].nunique()
    max_T = result.groupby("id").size().max()
    assert Y.shape    == (N, max_T, 2),  f"Y shape {Y.shape}"
    assert dt.shape   == (N, max_T),     f"dt shape {dt.shape}"
    assert mask.shape == (N, max_T),     f"mask shape {mask.shape}"
    assert len(ids)   == N


def test_long_to_arrays_dt_correct(tiny_dataset):
    """dt[t=0] should equal Time[0]; dt[t>0] should equal Time[t] - Time[t-1]."""
    surv_df, long_df = tiny_dataset
    Y, dt, mask, ids, times_list = long_to_arrays(
        long_df, "id", "Time", ["bili", "alb"]
    )
    for i, times in enumerate(times_list):
        # First dt = Time[0]
        np.testing.assert_allclose(float(dt[i, 0]), float(times[0]), rtol=1e-5)
        # Subsequent dts
        for t in range(1, len(times)):
            expected = float(times[t]) - float(times[t - 1])
            np.testing.assert_allclose(float(dt[i, t]), expected, rtol=1e-5)


def test_long_to_arrays_obs_mask_nan(tiny_dataset):
    """Rows with NaN biomarkers should have obs_mask=False."""
    surv_df, long_df = tiny_dataset
    split = split_surv_long(
        surv_df, long_df, max_interval=0.5,
        timeinvar_surv=["age"], biomarkers=["bili", "alb"]
    )
    Y, dt, mask, ids, times_list = long_to_arrays(
        split, "id", "Time", ["bili", "alb"]
    )
    # Verify that NaN rows in the DataFrame correspond to mask=False
    for i, (id_val, times) in enumerate(zip(ids, times_list)):
        grp = split[split["id"] == id_val].sort_values("Time").reset_index(drop=True)
        for t in range(len(times)):
            is_nan = grp.loc[t, ["bili", "alb"]].isna().all()
            assert bool(mask[i, t]) != is_nan, (
                f"id={id_val}, t={t}: nan={is_nan} but mask={mask[i,t]}"
            )


# ---------------------------------------------------------------------------
# test_trainsplit_*
# ---------------------------------------------------------------------------

def test_trainsplit_surv_drops_test_post_landmark(tiny_dataset):
    """trainsplit_surv removes test-fold intervals with tstop > landmark."""
    surv_df, long_df = tiny_dataset
    split = split_surv_long(
        surv_df, long_df, max_interval=1.0,
        timeinvar_surv=["age"], biomarkers=["bili", "alb"]
    )
    ids_test = [1]
    landmark = 1.5
    train = trainsplit_surv(ids_test, split, landmark_time=landmark)
    # Individual 1, post-landmark intervals should be gone
    id1_post = train[(train["id"] == 1) & (train["tstop"] > landmark)]
    assert len(id1_post) == 0, "Test-fold post-landmark intervals not removed"
    # Individual 2 (not in test) should be untouched
    assert len(train[train["id"] == 2]) == len(split[split["id"] == 2])


def test_trainsplit_long_blanks_post_landmark(tiny_dataset):
    """trainsplit_long sets biomarkers to NaN for test individuals post-landmark."""
    surv_df, long_df = tiny_dataset
    split = split_surv_long(
        surv_df, long_df, max_interval=1.0,
        timeinvar_surv=["age"], biomarkers=["bili", "alb"]
    )
    ids_test = [1]
    landmark = 1.0
    train_long = trainsplit_long(ids_test, split, landmark_time=landmark, biomarkers=["bili", "alb"])
    # Individual 1 rows with Time > 1.0 should be NaN
    id1_post = train_long[(train_long["id"] == 1) & (train_long["Time"] > landmark)]
    assert id1_post["bili"].isna().all(), "bili should be NaN post-landmark for id=1"
    # Rows at or before landmark should be unchanged
    id1_pre = train_long[(train_long["id"] == 1) & (train_long["Time"] <= landmark)]
    assert not id1_pre["bili"].isna().any(), "bili should not be NaN pre-landmark"


# ---------------------------------------------------------------------------
# Smoke test: end-to-end pipeline on synthetic data
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(N=30, seed=7):
    """Simulate a simple dataset for smoke testing."""
    rng = np.random.default_rng(seed)
    p = 2
    landmark = 2.0
    max_obs_time = 6.0
    biomarkers = ["s1", "s2"]
    timeinvar = ["age"]

    # True parameters (model_lmm: A=0, G=0, random walk mean b)
    b_true = np.array([0.2, -0.1])
    sigma_obs = np.array([0.3, 0.2])
    beta0_true = -3.0
    beta_true  = np.array([0.0, 0.3, -0.2, 0.4, -0.3])  # tstart, age, s1, s2

    obs_times_per_indiv = [
        sorted(rng.uniform(0, min(max_obs_time, cens), size=rng.integers(4, 8)))
        for cens in rng.uniform(3, max_obs_time, N)
    ]
    cens_times = [t[-1] + rng.uniform(0.1, 0.5) for t in obs_times_per_indiv]
    ages = rng.normal(55, 10, N).astype(np.float32)

    long_rows = []
    surv_rows = []
    for i in range(N):
        times = obs_times_per_indiv[i]
        cens = min(cens_times[i], max_obs_time)
        # Simulate observations ~ N(b_true * t, sigma_obs^2)
        for t in times:
            mean = b_true * t
            y = rng.normal(mean, sigma_obs).astype(np.float32)
            long_rows.append({"id": i + 1, "Time": float(t), "s1": y[0], "s2": y[1], "age": ages[i]})
        # Simulate event time (simplified: event if last obs b1 > threshold)
        last_s1 = long_rows[-1]["s1"]
        event = int(last_s1 > 0.3)
        surv_rows.append({"id": i + 1, "Time.cens": cens, "event": event, "age": float(ages[i])})

    long_df = pd.DataFrame(long_rows)
    surv_df  = pd.DataFrame(surv_rows)
    return surv_df, long_df, biomarkers, timeinvar, landmark


@pytest.mark.slow
def test_pipeline_smoke():
    """End-to-end pipeline: fit LGSSM + Poisson, predict on test fold."""
    surv_df, long_df, biomarkers, timeinvar, landmark = _make_synthetic_dataset(N=30)

    # Preprocess
    split = split_surv_long(
        surv_df, long_df, max_interval=0.5,
        timeinvar_surv=timeinvar, biomarkers=biomarkers
    )
    dataSurv = split[["id", "Time", "tstop", "event"] + timeinvar]
    dataLong = split[["id", "Time", "tstop"] + timeinvar + biomarkers]

    # Cross-validation: 1 fold for smoke test
    ids_all = surv_df[surv_df["Time.cens"] > landmark]["id"].tolist()
    ids_test = ids_all[:8]

    train_surv = trainsplit_surv(ids_test, dataSurv, landmark)
    train_long = trainsplit_long(ids_test, dataLong, landmark, biomarkers)

    # Fit LGSSM
    config = make_config("model_lmm", p=len(biomarkers))
    lgssm_result = fit_lgssm_fold(
        train_long, config,
        biomarker_cols=biomarkers,
        n_steps=1_000,
        lr=0.05,
    )
    assert lgssm_result.params is not None

    # Run Kalman filter on all data (including test fold)
    kalman_df = run_kalman_fold(dataLong, lgssm_result)
    assert set(kalman_df.columns) >= {"id", "tstart"} | set(biomarkers)
    assert len(kalman_df) == len(dataLong)

    # Fit Poisson GLM
    poisson_result = fit_poisson_fold(
        train_surv, kalman_df[kalman_df["id"].isin(train_surv["id"].unique())],
        timeinvar, biomarkers,
        n_steps=2_000,
    )
    assert len(poisson_result.beta) == len(poisson_result.feature_names)

    # Predict on test fold
    kal_test = kalman_df[
        kalman_df["id"].isin(ids_test) & (kalman_df["tstart"] > landmark)
    ]
    pred_df = predict_testdata(
        dataSurv, kal_test, poisson_result,
        timeinvar, biomarkers, landmark_time=landmark
    )
    assert "p_event" in pred_df.columns
    assert "survprob" in pred_df.columns
    assert pred_df["p_event"].between(0, 1).all()

    # Evaluate
    stats = tabulate_predictions(pred_df)
    assert "C-statistic" in stats
    assert "Log score" in stats
    # Sanity: observed events should be a non-negative integer
    assert stats["Observed"] >= 0
