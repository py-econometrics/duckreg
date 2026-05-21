import ibis
import numpy as np
import pandas as pd
import pytest

from duckreg.dbreg import DBDML, DBDoubleDemeaning, DBMundlak, DBRegression
from duckreg.estimators import DuckDML, DuckDoubleDemeaning, DuckMundlak, DuckRegression

pf = pytest.importorskip("pyfixest")


def _panel_df(seed=123, n_units=18, n_times=7):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_times)
    time = np.tile(np.arange(n_times), n_units)
    n = len(unit)
    f1 = unit % 3
    f2 = time % 4
    d = 0.4 * (unit % 2) + 0.25 * time + rng.normal(size=n)
    x = rng.normal(size=n)
    unit_fe = rng.normal(size=n_units)[unit]
    time_fe = rng.normal(size=n_times)[time]
    y = 1.0 + 1.7 * d - 0.6 * x + 0.3 * f1 - 0.2 * f2 + unit_fe + time_fe + rng.normal(scale=0.2, size=n)
    y2 = -0.5 - 0.8 * d + 0.4 * x + 0.1 * f1 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame(
        {
            "rowid": np.arange(n),
            "unit": unit,
            "time": time,
            "cluster": unit,
            "Y": y,
            "Y2": y2,
            "D": d,
            "X": x,
            "f1": f1.astype(float),
            "f2": f2.astype(float),
        }
    )


def _discrete_df(seed=321, n=500):
    rng = np.random.default_rng(seed)
    f1 = rng.integers(0, 5, size=n).astype(float)
    f2 = rng.integers(0, 4, size=n).astype(float)
    d = rng.integers(0, 3, size=n).astype(float)
    x = rng.integers(0, 2, size=n).astype(float)
    cluster = rng.integers(0, 25, size=n)
    y = 2.0 + 1.25 * d - 0.4 * f1 + 0.2 * f2 + 0.7 * x + rng.normal(scale=0.5, size=n)
    y2 = -1.0 - 0.75 * d + 0.3 * f1 + rng.normal(scale=0.4, size=n)
    return pd.DataFrame(
        {
            "rowid": np.arange(n),
            "cluster": cluster,
            "Y": y,
            "Y2": y2,
            "D": d,
            "X": x,
            "f1": f1,
            "f2": f2,
        }
    )


def _con(tmp_path, df, stem):
    con = ibis.duckdb.connect(tmp_path / f"{stem}.db")
    con.create_table("data", df, overwrite=True)
    return con


def _fit_old_new(cls_old, cls_new, kwargs):
    old = cls_old(**kwargs)
    new = cls_new(**kwargs)
    old.fit()
    new.fit()
    return old, new


@pytest.mark.parametrize(
    "formula, coef_names",
    [
        ("Y ~ D", ["Intercept", "D"]),
        ("Y ~ D + f1", ["Intercept", "D", "f1"]),
        ("Y2 ~ D + f1 + f2", ["Intercept", "D", "f1", "f2"]),
    ],
)
def test_db_regression_matches_duckreg_and_pyfixest(tmp_path, formula, coef_names):
    df = _discrete_df()
    con = _con(tmp_path, df, "ols")
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        formula=formula,
        cluster_col="cluster",
        seed=42,
        n_bootstraps=0,
    )
    old, new = _fit_old_new(DuckRegression, DBRegression, kwargs)
    np.testing.assert_allclose(new.point_estimate, old.point_estimate, rtol=1e-11, atol=1e-11)

    fx = pf.feols(formula, data=df, vcov="iid").coef().reindex(coef_names).to_numpy()
    np.testing.assert_allclose(new.point_estimate, fx, rtol=1e-11, atol=1e-11)


def test_db_regression_cluster_bootstrap_runs_without_duckdb_unnest(tmp_path):
    df = _discrete_df(n=180)
    con = _con(tmp_path, df, "boot")
    model = DBRegression(
        db_name=None,
        connection=con,
        table_name="data",
        formula="Y ~ D + f1",
        cluster_col="cluster",
        seed=7,
        n_bootstraps=8,
    )
    model.fit()
    assert model.vcov.shape == (3, 3)
    assert np.isfinite(model.vcov).all()
    np.testing.assert_allclose(model.vcov, model.vcov.T, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("treatment_vars", ["D", ["D", "X"]])
def test_db_dml_matches_duckreg_for_single_and_multi_treatment(tmp_path, treatment_vars):
    df = _discrete_df(n=400)
    con = _con(tmp_path, df, "dml")
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y",
        treatment_var=treatment_vars,
        discrete_covars=["f1", "f2"],
        seed=42,
        n_bootstraps=0,
    )
    old, new = _fit_old_new(DuckDML, DBDML, kwargs)
    np.testing.assert_allclose(new.point_estimate, old.point_estimate, rtol=1e-11, atol=1e-11)


def test_db_mundlak_matches_duckreg_and_pyfixest_explicit_design(tmp_path):
    df = _panel_df()
    con = _con(tmp_path, df, "mundlak")
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y",
        covariates=["D", "X"],
        seed=42,
        unit_col="unit",
        time_col="time",
        n_bootstraps=0,
    )
    old, new = _fit_old_new(DuckMundlak, DBMundlak, kwargs)
    np.testing.assert_allclose(new.point_estimate, old.point_estimate, rtol=1e-10, atol=1e-10)

    design = df.copy()
    for cov in ["D", "X"]:
        design[f"avg_{cov}_unit"] = design.groupby("unit")[cov].transform("mean")
        design[f"avg_{cov}_time"] = design.groupby("time")[cov].transform("mean")
    formula = "Y ~ D + X + avg_D_unit + avg_X_unit + avg_D_time + avg_X_time"
    fx = pf.feols(formula, data=design, vcov="iid").coef().to_numpy().reshape(-1, 1)
    np.testing.assert_allclose(new.point_estimate, fx, rtol=1e-10, atol=1e-10)


def test_db_double_demeaning_matches_duckreg_and_pyfixest_twfe(tmp_path):
    df = _panel_df()
    con = _con(tmp_path, df, "twfe")
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y",
        treatment_var="D",
        unit_col="unit",
        time_col="time",
        seed=42,
        n_bootstraps=0,
    )
    old, new = _fit_old_new(DuckDoubleDemeaning, DBDoubleDemeaning, kwargs)
    np.testing.assert_allclose(new.point_estimate, old.point_estimate, rtol=1e-10, atol=1e-10)

    fx_coef = pf.feols("Y ~ D | unit + time", data=df, vcov="iid").coef().loc["D"]
    np.testing.assert_allclose(new.point_estimate.flatten()[1], fx_coef, rtol=1e-10, atol=1e-10)
