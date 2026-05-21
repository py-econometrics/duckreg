import ibis
import numpy as np
import pandas as pd

import duckreg
from duckreg.dbreg import (
    DBLogisticRegression,
    DBMultinomialLogisticRegression,
    DBMundlakEventStudy,
    DBPoissonMultinomialRegression,
    DBPoissonRegression,
)
from duckreg.estimators import (
    DuckLogisticRegression,
    DuckMultinomialLogisticRegression,
    DuckMundlakEventStudy,
    DuckPoissonMultinomialRegression,
    DuckPoissonRegression,
)


def _make_con(tmp_path, df, table="data"):
    con = ibis.duckdb.connect(tmp_path / "dbreg_glm_event.db")
    con.create_table(table, df, overwrite=True)
    return con


def test_db_glm_exports():
    assert duckreg.DBLogisticRegression is DBLogisticRegression
    assert duckreg.DBPoissonRegression is DBPoissonRegression
    assert duckreg.DBMultinomialLogisticRegression is DBMultinomialLogisticRegression
    assert duckreg.DBPoissonMultinomialRegression is DBPoissonMultinomialRegression
    assert duckreg.DBMundlakEventStudy is DBMundlakEventStudy


def test_db_logistic_matches_duck_logistic(tmp_path):
    rng = np.random.default_rng(123)
    n = 5000
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 3, n)
    eta = -0.7 + 0.45 * x1 - 0.35 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    con = _make_con(tmp_path, pd.DataFrame({"y": y, "x1": x1, "x2": x2}))
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=123,
        method="irls",
        n_bootstraps=0,
    )
    old = DuckLogisticRegression(**kwargs)
    new = DBLogisticRegression(**kwargs)
    old.fit()
    old.fit_vcov()
    new.fit()
    new.fit_vcov()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)
    np.testing.assert_allclose(new.vcov, old.vcov)


def test_db_poisson_matches_duck_poisson(tmp_path):
    rng = np.random.default_rng(456)
    n = 5000
    x1 = rng.integers(0, 5, n)
    x2 = rng.integers(0, 2, n)
    mu = np.exp(0.2 + 0.18 * x1 - 0.25 * x2)
    y = rng.poisson(mu)
    con = _make_con(tmp_path, pd.DataFrame({"y": y, "x1": x1, "x2": x2}))
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=456,
        method="irls",
        n_bootstraps=0,
    )
    old = DuckPoissonRegression(**kwargs)
    new = DBPoissonRegression(**kwargs)
    old.fit()
    old.fit_vcov()
    new.fit()
    new.fit_vcov()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)
    np.testing.assert_allclose(new.vcov, old.vcov)


def test_db_multinomial_logit_matches_duck_multinomial_logit(tmp_path):
    rng = np.random.default_rng(789)
    n = 4000
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 3, n)
    X = np.c_[np.ones(n), x1, x2]
    beta = np.array([[0.2, 0.35, -0.2], [-0.4, -0.15, 0.3]])
    eta = np.c_[X @ beta.T, np.zeros(n)]
    probs = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(["a", "b", "c"], p=prob) for prob in probs])
    con = _make_con(tmp_path, pd.DataFrame({"y": y, "x1": x1, "x2": x2}))
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=789,
        labels=["a", "b", "c"],
        baseline="c",
        n_bootstraps=0,
    )
    old = DuckMultinomialLogisticRegression(**kwargs)
    new = DBMultinomialLogisticRegression(**kwargs)
    old.fit()
    old.fit_vcov()
    new.fit()
    new.fit_vcov()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)
    np.testing.assert_allclose(new.vcov, old.vcov)


def test_db_poisson_multinomial_matches_duck_poisson_multinomial(tmp_path):
    rng = np.random.default_rng(321)
    labels = [f"label_{j}" for j in range(8)]
    rows = []
    for i in range(250):
        x1 = rng.integers(0, 4)
        x2 = rng.integers(0, 2)
        for j, label in enumerate(labels):
            mu = np.exp(-1.0 + 0.08 * j + 0.15 * x1 - 0.1 * x2)
            rows.append((i, label, rng.poisson(mu), x1, x2))
    df = pd.DataFrame(rows, columns=["row_id", "label", "count", "x1", "x2"])
    con = _make_con(tmp_path, df)
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        count_col="count",
        label_col="label",
        covars=["x1", "x2"],
        seed=321,
        n_bootstraps=0,
    )
    old = DuckPoissonMultinomialRegression(**kwargs)
    new = DBPoissonMultinomialRegression(**kwargs)
    old.fit()
    new.fit()
    pd.testing.assert_frame_equal(
        new.summary()["point_estimate"],
        old.summary()["point_estimate"],
        check_exact=False,
    )


def test_db_event_study_matches_duck_event_study(tmp_path):
    rows = []
    for unit, cohort in [(1, 2), (2, 3), (3, None), (4, 2), (5, None)]:
        for time in range(5):
            treated = int(cohort is not None and time >= cohort)
            y = 1.0 + 0.2 * unit + 0.3 * time + 0.8 * treated
            rows.append((unit, time, treated, y, unit))
    df = pd.DataFrame(rows, columns=["unit", "time", "D", "Y", "cluster"])
    con = _make_con(tmp_path, df)
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y",
        treatment_col="D",
        unit_col="unit",
        time_col="time",
        cluster_col="cluster",
        seed=42,
        n_bootstraps=0,
    )
    old = DuckMundlakEventStudy(**kwargs)
    new = DBMundlakEventStudy(**kwargs)
    old.fit()
    new.fit()

    old_est = old.summary()["point_estimate"]
    new_est = new.summary()["point_estimate"]
    assert new_est.keys() == old_est.keys()
    for cohort in old_est:
        pd.testing.assert_frame_equal(new_est[cohort], old_est[cohort], check_exact=False)
