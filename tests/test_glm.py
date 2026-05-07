import os

import duckdb
import numpy as np
import pandas as pd

from duckreg.estimators import (
    DuckLogisticRegression,
    DuckMultinomialLogisticRegression,
    DuckPoissonMultinomialRegression,
    DuckPoissonRegression,
    _add_intercept,
    _multinomial_irls,
    _weighted_logistic_irls,
    _weighted_poisson_irls,
)


def _write_db(tmp_path, df, table="data"):
    db_path = os.path.join(tmp_path, "glm_test.db")
    conn = duckdb.connect(db_path)
    conn.register("df", df)
    conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
    conn.close()
    return db_path


def test_logistic_regression_matches_uncompressed_irls(tmp_path):
    rng = np.random.default_rng(123)
    n = 5000
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 3, n)
    eta = -0.7 + 0.45 * x1 - 0.35 * x2
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    db_path = _write_db(tmp_path, df)

    model = DuckLogisticRegression(
        db_name=db_path,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=123,
        method="irls",
        n_bootstraps=0,
    )
    model.fit()
    model.fit_vcov()

    X = _add_intercept(df[["x1", "x2"]].values)
    expected = _weighted_logistic_irls(X, df["y"].values, np.ones(n))
    np.testing.assert_allclose(model.point_estimate, expected, rtol=1e-8, atol=1e-8)
    assert model.summary()["standard_error"].shape == (3,)
    assert model.df_compressed["count"].sum() == n


def test_poisson_regression_matches_uncompressed_irls(tmp_path):
    rng = np.random.default_rng(456)
    n = 5000
    x1 = rng.integers(0, 5, n)
    x2 = rng.integers(0, 2, n)
    mu = np.exp(0.2 + 0.18 * x1 - 0.25 * x2)
    y = rng.poisson(mu)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    db_path = _write_db(tmp_path, df)

    model = DuckPoissonRegression(
        db_name=db_path,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=456,
        method="irls",
        n_bootstraps=0,
    )
    model.fit()
    model.fit_vcov()

    X = _add_intercept(df[["x1", "x2"]].values)
    expected = _weighted_poisson_irls(X, df["y"].values, np.ones(n))
    np.testing.assert_allclose(model.point_estimate, expected, rtol=1e-8, atol=1e-8)
    assert model.summary()["standard_error"].shape == (3,)
    assert model.df_compressed["count"].sum() == n


def test_multinomial_logistic_regression_matches_uncompressed_irls(tmp_path):
    rng = np.random.default_rng(789)
    n = 4000
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 3, n)
    X = _add_intercept(np.c_[x1, x2])
    beta = np.array([[0.2, 0.35, -0.2], [-0.4, -0.15, 0.3]])
    eta = X @ beta.T
    eta = np.c_[eta, np.zeros(n)]
    probs = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(["a", "b", "c"], p=prob) for prob in probs])
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    db_path = _write_db(tmp_path, df)

    model = DuckMultinomialLogisticRegression(
        db_name=db_path,
        table_name="data",
        formula="y ~ x1 + x2",
        seed=789,
        labels=["a", "b", "c"],
        baseline="c",
        n_bootstraps=0,
    )
    model.fit()
    model.fit_vcov()

    counts = np.column_stack([(y == label).astype(float) for label in ["a", "b", "c"]])
    expected = _multinomial_irls(X, counts)
    np.testing.assert_allclose(model.point_estimate, expected, rtol=1e-8, atol=1e-8)
    assert model.summary()["standard_error"].shape == (2, 3)
    assert model.df_compressed["count"].sum() == n


def test_poisson_multinomial_many_label_shape(tmp_path):
    rng = np.random.default_rng(321)
    n = 250
    labels = [f"label_{j}" for j in range(8)]
    rows = []
    for i in range(n):
        x1 = rng.integers(0, 4)
        x2 = rng.integers(0, 2)
        for j, label in enumerate(labels):
            mu = np.exp(-1.0 + 0.08 * j + 0.15 * x1 - 0.1 * x2)
            rows.append((i, label, rng.poisson(mu), x1, x2))
    df = pd.DataFrame(rows, columns=["row_id", "label", "count", "x1", "x2"])
    db_path = _write_db(tmp_path, df)

    model = DuckPoissonMultinomialRegression(
        db_name=db_path,
        table_name="data",
        count_col="count",
        label_col="label",
        covars=["x1", "x2"],
        seed=321,
        n_bootstraps=0,
    )
    model.fit()
    coefs = model.summary()["point_estimate"]
    assert list(coefs.columns) == ["Intercept", "x1", "x2"]
    assert list(coefs.index) == labels
    assert np.isfinite(coefs.values).all()
