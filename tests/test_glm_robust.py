import ibis
import numpy as np
import pandas as pd
import pytest

from duckreg.dbreg import DBLogisticRegression, DBPoissonRegression
from duckreg.estimators import DuckLogisticRegression, DuckPoissonRegression


def _data(family):
    rng = np.random.default_rng(7)
    n = 4000
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 3, n)
    d = rng.integers(0, 2, n)
    eta = -0.4 + 0.3 * x1 - 0.25 * x2 + 0.2 * d
    if family == "logistic":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    else:
        y = rng.poisson(np.exp(eta))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "d": d})


def _hc0_uncompressed(df, covars, beta, family):
    """Sandwich with one score per observation, computed on the raw rows."""
    X = np.column_stack([np.ones(len(df)), df[covars].to_numpy(float)])
    y = df["y"].to_numpy(float)
    eta = X @ beta
    mu = 1 / (1 + np.exp(-eta)) if family == "logistic" else np.exp(eta)
    var = mu * (1 - mu) if family == "logistic" else mu
    bread = np.linalg.inv(X.T @ (var[:, None] * X))
    meat = X.T @ (((y - mu) ** 2)[:, None] * X)
    return bread @ meat @ bread


@pytest.mark.parametrize(
    "cls,family",
    [
        (DuckLogisticRegression, "logistic"),
        (DuckPoissonRegression, "poisson"),
        (DBLogisticRegression, "logistic"),
        (DBPoissonRegression, "poisson"),
    ],
)
# "y ~ d" with binary d is saturated: one coefficient per cell, so every
# cell's summed score is zero, while the per-observation scores are not.
@pytest.mark.parametrize("formula", ["y ~ x1 + x2", "y ~ d"])
def test_robust_vcov_matches_per_observation_sandwich(tmp_path, cls, family, formula):
    df = _data(family)
    con = ibis.duckdb.connect(tmp_path / "glm_robust.db")
    con.create_table("data", df, overwrite=True)
    model = cls(
        db_name=None,
        connection=con,
        table_name="data",
        formula=formula,
        seed=1,
        method="irls",
        n_bootstraps=0,
    )
    model.fit()
    vcov = model.fit_vcov(robust=True)

    expected = _hc0_uncompressed(df, model.covars, model.point_estimate, family)
    np.testing.assert_allclose(vcov, expected, rtol=1e-8)
    assert np.all(np.sqrt(np.diag(vcov)) > 0)
