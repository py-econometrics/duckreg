import ibis
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from duckreg import DBRegression, DuckRegression


@pytest.mark.parametrize("estimator", [DuckRegression, DBRegression])
@pytest.mark.parametrize("groups", [3, 4])
def test_hc1_matches_raw_rows_without_dense_group_weights(
    estimator, groups, monkeypatch
):
    # With three columns and three groups, applying 1-D weights across columns
    # broadcasts successfully but gives the wrong covariance.
    cells = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])[:groups]
    x = np.repeat(cells, [2, 5, 3, 7][:groups], axis=0)
    y = x[:, 0] + np.random.default_rng(42).normal(scale=1 + x[:, 1])
    data = pd.DataFrame({"y": y, "a": x[:, 0], "b": x[:, 1]})
    design = np.column_stack([np.ones(len(y)), x])
    influence = np.linalg.pinv(design)
    beta = influence @ y
    scores = influence * (y - design @ beta)
    expected = (scores @ scores.T) * len(y) / (len(y) - len(beta))

    db = ibis.duckdb.connect()
    try:
        db.create_table("observations", data)
        model = estimator(
            db_name=None,
            table_name="observations",
            formula="y ~ a + b",
            connection=db,
            cluster_col=None,
            seed=1,
            n_bootstraps=0,
        )
        model.fit()
        diagonal = np.diag

        def guard_group_weights(array, *args, **kwargs):
            if np.shape(array) == (groups,):
                pytest.fail("Allocated a square matrix for group weights")
            return diagonal(array, *args, **kwargs)

        with monkeypatch.context() as guard:
            guard.setattr(np, "diag", guard_group_weights)
            model.fit_vcov()
        assert_allclose(model.point_estimate, beta, atol=1e-12, rtol=0)
        assert_allclose(model.vcov, expected, atol=1e-12, rtol=0)
    finally:
        db.disconnect()
