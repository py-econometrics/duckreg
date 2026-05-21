import pytest
import ibis
import numpy as np
import pandas as pd

import duckreg
from duckreg.estimators import DuckLogisticRegression, DuckRegression


def test_public_api_exports_estimators():
    assert duckreg.DuckRegression is DuckRegression
    assert duckreg.DuckLogisticRegression is DuckLogisticRegression
    assert hasattr(duckreg, "DuckRidge")


def test_duck_regression_rejects_formula_fixed_effects():
    with pytest.raises(NotImplementedError, match="Fixed effects"):
        DuckRegression(
            db_name=":memory:",
            table_name="data",
            formula="Y ~ D | unit",
            cluster_col="",
            seed=42,
            n_bootstraps=0,
        )


def test_glm_bootstrap_fails_explicitly():
    model = DuckLogisticRegression(
        db_name=":memory:",
        table_name="data",
        formula="y ~ x",
        seed=42,
        n_bootstraps=1,
    )
    with pytest.raises(NotImplementedError, match="Bootstrap is not implemented"):
        model.bootstrap()


def test_duck_regression_accepts_ibis_backend_connection(tmp_path):
    df = pd.DataFrame(
        {
            "Y": [1.0, 2.0, 2.0, 4.0],
            "D": [0.0, 1.0, 0.0, 1.0],
            "f1": [0.0, 0.0, 1.0, 1.0],
        }
    )
    con = ibis.duckdb.connect(tmp_path / "api.db")
    con.create_table("data", df, overwrite=True)

    model = DuckRegression(
        db_name=None,
        connection=con,
        table_name="data",
        formula="Y ~ D + f1",
        cluster_col="",
        seed=42,
        n_bootstraps=0,
    )
    model.fit()

    X = np.c_[np.ones(len(df)), df[["D", "f1"]].values]
    expected = np.linalg.lstsq(X, df["Y"].values, rcond=None)[0]
    np.testing.assert_allclose(model.point_estimate, expected)
    assert "data" in con.list_tables()
