import pytest

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
