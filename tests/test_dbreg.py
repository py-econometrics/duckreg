import ibis
import numpy as np
import pandas as pd

import duckreg
from duckreg.dbreg import DBDML, DBDoubleDemeaning, DBMundlak, DBRegression
from duckreg.estimators import DuckDML, DuckDoubleDemeaning, DuckMundlak, DuckRegression


def _make_con(tmp_path, df, table="data"):
    con = ibis.duckdb.connect(tmp_path / "dbreg.db")
    con.create_table(table, df, overwrite=True)
    return con


def test_dbreg_exports():
    assert duckreg.DBRegression is DBRegression
    assert duckreg.DBDML is DBDML


def test_db_regression_matches_duck_regression(tmp_path):
    df = pd.DataFrame(
        {
            "rowid": range(8),
            "Y": [1.0, 2.0, 2.0, 4.0, 2.0, 3.0, 4.0, 5.0],
            "D": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "f1": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "cluster": [1, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    con = _make_con(tmp_path, df)
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        formula="Y ~ D + f1",
        cluster_col="cluster",
        seed=42,
        n_bootstraps=0,
    )
    old = DuckRegression(**kwargs)
    new = DBRegression(**kwargs)
    old.fit()
    new.fit()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)


def test_db_dml_matches_duck_dml_compression_and_estimate(tmp_path):
    df = pd.DataFrame(
        {
            "Y": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 5.0, 7.0],
            "D": [0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0],
            "X": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "Z": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    con = _make_con(tmp_path, df)
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y",
        treatment_var="D",
        discrete_covars=["X", "Z"],
        seed=42,
        n_bootstraps=0,
    )
    old = DuckDML(**kwargs)
    new = DBDML(**kwargs)
    old.fit()
    new.fit()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)


def test_db_mundlak_matches_duck_mundlak(tmp_path):
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "Y": [1.0, 2.0, 2.0, 4.0, 1.5, 3.0],
            "D": [0.0, 1.0, 0.0, 2.0, 1.0, 1.0],
        }
    )
    con = _make_con(tmp_path, df)
    kwargs = dict(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y",
        covariates=["D"],
        seed=42,
        unit_col="unit",
        time_col="time",
        n_bootstraps=0,
    )
    old = DuckMundlak(**kwargs)
    new = DBMundlak(**kwargs)
    old.fit()
    new.fit()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)


def test_db_double_demeaning_matches_duck_double_demeaning(tmp_path):
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "Y": [1.0, 2.0, 2.0, 4.0, 1.5, 3.0],
            "D": [0.0, 1.0, 0.0, 2.0, 1.0, 1.0],
        }
    )
    con = _make_con(tmp_path, df)
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
    old = DuckDoubleDemeaning(**kwargs)
    new = DBDoubleDemeaning(**kwargs)
    old.fit()
    new.fit()
    np.testing.assert_allclose(new.point_estimate, old.point_estimate)
