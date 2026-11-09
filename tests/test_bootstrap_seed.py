"""A seeded bootstrap must not depend on the order the database returns rows."""

import duckdb
import numpy as np
import pandas as pd
import pytest

import duckreg as dr
from duckreg.duckreg import sorted_clusters


def _panel():
    rng = np.random.default_rng(0)
    units, periods = 400, 5
    df = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(units), periods),
            "time": np.tile(np.arange(periods), units),
        }
    )
    # Staggered adoption: treatment starts in period 2, period 3, or never.
    cohort = rng.choice([2, 3, periods], units)[df.unit]
    df["D"] = (df.time >= cohort).astype(int)
    df["town_id"] = rng.integers(0, 40, len(df))
    df["day_id"] = df.time
    df["X1"] = rng.normal(size=len(df)) + 0.05 * df.town_id
    df["Y"] = (
        0.5 * df.D
        + 0.3 * df.X1
        + rng.normal(size=units)[df.unit]
        + rng.normal(size=len(df))
    )
    return df


def _write(path, df):
    con = duckdb.connect(str(path))
    con.register("df", df)
    con.execute("CREATE TABLE data AS SELECT * FROM df")
    con.close()
    return str(path)


@pytest.fixture(scope="module")
def same_rows_two_orders(tmp_path_factory):
    """The same rows stored in two physical orders."""
    df = _panel()
    tmp = tmp_path_factory.mktemp("roworder")
    return (
        _write(tmp / "sorted.db", df),
        _write(tmp / "shuffled.db", df.sample(frac=1, random_state=1)),
    )


KW = dict(seed=3, n_bootstraps=30)
PANEL = dict(unit_col="unit", time_col="time", cluster_col="unit")
DML = dict(outcome_var="Y", treatment_var="X1", discrete_covars=["town_id", "day_id"])
MAKERS = {
    "DuckRegression": lambda db: dr.DuckRegression(
        db, "data", "Y ~ D", cluster_col="unit", **KW
    ),
    "DBRegression": lambda db: dr.DBRegression(
        db, "data", "Y ~ D", cluster_col="unit", **KW
    ),
    "DuckMundlak": lambda db: dr.DuckMundlak(db, "data", "Y", ["D"], **PANEL, **KW),
    "DBMundlak": lambda db: dr.DBMundlak(db, "data", "Y", ["D"], **PANEL, **KW),
    "DuckDoubleDemeaning": lambda db: dr.DuckDoubleDemeaning(
        db, "data", "Y", "D", "unit", "time", cluster_col="unit", **KW
    ),
    "DuckMundlakEventStudy": lambda db: dr.DuckMundlakEventStudy(
        db, "data", "Y", "D", "unit", "time", cluster_col="unit", **KW
    ),
    "DBMundlakEventStudy": lambda db: dr.DBMundlakEventStudy(
        db, "data", "Y", "D", "unit", "time", cluster_col="unit", **KW
    ),
    "DuckDML": lambda db: dr.DuckDML(db_name=db, table_name="data", **DML, **KW),
    "DBDML": lambda db: dr.DBDML(db_name=db, table_name="data", **DML, **KW),
}


def _standard_errors(model):
    if isinstance(model, (dr.DuckMundlakEventStudy, dr.DBMundlakEventStudy)):
        summary = model.summary()
        return np.concatenate(
            [summary[k]["se"].sort_index().to_numpy() for k in sorted(summary)]
        )
    return np.sqrt(np.diag(np.atleast_2d(model.vcov)))


# Floating-point sums can be added in a different order, so runs agree to
# rounding error, not bit for bit.
RTOL = 1e-10


@pytest.mark.parametrize("name", list(MAKERS))
def test_bootstrap_does_not_depend_on_row_order(same_rows_two_orders, name):
    ses = []
    for db in same_rows_two_orders:
        model = MAKERS[name](db)
        model.fit()
        ses.append(_standard_errors(model))
    np.testing.assert_allclose(ses[0], ses[1], rtol=RTOL)


@pytest.mark.parametrize("name", ["DuckDML", "DBDML"])
def test_dml_bootstrap_does_not_depend_on_group_order(same_rows_two_orders, name):
    # DuckDB happens to return GROUP BY keys in a stable order, so shuffle the
    # compressed groups directly, as another backend might return them.
    model = MAKERS[name](same_rows_two_orders[0])
    model.fit()
    expected = model.vcov
    model.df_compressed = model.df_compressed.sample(frac=1, random_state=2)
    model.rng = np.random.default_rng(KW["seed"])
    np.testing.assert_allclose(model.bootstrap(), expected, rtol=RTOL)


def test_sorted_clusters_puts_missing_last():
    got = sorted_clusters(["b", None, "a", "b"]).tolist()
    assert got[:2] == ["a", "b"]
    assert pd.isna(got[2]) and len(got) == 3
