import duckdb
import numpy as np
import pandas as pd
import pytest

from duckreg.estimators import (
    DuckDoubleDemeaning,
    DuckMundlak,
    DuckMundlakEventStudy,
    DuckRegression,
)


@pytest.fixture
def panel_db(tmp_path):
    rng = np.random.default_rng(0)
    units, periods = 500, 5
    df = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(units), periods),
            "time": np.tile(np.arange(periods), units),
        }
    )
    # Staggered adoption: a third of the units start treatment in period 2, a
    # third in period 3, and the rest never do.
    cohort = rng.choice([2, 3, periods], units)[df.unit]
    df["D"] = (df.time >= cohort).astype(int)
    df["Y"] = 0.5 * df.D + rng.normal(size=units)[df.unit] + rng.normal(size=len(df))
    path = str(tmp_path / "panel.db")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE data AS SELECT * FROM df")
    con.close()
    return path


MAKERS = {
    "DuckRegression": lambda db: DuckRegression(
        db, "data", "Y ~ D", cluster_col="unit", seed=3, n_bootstraps=20
    ),
    "DuckMundlak": lambda db: DuckMundlak(
        db, "data", "Y", ["D"], unit_col="unit", time_col="time",
        cluster_col="unit", seed=3, n_bootstraps=20,
    ),
    "DuckDoubleDemeaning": lambda db: DuckDoubleDemeaning(
        db, "data", "Y", "D", "unit", "time", cluster_col="unit", seed=3, n_bootstraps=20
    ),
    "DuckMundlakEventStudy": lambda db: DuckMundlakEventStudy(
        db, "data", "Y", "D", "unit", "time", cluster_col="unit", seed=3, n_bootstraps=20
    ),
}


def _bootstrap_output(model):
    if isinstance(model, DuckMundlakEventStudy):
        return np.concatenate([v["se"].to_numpy() for v in model.summary().values()])
    return model.vcov


@pytest.mark.parametrize("name", list(MAKERS))
def test_cluster_bootstrap_is_reproducible_with_seed(panel_db, name):
    vcovs = []
    for _ in range(5):
        model = MAKERS[name](panel_db)
        model.fit()
        vcovs.append(_bootstrap_output(model))
    # DuckDB's parallel SUM can reorder floating-point additions, so runs
    # agree to rounding error, not bit for bit.
    for vcov in vcovs[1:]:
        np.testing.assert_allclose(vcov, vcovs[0], rtol=1e-10)
