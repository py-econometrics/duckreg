import numpy as np
import pytest
import os
from duckreg.estimators import DuckRegression
from tests.utils import generate_sample_data, create_duckdb_database
import duckdb


@pytest.fixture(scope="session")
def get_data(force_regen):
    if force_regen:
        return generate_sample_data(1_000_000, seed=42)
    else:
        return generate_sample_data(1_000_000, seed=42)


@pytest.fixture(scope="session")
def database(get_data, force_regen):
    df = get_data
    db_name = "test_dataset.db"
    if force_regen and os.path.exists(db_name):
        os.remove(db_name)
    db_path = create_duckdb_database(df, db_name)
    return db_path


def get_numpy_coefficients(db_path, formula):
    conn = duckdb.connect(db_path)
    df = conn.execute("SELECT * FROM data").df()
    conn.close()

    y = df["Y"].values
    X_cols = [x.strip() for x in formula.split("~")[1].strip().split("+")]
    X = df[X_cols].values
    X = np.column_stack([np.ones(X.shape[0]), X])

    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    return coeffs[1:]


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ D",
        "Y ~ D + f1",
        "Y ~ D + f1 + f2",
    ],
)
def test_fitters(database, fml):
    db_path = database

    m_duck = DuckRegression(
        db_name=db_path,
        table_name="data",
        formula=fml,
        cluster_col="",
        n_bootstraps=0,
        seed=42,
    )
    m_duck.fit()
    m_duck.fit_vcov()
    # nobs
    (
        np.testing.assert_allclose(
            m_duck.df_compressed["count"].sum(), 1_000_000, rtol=1e-4
        ),
        "Number of observations are not equal",
    )

    results = m_duck.summary()
    compressed_coeffs, _ = (
        results["point_estimate"][1:],
        results["standard_error"][1:],
    )
    uncompressed_coeffs = get_numpy_coefficients(db_path, fml)
    # coefs
    (
        np.testing.assert_allclose(compressed_coeffs, uncompressed_coeffs, rtol=1e-4),
        f"Coefficients are not equal for formula {fml}",
    )
