
import pytest
import numpy as np
import pandas as pd
import duckdb
import os
from duckreg.estimators import DuckDML

def generate_data_multivariate(n, n_towns, n_days, true_betas, seed=None):
    if seed is not None:
        np.random.seed(seed)

    df = pd.DataFrame({
        'town_id': np.random.randint(0, n_towns, n),
        'day_id': np.random.randint(0, n_days, n),
    })

    # Nonlinear function of covariates
    g_z = 0.5 * df['town_id'] + 0.01 * df['day_id'] * df['town_id'] + np.sin(df['day_id'])

    # Treatment correlated with fixed effects and each other
    df['X1'] = 0.2 * df['town_id'] + 0.1 * df['day_id'] + np.random.randn(n)
    df['X2'] = 0.3 * df['town_id'] - 0.05 * df['day_id'] + 0.5 * df['X1'] + np.random.randn(n)

    # Errors
    errors = np.random.normal(0, 2, n)

    df['Y'] = true_betas[0] * df['X1'] + true_betas[1] * df['X2'] + g_z + errors
    return df

@pytest.fixture
def dml_db():
    db_name = "test_dml_pytest.db"
    if os.path.exists(db_name):
        os.remove(db_name)
    yield db_name
    if os.path.exists(db_name):
        os.remove(db_name)

def test_dml_multivariate(dml_db):
    n = 10000
    n_towns = 50
    n_days = 20
    true_betas = [1.5, -0.8]
    n_bootstraps = 10  # Low for speed in tests
    
    df = generate_data_multivariate(n, n_towns, n_days, true_betas, seed=42)
    
    con = duckdb.connect(dml_db)
    con.register('df_pandas', df)
    con.execute("CREATE TABLE data AS SELECT * FROM df_pandas")
    con.close()
    
    dml = DuckDML(
        db_name=dml_db,
        table_name="data",
        outcome_var='Y',
        treatment_var=['X1', 'X2'],
        discrete_covars=['town_id', 'day_id'],
        seed=42,
        n_bootstraps=n_bootstraps
    )
    
    dml.fit()
    res = dml.summary()
    
    estimates = res['point_estimate']
    
    # Check if estimates are reasonable (atol=0.2 for small N/sim)
    np.testing.assert_allclose(estimates, true_betas, atol=0.2)
    
    # Check shapes
    assert len(estimates) == 2
    assert res['standard_error'].shape == (2,) # diagonal
    # Actually summary returns sqrt(diag(vcov)), so it's 1D array
    
def test_dml_univariate(dml_db):
    n = 10000
    n_towns = 50
    n_days = 20
    true_betas = [1.5, 0.0] # Use X1 only
    n_bootstraps = 10
    
    df = generate_data_multivariate(n, n_towns, n_days, true_betas, seed=42)
    
    con = duckdb.connect(dml_db)
    con.register('df_pandas', df)
    con.execute("CREATE TABLE data AS SELECT * FROM df_pandas")
    con.close()
    
    dml = DuckDML(
        db_name=dml_db,
        table_name="data",
        outcome_var='Y',
        treatment_var='X1', # String input
        discrete_covars=['town_id', 'day_id'],
        seed=42,
        n_bootstraps=n_bootstraps
    )
    
    dml.fit()
    res = dml.summary()
    
    estimates = res['point_estimate']
    
    np.testing.assert_allclose(estimates, [1.5], atol=0.2)
    assert len(estimates) == 1
