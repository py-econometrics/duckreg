
import numpy as np
import pandas as pd
import duckdb
from duckreg.estimators import DuckDML
import pytest

def test_dml_analytic_vcov():
    # Generate synthetic data
    np.random.seed(42)
    n = 10000
    n_towns = 10
    
    df = pd.DataFrame({
        'town_id': np.random.randint(0, n_towns, n),
    })
    
    # Treatment X
    df['X'] = np.random.randn(n) + 0.5 * df['town_id']
    # Outcome Y
    df['Y'] = 2.0 * df['X'] + df['town_id'] + np.random.randn(n)
    
    db_name = "test_analytic.db"
    con = duckdb.connect(db_name)
    con.register('df', df)
    con.execute("CREATE TABLE data AS SELECT * FROM df")
    con.close()
    
    try:
        dml = DuckDML(
            db_name=db_name,
            table_name="data",
            outcome_var='Y',
            treatment_var='X',
            discrete_covars=['town_id'],
            seed=42,
            n_bootstraps=0 # No bootstrap
        )
        
        # Run standard fit pipeline
        dml.fit()
        
        # Now calculate analytic vcov
        dml.fit_vcov()
        
        print("Point estimate:", dml.point_estimate)
        print("Analytic VCOV:", dml.vcov)
        
        # Basic check: SE should be small but positive
        se = np.sqrt(dml.vcov[0,0])
        assert se > 0
        assert se < 0.1 # given N=10000
        
        # Check summary
        summ = dml.summary()
        print("Summary:", summ)
        assert 'standard_error' in summ
        assert np.isclose(summ['standard_error'][0], se)
        
    finally:
        import os
        if os.path.exists(db_name):
            os.remove(db_name)

if __name__ == "__main__":
    test_dml_analytic_vcov()
