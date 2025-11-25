import duckdb
import numpy as np
import pytest
from duckreg.stream import StreamingRegression


@pytest.fixture
def duckdb_conn():
    """Create an in-memory DuckDB connection."""
    conn = duckdb.connect(':memory:')
    yield conn
    conn.close()


def test_streaming_regression(duckdb_conn):
    """Test streaming regression with a simple example."""
    # Create sample data
    duckdb_conn.execute("""
        CREATE TABLE regression_data AS
        WITH features AS (
            SELECT
                random() as x0,
                random() as x1,
                random() as x2
            FROM generate_series(1, 100000) t(i)
        )
        SELECT
            x0,
            x1,
            x2,
            2.0*x0 - 1.5*x1 + 0.8*x2 + 0.1*random() as y
        FROM features
    """)

    # Perform streaming regression
    stream_reg = StreamingRegression.from_table(duckdb_conn, "regression_data")
    stream_reg.fit(feature_cols=["x0", "x1", "x2"], target_col="y")
    beta = stream_reg.solve_ols()

    # Check the results
    true_beta = np.array([2.0, -1.5, 0.8])
    assert np.allclose(beta, true_beta, atol=0.1)

    # Check that the condition number warning is raised
    with pytest.warns(UserWarning, match='High condition number'):
        stream_reg.stats.XtX = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        stream_reg.stats.check_condition()