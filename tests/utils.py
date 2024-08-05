import numpy as np
import pandas as pd
import duckdb

# Generate sample data
def generate_sample_data(N=10_000, seed=12345):
    rng = np.random.default_rng(seed)
    D = rng.choice([0, 1], size=(N, 1))
    X = rng.choice(range(20), (N, 2), True)
    Y = D + X @ np.array([1, 2]).reshape(2, 1) + rng.normal(size=(N, 1))
    Y2 = -1 * D + X @ np.array([1, 2]).reshape(2, 1) + rng.normal(size=(N, 1))
    df = pd.DataFrame(
        np.concatenate([Y, Y2, D, X], axis=1), columns=["Y", "Y2", "D", "f1", "f2"]
    ).assign(rowid=range(N))
    return df


# Function to create and populate DuckDB database
def create_duckdb_database(df, db_name="large_dataset.db", table="data"):
    conn = duckdb.connect(db_name)
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    conn.close()
    print(f"Data loaded into DuckDB database: {db_name}")