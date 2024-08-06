import os
import numpy as np
import pandas as pd
import duckdb


# Generate sample data
def generate_sample_data(N=10_000_000, seed=42):
    rng = np.random.default_rng(seed)
    D = rng.choice([0, 1], size=(N, 1))
    X = rng.choice(range(20), (N, 2), True)
    Y = D + X @ np.array([1, 2]).reshape(2, 1) + rng.normal(size=(N, 1))
    Y2 = -1 * D + X @ np.array([1, 2]).reshape(2, 1) + rng.normal(size=(N, 1))
    df = pd.DataFrame(
        np.concatenate([Y, Y2, D, X], axis=1), columns=["Y", "Y2", "D", "f1", "f2"]
    )
    return df


def create_duckdb_database(df, db_name="test_dataset.db", table="data"):
    db_path = os.path.abspath(db_name)
    conn = duckdb.connect(db_path)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"Created table '{table}' with {result[0]} rows in database: {db_path}")
    finally:
        conn.close()
    return db_path
