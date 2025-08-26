# %%
import numpy as np
import pandas as pd
from duckreg.estimators import DuckRegression
import duckdb
import pyfixest as pf
import statsmodels.formula.api as smf
import time
import gc

# %%
# Generate sample data
# def generate_sample_data(N=10_000_000, seed=12345):
#     rng = np.random.default_rng(seed)
#     D = rng.choice([0, 1], size=(N, 1))
#     X = rng.choice(range(20), (N, 2), True)
#     Y = D + X @ np.array([1, 2]).reshape(2, 1) + rng.normal(size=(N, 1))
#     Y2 = -1 * D + X @ np.array([1, 2]).reshape(2, 1) + rng.normal(size=(N, 1))
#     df = pd.DataFrame(
#         np.concatenate([Y, Y2, D, X], axis=1), columns=["Y", "Y2", "D", "f1", "f2"]
#     ).assign(rowid=range(N))
#     return df


# # Function to create and populate DuckDB database
# def create_duckdb_database(df, db_name="large_dataset.db", table="data"):
#     conn = duckdb.connect(db_name)
#     conn.execute(f"DROP TABLE IF EXISTS {table}")
#     conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
#     conn.close()
#     print(f"Data loaded into DuckDB database: {db_name}")


# # %%
# # Generate and save data
# df = generate_sample_data()
# db_name = "large_dataset.db"
# create_duckdb_database(df, db_name)

# %%
df = duckdb.connect("large_dataset.db").execute("SELECT * FROM data").fetchdf()
print(df.shape)


def benchmark_duckreg_optimized():
    """Benchmark the optimized DuckRegression implementation"""
    gc.collect()
    start_time = time.time()

    # Import the optimized version (should use the current code in the repo)
    from duckreg.estimators import DuckRegression

    m = DuckRegression(
        db_name="large_dataset.db",
        table_name="data",
        formula="Y ~ D + f1 + f2",
        cluster_col="",
        n_bootstraps=0,
        seed=42,
    )
    m.fit()
    m.fit_vcov()
    results = m.summary()

    end_time = time.time()
    execution_time = end_time - start_time

    restab = pd.DataFrame(
        np.c_[results["point_estimate"], results["standard_error"]],
        columns=["point_estimate", "standard_error"],
    )

    print(f"Optimized DuckRegression time: {execution_time:.4f} seconds")
    return restab, execution_time


# Run optimized benchmark
optimized_results, optimized_time = benchmark_duckreg_optimized()
print("\nOptimized results:")
print(optimized_results)

# Compare performance
print(f"Optimized time: {optimized_time:.4f} seconds")

# %%

m_pf = pf.feols("Y ~ D | f1 + f2", df, vcov="hetero")
m_pf.tidy()

# %%
# Pyfixest with full data takes ~ 40x longer to compute.

# %%
# %%time


# m_smf = smf.ols("Y ~ D + C(f1) + C(f2)", df).fit(cov_type="HC1")
# m_smf.params.loc["D"], m_smf.bse.loc["D"]

# # %%
# # The full data run in statsmodels takes around 600x longer than the compressed representation in `DuckRegression`.

# # %%
# pd.DataFrame(
#     np.c_[
#         np.r_[results["point_estimate"][1], results["standard_error"][1]],
#         m_pf.tidy().iloc[0][["Estimate", "Std. Error"]].values,
#         np.r_[m_smf.params.loc["D"], m_smf.bse.loc["D"]],
#     ],
#     columns=["duckreg", "pyfixest", "statsmodels"],
#     index=["estimate", "std.error"],
# )
