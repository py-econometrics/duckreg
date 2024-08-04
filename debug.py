import numpy as np
import pandas as pd
from duckreg.estimators import DuckRegression
import duckdb

m = DuckRegression(
    db_name='large_dataset.db',
    table_name='data',
    formula="Y ~ D | f1 + f2",
    cluster_col="f1",
    n_bootstraps=100
)
m.fit()
results = m.summary()

restab = pd.DataFrame(
    np.c_[results["point_estimate"], results["standard_error"]],
    columns=["point_estimate", "standard_error"],
)
restab