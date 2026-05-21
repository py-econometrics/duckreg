# `duckreg`: very fast out-of-memory regressions on SQL backends

python package to run stratified/saturated regressions out-of-memory through [Ibis](https://ibis-project.org/). DuckDB remains the default local backend for backwards compatibility, and the same estimators can run against any Ibis SQL backend that supports the generated aggregation queries. R users, check out [Grant McDermott's port of this package](https://github.com/grantmcdermott/duckreg). 

The package provides a simple interface to run regressions on very large datasets that do not fit in memory by reducing the data inside the database to a set of summary statistics and then running weighted least squares with frequency weights. Robust standard errors are computed from sufficient statistics, while clustered standard errors are computed using the cluster bootstrap. Methodological details and benchmarks are provided in [this](https://arxiv.org/abs/2410.09952) paper. See examples in `notebooks/introduction.ipynb`.

<p align="center">
  <img src="https://static.independent.co.uk/s3fs-public/thumbnails/image/2016/02/14/12/duck-rabbit.png" width="350">
</p>

- install

```
pip install duckreg
```

- dev install (preferably in a `venv`) with
```
uv pip install -e '.[test]'
```

or install from git with `uv pip install git+https://github.com/py-econometrics/duckreg.git`.

By default, legacy DuckDB paths still work:

```python
from duckreg import DuckRegression

model = DuckRegression(
    db_name="large_dataset.db",
    table_name="data",
    formula="Y ~ D + X",
    cluster_col="cluster_id",
    seed=42,
)
model.fit()
```

For a remote database, create an Ibis backend and pass it through `connection`.
For example, with Databricks:

```python
import ibis
from duckreg import DuckRegression

con = ibis.databricks.connect(
    server_hostname="...",
    http_path="...",
    access_token="...",
    catalog="main",
    schema="analytics",
)

model = DuckRegression(
    db_name=None,
    connection=con,
    table_name="large_experiment_table",
    formula="Y ~ D + X",
    cluster_col="cluster_id",
    seed=42,
    n_bootstraps=0,
)
model.fit()
```

---

Currently supports the following regression specifications:
1. `DuckRegression`: general linear regression, which compresses the data to y averages stratified by all unique values of the x variables
2. `DuckMundlak`: One- or Two-Way Mundlak regression, which compresses the data to the following RHS and avoids the need to incorporate unit (and time FEs)

$$
y \sim 1, w, \bar{w}\_{i, .}, \bar{w}\_{., t}
$$

3. `DuckDoubleDemeaning`: Double demeaning regression, which compresses the data to y averages by all values of $w$ after demeaning. This also eliminates unit and time FEs

$$
y \sim (W\_{it} - \bar{w}\_{i, .} - \bar{w}\_{., t} + \bar{w}\_{., .})
$$

4. `DuckMundlakEventStudy`: Two-way mundlak with dynamic treatment effects. This incorporates treatment-cohort FEs ($\psi\_i$), time-period FEs ($\gamma\_t$) and dynamic treatment effects $\tau\_k$ given by cohort X time interactions.

$$
y \sim \psi\_i + \gamma\_t + \sum\_{k=1}^{T} \tau\_{k} D\_i 1(t = k)
$$

All the above regressions are run in compressed fashion through the configured Ibis backend. Formula-level fixed effects are not part of `DuckRegression`; use the panel-specific `DuckMundlak` or `DuckDoubleDemeaning` estimators for fixed-effect style designs.

Please cite the following paper if you use `duckreg` in your research: 

```
@misc{lal2024largescalelongitudinalexperiments,
      title={Large Scale Longitudinal Experiments: Estimation and Inference}, 
      author={Apoorva Lal and Alexander Fischer and Matthew Wardrop},
      year={2024},
      eprint={2410.09952},
      archivePrefix={arXiv},
      primaryClass={econ.EM},
      url={https://arxiv.org/abs/2410.09952}, 
}
```

---
references:

methods:
+ [Arkhangelsky and Imbens (2023)](https://arxiv.org/abs/1807.02099)
+ [Wooldridge 2021](https://www.researchgate.net/publication/353938385_Two-Way_Fixed_Effects_the_Two-Way_Mundlak_Regression_and_Difference-in-Differences_Estimators)
+ [Wong et al (2021)](https://arxiv.org/abs/2102.11297)

libraries:
+ [Grant McDermott's duckdb lecture](https://grantmcdermott.com/duckdb-polars/)
