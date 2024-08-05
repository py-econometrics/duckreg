# `duckreg` : very fast out-of-memory regressions with `duckdb`

python package to run stratified/saturated regressions out-of-memory with duckdb. The package is a wrapper around the `duckdb` package and provides a simple interface to run regressions on very large datasets that do not fit in memory by reducing the data to a set of summary statistics and runs weighted least squares with frequency weights. Robust standard errors are computed from sufficient statistics, while clustered standard errors are computed using the cluster bootstrap.

See examples in `notebooks/introduction.ipynb`.

<p align="center">
  <img src="https://static.independent.co.uk/s3fs-public/thumbnails/image/2016/02/14/12/duck-rabbit.png" width="350">
</p>

install (preferably in a `venv`) with
```
(uv) pip install git+https://github.com/apoorvalal/duckreg.git
```

or git clone this repository and install in editable mode.

---

Currently supports the following regression specifications:
+ `DuckRegression`: general linear regression, which compresses the data to y averages stratified by all unique values of the x variables
+ `DuckMundlak`: Mundlak regression, which compresses the data to y averages stratified by $1, w, \bar{w}_{i, .}, \bar{w}_{., t}$  where $w$ is a covariate (typically treatment)
+ `DuckDoubleDemeaning`: Double demeaning regression, which compresses the data to y averages by all values of $w$ after demeaning by $\bar{w}_{i, .}, \bar{w}_{., t}, \bar{w}$ .

---
references:

methods:
+ [Arkhangelsky and Imbens (2023)](https://arxiv.org/abs/1807.02099)
+ [Wooldridge 2021](https://www.researchgate.net/publication/353938385_Two-Way_Fixed_Effects_the_Two-Way_Mundlak_Regression_and_Difference-in-Differences_Estimators)
+ [Wong et al 2021](https://arxiv.org/abs/2102.11297)

libraries:
+ [Grant McDermott's duckdb lecture](https://grantmcdermott.com/duckdb-polars/)
