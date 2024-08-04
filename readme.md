#`duckreg` : out-of-memory regressions with duckdb

Package to run stratified out-of-memory regressions with duckdb. The package is a wrapper around the `duckdb` package and provides a simple interface to run regressions on very large datasets that do not fit in memory by reducing the data to a set of summary statistics and runs weighted least squares with frequency weights.

Specific regressions:
+ `DuckRegression`: general linear regression, which compresses the data to y averages stratified by all unique values of the x variables
+ `DuckMundlak`: Mundlak regression, which compresses the data to y averages stratified by [$1, w, \bar{w}_{i, \cdot}, \bar{w}_{\cdot, t}$] where $w$ is a covariate (typically treatment)
+ `DuckDoubleDemeaning`: Double demeaning regression, which compresses the data to y averages by all values of $w$ after demeaning by $\bar{w}_{i, \cdot}, \bar{w}_{\cdot, t}$.
