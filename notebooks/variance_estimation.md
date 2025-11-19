# Variance Estimation Strategies with Compressed Regression

This document outlines strategies for variance estimation in the context of `duckreg`, specifically focusing on moving beyond the current bootstrap implementation to closed-form solutions using sufficient statistics.

## 1. Current Implementation: Hybrid Approach

Currently, `duckreg` employs a hybrid strategy:
*   **Homoskedastic / HC1 (Robust) Errors:** computed analytically using compressed sufficient statistics ($\sum y, \sum y^2$) via `fit_vcov()`.
*   **Cluster-Robust Standard Errors (CRSE):** computed via **Cluster Bootstrap** (resampling unique clusters).

While the bootstrap is robust and easy to implement, it can be computationally expensive ($B$ repetitions) compared to a closed-form solution.

## 2. Proposal: Closed-Form Cluster-Robust Inference

It is possible to compute CRSE analytically from compressed data, provided the compression granularity preserves cluster boundaries.

### The Algebra of Compressed CRSE

The standard Cluster-Robust Variance Estimator is:
$$ \hat{V}_{CR} = (X'X)^{-1} \left( \sum_{g=1}^G u_g u_g' \right) (X'X)^{-1} $$

Where $u_g$ is the score vector for cluster $g$:
$$ u_g = \sum_{i \in g} x_i e_i = \sum_{i \in g} x_i (y_i - x_i' \hat{\beta}) $$

In the compressed setting, let $k$ index the "strata" (rows in the compressed dataframe). If we ensure that **every stratum $k$ belongs to exactly one cluster $g$**, we can reconstruct the cluster scores perfectly from the compressed data:

$$ u_g = \sum_{k \in g} x_k (\text{sum\_Y}_k - n_k x_k' \hat{\beta}) $$

*   $x_k$: Covariates for stratum $k$ (constant within stratum).
*   $\text{sum\_Y}_k$: Sum of outcomes in stratum $k$ (available in compressed data).
*   $n_k$: Count of observations in stratum $k$.
*   $\hat{\beta}$: Point estimate (already computed).

The "Meat" matrix is then simply $\sum_g u_g u_g'$.

### Implementation Strategy

To support this in `duckreg`:

1.  **Conditional Compression:** If `cluster_col` is provided, add it to the `GROUP BY` clause during the `compress_data` step.
    *   *Current:* `GROUP BY covariates`
    *   *Proposed:* `GROUP BY covariates, cluster_col`
2.  **Score Aggregation:** After fitting $\hat{\beta}$, iterate through the compressed dataframe (or use DuckDB SQL) to compute $u_g$ for each unique `cluster_col`.
3.  **Matrix Calculation:** Compute the outer product of scores and sandwich with the bread.

### Edge Cases and Trade-offs

#### 1. Compression Efficiency vs. Cluster Cardinality
*   **Low Cardinality Clustering (e.g., State, Region):** Adding `cluster_col` to compression has negligible impact on the compressed size. This is the **ideal case** for analytic CRSE.
*   **High Cardinality Clustering (e.g., Unit-level in Short Panels):** If $N \approx G$ (e.g., clustering by user in an A/B test), adding `cluster_col` to the group-by effectively disables compression. In this scenario, the **Bootstrap** remains the superior strategy as it allows `duckreg` to compress by covariates first (ignoring clusters) for the point estimate, and then resample from the raw data.

#### 2. Fixed Effects and Degrees of Freedom
When using `DuckMundlak` or `DuckDoubleDemeaning`, the regressors $X$ include generated means.
*   Standard CRSE formulas are generally asymptotically valid for the structural parameters.
*   **Finite Sample Correction:** Ensure the standard correction $c = \frac{G}{G-1} \frac{N-1}{N-K}$ is applied. With compressed data, $N$ is the sum of counts, not the number of compressed rows.

#### 3. Singleton Clusters
Clusters with a single observation (or single stratum) can cause issues in bootstrap resampling (if they are pivotal). Analytic CRSE handles these naturally, though they contribute little to the variance estimation.

## 3. Summary of Recommendations

| Scenario | Recommended Strategy |
| :--- | :--- |
| **IID / Robust Errors** | **Analytic HC1** |
| **Clustered (Few Clusters)** | **Analytic CRSE (Proposed)** |
| **Clustered (Many Clusters)** | **Cluster Bootstrap** |

| Why? |
| :--- |
| Fast, exact, requires minimal compression. |
| Faster than bootstrap, exact. Requires grouping by cluster. |
| Preserves high compression ratio for point estimation. Analytic CRSE would require loading near-raw data. |
