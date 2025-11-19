import numpy as np
import pandas as pd
from typing import Union
from tqdm import tqdm
from .demean import demean, _convert_to_int
from .duckreg import DuckReg, wls

################################################################################


class DuckRegression(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        formula: str,
        cluster_col: str,
        seed: int,
        n_bootstraps: int = 100,
        rowid_col: str = "rowid",
        fitter: str = "numpy",
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            fitter=fitter,
            **kwargs,
        )
        self.formula = formula
        self.cluster_col = cluster_col
        self.rowid_col = rowid_col
        self._parse_formula()

    def _parse_formula(self):
        lhs, rhs = self.formula.split("~")
        rhs_deparsed = rhs.split("|")
        covars, fevars = rhs.split("|") if len(rhs_deparsed) > 1 else (rhs, None)

        self.outcome_vars = [x.strip() for x in lhs.split("+")]
        self.covars = [x.strip() for x in covars.split("+")]
        self.fevars = [x.strip() for x in fevars.split("+")] if fevars else []
        self.strata_cols = self.covars + self.fevars

        if not self.outcome_vars:
            raise ValueError("No outcome variables found in the formula")

    def prepare_data(self):
        # No preparation needed for simple regression
        pass

    def compress_data(self):
        # Pre-compute expressions once to avoid repeated string operations
        group_by_cols = ", ".join(self.strata_cols)

        # Build aggregation expressions more efficiently
        agg_parts = ["COUNT(*) as count"]
        sum_expressions = []
        sum_sq_expressions = []

        for var in self.outcome_vars:
            sum_expr = f"SUM({var}) as sum_{var}"
            sum_sq_expr = f"SUM(POW({var}, 2)) as sum_{var}_sq"
            sum_expressions.append(sum_expr)
            sum_sq_expressions.append(sum_sq_expr)

        # Single join operation instead of multiple concatenations
        all_agg_expressions = ", ".join(
            agg_parts + sum_expressions + sum_sq_expressions
        )

        self.agg_query = f"""
        SELECT {group_by_cols}, {all_agg_expressions}
        FROM {self.table_name}
        GROUP BY {group_by_cols}
        """

        self.df_compressed = self.conn.execute(self.agg_query).fetchdf()

        # Pre-compute column lists
        sum_cols = [f"sum_{var}" for var in self.outcome_vars]
        sum_sq_cols = [f"sum_{var}_sq" for var in self.outcome_vars]

        self.df_compressed.columns = (
            self.strata_cols + ["count"] + sum_cols + sum_sq_cols
        )

        # Single eval operation for all means
        mean_expressions = [
            f"mean_{var} = sum_{var}/count" for var in self.outcome_vars
        ]
        if mean_expressions:
            self.df_compressed.eval("\n".join(mean_expressions), inplace=True)

    def collect_data(self, data: pd.DataFrame) -> pd.DataFrame:
        y = data.filter(
            regex=f"mean_{'(' + '|'.join(self.outcome_vars) + ')'}", axis=1
        ).values
        X = data[self.covars].values
        n = data["count"].values

        # y, X, w need to be two-dimensional for the demean function
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        if self.fevars:
            # fe needs to contain of only integers for
            # the demean function to work
            fe = _convert_to_int(data[self.fevars])
            fe = fe.reshape(-1, 1) if fe.ndim == 1 else fe

            y, _ = demean(x=y, flist=fe, weights=n)
            X, _ = demean(x=X, flist=fe, weights=n)
        else:
            X = np.c_[np.ones(X.shape[0]), X]

        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        betahat = wls(X, y, n).flatten()
        return betahat

    def fit_vcov(self):
        """compressed estimation of the heteroskedasticity-robust variance covariance matrix"""
        self.se = "hc1"
        y, X, n = self.collect_data(data=self.df_compressed)
        betahat = wls(X, y, n).flatten()
        # only works for single outcome for now
        self.n_bootstraps = 0  # disable bootstrap
        yprime = self.df_compressed[f"sum_{self.outcome_vars[0]}"].values.reshape(-1, 1)
        yprimeprime = self.df_compressed[
            f"sum_{self.outcome_vars[0]}_sq"
        ].values.reshape(-1, 1)
        yhat = (X @ betahat).reshape(-1, 1)
        rss_g = (yhat**2) * n.reshape(-1, 1) - 2 * yhat * yprime + yprimeprime
        bread = np.linalg.inv(X.T @ np.diag(n.flatten()) @ X)
        meat = X.T @ np.diag(rss_g.flatten()) @ X
        n_nk = n.sum() / (n.sum() - X.shape[1])
        self.vcov = n_nk * (bread @ meat @ bread)

    def bootstrap(self):
        self.se = "bootstrap"
        if self.fevars:
            boot_coefs = np.zeros(
                (self.n_bootstraps, len(self.covars) * len(self.outcome_vars))
            )
        else:
            boot_coefs = np.zeros(
                (
                    self.n_bootstraps,
                    (len(self.strata_cols) + 1) * len(self.outcome_vars),
                )
            )

        if not self.cluster_col:
            # IID bootstrap
            total_rows = self.conn.execute(
                f"SELECT COUNT(DISTINCT {self.rowid_col}) FROM {self.table_name}"
            ).fetchone()[0]
            # unique_rows = total_rows
            unique_groups = np.arange(total_rows)  # Add this line
            self.bootstrap_query = f"""
            SELECT {", ".join(self.strata_cols)}, {", ".join(["COUNT(*) as count"] + [f"SUM({var}) as sum_{var}" for var in self.outcome_vars])}
            FROM {self.table_name}
            GROUP BY {", ".join(self.strata_cols)}
            """
        else:
            # Cluster bootstrap - FIX
            unique_groups = self.conn.execute(
                f"SELECT DISTINCT {self.cluster_col} FROM {self.table_name}"
            ).fetchall()
            unique_groups = [group[0] for group in unique_groups]
            self.bootstrap_query = f"""
            WITH resampled AS (
                SELECT cluster_id, COUNT(*) as mult
                FROM (SELECT unnest(?) as cluster_id)
                GROUP BY cluster_id
            ),
            grouped_data AS (
                SELECT {", ".join(self.strata_cols)}, {self.cluster_col},
                    COUNT(*) as count,
                    {", ".join([f"SUM({var}) as sum_{var}" for var in self.outcome_vars])}
                FROM {self.table_name}
                GROUP BY {", ".join(self.strata_cols)}, {self.cluster_col}
            )
            SELECT {", ".join(self.strata_cols)},
                SUM(gd.count * r.mult) as count,
                {", ".join([f"SUM(gd.sum_{var} * r.mult) as sum_{var}" for var in self.outcome_vars])}
            FROM grouped_data gd
            JOIN resampled r ON gd.{self.cluster_col} = r.cluster_id
            GROUP BY {", ".join(self.strata_cols)}
            """

        for b in tqdm(range(self.n_bootstraps)):
            resampled_rows = self.rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            df_boot = pd.DataFrame(
                self.conn.execute(
                    self.bootstrap_query, [resampled_rows.tolist()]
                ).fetchall()
            )
            df_boot.columns = (
                self.strata_cols
                + ["count"]
                + [f"sum_{var}" for var in self.outcome_vars]
            )
            create_means = "\n".join(
                [f"mean_{var} = sum_{var}/count" for var in self.outcome_vars]
            )
            df_boot.eval(create_means, inplace=True)

            y, X, n = self.collect_data(data=df_boot)

            boot_coefs[b, :] = wls(X, y, n).flatten()

            # else np.diag() fails if input is not at least 1-dim
            vcov = np.cov(boot_coefs.T)
            vcov = np.expand_dims(vcov, axis=0) if vcov.ndim == 0 else vcov

        return vcov

    def summary(
        self,
    ):  # ovveride the summary method to include the heteroskedasticity-robust variance covariance matrix when available
        if self.n_bootstraps > 0 or (hasattr(self, "se") and self.se == "hc1"):
            return {
                "point_estimate": self.point_estimate,
                "standard_error": np.sqrt(np.diag(self.vcov)),
            }
        return {"point_estimate": self.point_estimate}


################################################################################


class DuckDML(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_var: str,
        treatment_var: Union[str, list[str]],
        discrete_covars: list[str],
        seed: int,
        n_bootstraps: int = 200,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            **kwargs,
        )
        self.outcome_var = outcome_var
        if isinstance(treatment_var, str):
            self.treatment_vars = [treatment_var]
        else:
            self.treatment_vars = treatment_var
        self.discrete_covars = discrete_covars

    def prepare_data(self):
        pass

    def collect_data(self):
        pass

    def compress_data(self):
        y = self.outcome_var
        x_vars = self.treatment_vars
        group_by_cols = ", ".join(self.discrete_covars)

        # Aggregations
        aggs = [
            f"COUNT(*) as n_g",
            f"SUM({y}) as sum_y",
        ]
        # Sums of X and XY
        for x in x_vars:
            aggs.append(f"SUM({x}) as sum_{x}")
            aggs.append(f"SUM({y} * {x}) as sum_{self.outcome_var}_{x}")

        # Sums of Cross-products XX
        for i, x1 in enumerate(x_vars):
            for x2 in x_vars[i:]:
                aggs.append(f"SUM({x1} * {x2}) as sum_{x1}_{x2}")

        self.agg_query = f"""
        SELECT
            {group_by_cols},
            {", ".join(aggs)}
        FROM {self.table_name}
        GROUP BY {group_by_cols}
        HAVING COUNT(*) > 1
        """
        self.df_compressed = self.conn.execute(self.agg_query).fetchdf()

    def _calculate_beta_from_compressed(self, data: pd.DataFrame) -> np.ndarray:
        if data.empty:
            return np.full(len(self.treatment_vars), np.nan)

        df = data
        n_treat = len(self.treatment_vars)

        n_g = df["n_g"].values
        # Apply the N_g / (N_g - 1)^2 scaling factor from the LOO formula
        weight = n_g / (n_g - 1) ** 2

        # Construct S_X (n_groups x n_treat)
        S_X = np.stack([df[f"sum_{x}"] for x in self.treatment_vars], axis=1)

        # Construct S_Y (n_groups x 1)
        S_Y = df[f"sum_y"].values.reshape(-1, 1)

        # Construct S_XX (n_groups x n_treat x n_treat)
        S_XX = np.zeros((len(df), n_treat, n_treat))
        for i, x1 in enumerate(self.treatment_vars):
            for j, x2 in enumerate(self.treatment_vars):
                if j >= i:
                    col = f"sum_{x1}_{x2}"
                    # If we computed only upper triangular, we assume we can find it.
                    S_XX[:, i, j] = df[col]
                    S_XX[:, j, i] = df[col]

        # Construct S_XY (n_groups x n_treat x 1)
        S_XY = np.zeros((len(df), n_treat, 1))
        for i, x in enumerate(self.treatment_vars):
            col = f"sum_{self.outcome_var}_{x}"
            S_XY[:, i, 0] = df[col]

        # Broadcasting weights and n_g
        w_reshaped = weight.reshape(-1, 1, 1)
        n_g_reshaped = n_g.reshape(-1, 1, 1)

        # Numerator (Total XTY): Sum over g of w * [n_g * S_XY - S_X * S_Y]
        S_X_expanded = S_X.reshape(len(df), n_treat, 1)
        S_Y_expanded = S_Y.reshape(len(df), 1, 1)
        S_X_S_Y = S_X_expanded * S_Y_expanded

        XTY_g = w_reshaped * (n_g_reshaped * S_XY - S_X_S_Y)
        total_XTY = XTY_g.sum(axis=0)  # (n_treat, 1)

        # Denominator (Total XTX): Sum over g of w * [n_g * S_XX - S_X * S_X^T]
        S_X_outer = np.einsum("bi,bj->bij", S_X, S_X)

        XTX_g = w_reshaped * (n_g_reshaped * S_XX - S_X_outer)
        total_XTX = XTX_g.sum(axis=0)  # (n_treat, n_treat)

        try:
            beta_hat = np.linalg.solve(total_XTX, total_XTY).flatten()
        except np.linalg.LinAlgError:
            beta_hat = np.full(n_treat, np.nan)

        return beta_hat

    def estimate(self):
        return self._calculate_beta_from_compressed(self.df_compressed)

    def bootstrap(self):
        """
        Performs a cluster bootstrap by resampling the compressed groups.
        This is equivalent to resampling clusters defined by unique combinations
        of the discrete covariates.
        """
        n_groups = len(self.df_compressed)
        n_treat = len(self.treatment_vars)
        boot_coefs = np.zeros((self.n_bootstraps, n_treat))

        for b in tqdm(range(self.n_bootstraps), desc="Bootstrapping"):
            resampled_indices = self.rng.choice(n_groups, size=n_groups, replace=True)
            df_boot = self.df_compressed.iloc[resampled_indices]
            boot_coefs[b, :] = self._calculate_beta_from_compressed(df_boot)

        self.vcov = np.cov(boot_coefs, rowvar=False)
        # ensure vcov is 2D
        if self.vcov.ndim == 0:
            self.vcov = np.expand_dims(self.vcov, axis=0)
        return self.vcov



################################################################################
class DuckMundlak(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_var: str,
        covariates: list,
        seed: int,
        unit_col: str,
        time_col: str = None,
        n_bootstraps: int = 100,
        cluster_col: str = None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            **kwargs,
        )
        self.outcome_var = outcome_var
        self.covariates = covariates
        self.unit_col = unit_col
        self.time_col = time_col
        self.cluster_col = cluster_col

    def prepare_data(self):
        # Step 1: Compute unit averages
        self.unit_avg_query = f"""
        CREATE TEMP TABLE unit_avgs AS
        SELECT {self.unit_col},
               {", ".join([f"AVG({cov}) AS avg_{cov}_unit" for cov in self.covariates])}
        FROM {self.table_name}
        GROUP BY {self.unit_col}
        """
        self.conn.execute(self.unit_avg_query)

        # Step 2: Compute time averages (only if time_col is provided)
        if self.time_col is not None:
            self.time_avg_query = f"""
            CREATE TEMP TABLE time_avgs AS
            SELECT {self.time_col},
                   {", ".join([f"AVG({cov}) AS avg_{cov}_time" for cov in self.covariates])}
            FROM {self.table_name}
            GROUP BY {self.time_col}
            """
            self.conn.execute(self.time_avg_query)

        # Step 3: Create the design matrix
        self.design_matrix_query = f"""
        CREATE TEMP TABLE design_matrix AS
        SELECT
            t.{self.unit_col},
            {f"t.{self.time_col}," if self.time_col is not None else ""}
            {f"t.{self.cluster_col}," if self.cluster_col and self.cluster_col != self.unit_col else ""}
            t.{self.outcome_var},
            {", ".join([f"t.{cov}" for cov in self.covariates])},
            {", ".join([f"u.avg_{cov}_unit" for cov in self.covariates])}
            {", " + ", ".join([f"tm.avg_{cov}_time" for cov in self.covariates]) if self.time_col is not None else ""}
        FROM {self.table_name} t
        JOIN unit_avgs u ON t.{self.unit_col} = u.{self.unit_col}
        {f"JOIN time_avgs tm ON t.{self.time_col} = tm.{self.time_col}" if self.time_col is not None else ""}
        """
        self.conn.execute(self.design_matrix_query)

    def compress_data(self):
        # Pre-compute column lists to avoid repeated operations
        cov_cols = [f"{cov}" for cov in self.covariates]
        unit_avg_cols = [f"avg_{cov}_unit" for cov in self.covariates]
        time_avg_cols = (
            [f"avg_{cov}_time" for cov in self.covariates]
            if self.time_col is not None
            else []
        )

        # Build SELECT and GROUP BY columns once
        select_cols = cov_cols + unit_avg_cols + time_avg_cols
        select_clause = ", ".join(select_cols)
        group_by_clause = ", ".join(select_cols)

        self.compress_query = f"""
        SELECT
            {select_clause},
            COUNT(*) as count,
            SUM({self.outcome_var}) as sum_{self.outcome_var}
        FROM design_matrix
        GROUP BY {group_by_clause}
        """
        self.df_compressed = self.conn.execute(self.compress_query).fetchdf()

        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )

    def collect_data(self, data: pd.DataFrame):
        rhs = (
            self.covariates
            + [f"avg_{cov}_unit" for cov in self.covariates]
            + (
                [f"avg_{cov}_time" for cov in self.covariates]
                if self.time_col is not None
                else []
            )
        )

        X = data[rhs].values
        X = np.c_[np.ones(X.shape[0]), X]
        y = data[f"mean_{self.outcome_var}"].values
        n = data["count"].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        return wls(X, y, n)

    def bootstrap(self):
        rhs = (
            self.covariates
            + [f"avg_{cov}_unit" for cov in self.covariates]
            + (
                [f"avg_{cov}_time" for cov in self.covariates]
                if self.time_col is not None
                else []
            )
        )
        boot_coefs = np.zeros((self.n_bootstraps, len(rhs) + 1))

        if self.cluster_col is None:
            # IID bootstrap
            total_units = self.conn.execute(
                f"SELECT COUNT(DISTINCT {self.unit_col}) FROM {self.table_name}"
            ).fetchone()[0]
            self.bootstrap_query = f"""
            SELECT
                {", ".join([f"{cov}" for cov in self.covariates])},
                {", ".join([f"avg_{cov}_unit" for cov in self.covariates])}
                {", " + ", ".join([f"avg_{cov}_time" for cov in self.covariates]) if self.time_col is not None else ""},
                COUNT(*) as count,
                SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM design_matrix
            GROUP BY {", ".join([f"{cov}" for cov in self.covariates])},
                        {", ".join([f"avg_{cov}_unit" for cov in self.covariates])}
                        {", " + ", ".join([f"avg_{cov}_time" for cov in self.covariates]) if self.time_col is not None else ""}
            """
            total_samples = total_units
        else:
            # Cluster bootstrap
            unique_clusters = self.conn.execute(
                f"SELECT DISTINCT {self.cluster_col} FROM {self.table_name}"
            ).fetchall()
            unique_clusters = [c[0] for c in unique_clusters]

            self.bootstrap_query = f"""
            WITH resampled AS (
                SELECT cluster_id, COUNT(*) as mult
                FROM (SELECT unnest(?) as cluster_id)
                GROUP BY cluster_id
            ),
            grouped_data AS (
                SELECT
                    {", ".join([f"{cov}" for cov in self.covariates])},
                    {", ".join([f"avg_{cov}_unit" for cov in self.covariates])}
                    {", " + ", ".join([f"avg_{cov}_time" for cov in self.covariates]) if self.time_col is not None else ""},
                    {self.cluster_col},
                    COUNT(*) as count,
                    SUM({self.outcome_var}) as sum_{self.outcome_var}
                FROM design_matrix
                GROUP BY {", ".join([f"{cov}" for cov in self.covariates])},
                         {", ".join([f"avg_{cov}_unit" for cov in self.covariates])}
                         {", " + ", ".join([f"avg_{cov}_time" for cov in self.covariates]) if self.time_col is not None else ""},
                         {self.cluster_col}
            )
            SELECT
                {", ".join([f"{cov}" for cov in self.covariates])},
                {", ".join([f"avg_{cov}_unit" for cov in self.covariates])}
                {", " + ", ".join([f"avg_{cov}_time" for cov in self.covariates]) if self.time_col is not None else ""},
                SUM(gd.count * r.mult) as count,
                SUM(gd.sum_{self.outcome_var} * r.mult) as sum_{self.outcome_var}
            FROM grouped_data gd
            JOIN resampled r ON gd.{self.cluster_col} = r.cluster_id
            GROUP BY {", ".join([f"{cov}" for cov in self.covariates])},
                     {", ".join([f"avg_{cov}_unit" for cov in self.covariates])}
                     {", " + ", ".join([f"avg_{cov}_time" for cov in self.covariates]) if self.time_col is not None else ""}
            """
            total_samples = unique_clusters

        for b in tqdm(range(self.n_bootstraps)):
            resampled_samples = self.rng.choice(
                total_samples, size=len(total_samples), replace=True
            )
            df_boot = self.conn.execute(
                self.bootstrap_query, [resampled_samples.tolist()]
            ).fetchdf()
            df_boot[f"mean_{self.outcome_var}"] = (
                df_boot[f"sum_{self.outcome_var}"] / df_boot["count"]
            )

            y, X, n = self.collect_data(data=df_boot)

            boot_coefs[b, :] = wls(X, y, n).flatten()

        return np.cov(boot_coefs.T)


################################################################################
class DuckMundlakEventStudy(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_var: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        cluster_col: str,
        pre_treat_interactions: bool = True,
        n_bootstraps: int = 100,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            n_bootstraps=n_bootstraps,
            **kwargs,
        )
        self.table_name = table_name
        self.outcome_var = outcome_var
        self.treatment_col = treatment_col
        self.unit_col = unit_col
        self.time_col = time_col
        self.num_periods = None
        self.cohorts = None
        self.time_dummies = None
        self.post_treatment_dummies = None
        self.transformed_query = None
        self.compression_query = None
        self.cluster_col = cluster_col
        self.pre_treat_interactions = pre_treat_interactions

    def prepare_data(self):
        # Create cohort data using CTE instead of temp table
        self.cohort_cte = f"""
        WITH cohort_data AS (
            SELECT *,
                   CASE WHEN cohort_min = 2147483647 THEN NULL ELSE cohort_min END as cohort,
                   CASE WHEN cohort_min IS NOT NULL AND cohort_min != 2147483647 THEN 1 ELSE 0 END as ever_treated
            FROM (
                SELECT *,
                       (SELECT MIN({self.time_col})
                        FROM {self.table_name} AS p2
                        WHERE p2.{self.unit_col} = p1.{self.unit_col} AND p2.{self.treatment_col} = 1
                       ) as cohort_min
                FROM {self.table_name} p1
            )
        )
        """
        #  retrieve_num_periods_and_cohorts using CTE
        self.num_periods = self.conn.execute(
            f"{self.cohort_cte} SELECT MAX({self.time_col}) FROM cohort_data"
        ).fetchone()[0]
        cohorts = self.conn.execute(
            f"{self.cohort_cte} SELECT DISTINCT cohort FROM cohort_data WHERE cohort IS NOT NULL"
        ).fetchall()
        self.cohorts = [row[0] for row in cohorts]
        # generate_time_dummies
        self.time_dummies = ",\n".join(
            [
                f"CASE WHEN {self.time_col} = {i} THEN 1 ELSE 0 END AS time_{i}"
                for i in range(self.num_periods + 1)
            ]
        )
        # generate cohort dummies
        cohort_intercepts = []
        for cohort in self.cohorts:
            cohort_intercepts.append(
                f"CASE WHEN cohort = {cohort} THEN 1 ELSE 0 END AS cohort_{cohort}"
            )
        self.cohort_intercepts = ",\n".join(cohort_intercepts)

        # generate_treatment_dummies
        treatment_dummies = []
        for cohort in self.cohorts:
            for i in range(self.num_periods + 1):
                treatment_dummies.append(
                    f"""CASE WHEN cohort = {cohort} AND
                        {self.time_col} = {i}
                        {f"AND {self.treatment_col} == 1" if not self.pre_treat_interactions else ""}
                        THEN 1 ELSE 0 END AS treatment_time_{cohort}_{i}"""
                )
        self.treatment_dummies = ",\n".join(treatment_dummies)

        #  create_transformed_query using CTE instead of temp table
        self.design_matrix_cte = f"""
        {self.cohort_cte},
        transformed_panel_data AS (
            SELECT
                p.{self.unit_col},
                p.{self.time_col},
                p.{self.treatment_col},
                p.{self.outcome_var},
                {f"p.{self.cluster_col}," if self.cluster_col != self.unit_col else ""}
                -- Intercept (constant term)
                1 AS intercept,
                -- cohort intercepts
                {self.cohort_intercepts},
                -- Time dummies for each period
                {self.time_dummies},
                -- Treated group interacted with treatment time dummies
                {self.treatment_dummies}
            FROM cohort_data p
        )
        """

    def compress_data(self):
        # Pre-compute RHS columns to avoid repeated string operations
        cohort_cols = [f"cohort_{cohort}" for cohort in self.cohorts]
        time_cols = [f"time_{i}" for i in range(self.num_periods + 1)]
        treatment_cols = [
            f"treatment_time_{cohort}_{i}"
            for cohort in self.cohorts
            for i in range(self.num_periods + 1)
        ]

        rhs_cols = ["intercept"] + cohort_cols + time_cols + treatment_cols
        rhs_clause = ", ".join(rhs_cols)

        # Use single query with CTE instead of temp table
        self.compression_query = f"""
        {self.design_matrix_cte}
        SELECT
            {rhs_clause},
            COUNT(*) AS count,
            SUM({self.outcome_var}) AS sum_{self.outcome_var}
        FROM transformed_panel_data
        GROUP BY {rhs_clause}
        """

        self.df_compressed = self.conn.execute(self.compression_query).fetchdf()
        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )

        # Store for later use
        self.rhs_cols = rhs_cols

    def collect_data(self, data):
        self._rhs_list = self.rhs_cols
        X = data[self._rhs_list].values
        y = data[f"mean_{self.outcome_var}"].values
        n = data["count"].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        coef = wls(X, y, n)
        res = pd.DataFrame(
            {
                "est": coef.squeeze(),
            },
            index=self._rhs_list,
        )
        cohort_names = [x.split("_")[1] for x in self._rhs_list if "cohort_" in x]
        event_study_coefs = {}
        for c in cohort_names:
            offset = res.filter(regex=f"^cohort_{c}", axis=0).values
            event_study_coefs[c] = (
                res.filter(regex=f"treatment_time_{c}_", axis=0) + offset
            )

        return event_study_coefs

    def bootstrap(self):
        # list all clusters
        unique_clusters = self.conn.execute(
            f"{self.design_matrix_cte} SELECT DISTINCT {self.cluster_col} FROM transformed_panel_data"
        ).fetchall()
        unique_clusters = [c[0] for c in unique_clusters]

        boot_coefs = {str(cohort): [] for cohort in self.cohorts}
        
        rhs_clause = ", ".join(self.rhs_cols)
        
        self.bootstrap_query = f"""
        {self.design_matrix_cte},
        resampled AS (
            SELECT cluster_id, COUNT(*) as mult
            FROM (SELECT unnest(?) as cluster_id)
            GROUP BY cluster_id
        ),
        grouped_data AS (
            SELECT
                {rhs_clause}, {self.cluster_col},
                COUNT(*) as count,
                SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM transformed_panel_data
            GROUP BY {rhs_clause}, {self.cluster_col}
        )
        SELECT
            {rhs_clause},
            SUM(gd.count * r.mult) as count,
            SUM(gd.sum_{self.outcome_var} * r.mult) as sum_{self.outcome_var}
        FROM grouped_data gd
        JOIN resampled r ON gd.{self.cluster_col} = r.cluster_id
        GROUP BY {rhs_clause}
        """

        # bootstrap loop
        for _ in tqdm(range(self.n_bootstraps)):
            resampled_clusters = self.rng.choice(
                unique_clusters, size=len(unique_clusters), replace=True
            )
            
            df_boot = self.conn.execute(
                self.bootstrap_query, [resampled_clusters.tolist()]
            ).fetchdf()
            
            df_boot[f"mean_{self.outcome_var}"] = (
                df_boot[f"sum_{self.outcome_var}"] / df_boot["count"]
            )

            y, X, n = self.collect_data(data=df_boot)
            coef = wls(X, y, n)
            res = pd.DataFrame(
                {
                    "est": coef.squeeze(),
                },
                index=self._rhs_list,
            )
            cohort_names = [x.split("_")[1] for x in self._rhs_list if "cohort_" in x]
            for c in cohort_names:
                offset = res.filter(regex=f"^cohort_{c}", axis=0).values
                event_study_coefs = (
                    res.filter(regex=f"treatment_time_{c}_", axis=0) + offset
                )
                boot_coefs[c].append(event_study_coefs.values.flatten())

        # Calculate the covariance matrix for each cohort
        bootstrap_cov_matrix = {
            cohort: np.cov(np.array(coefs).T) for cohort, coefs in boot_coefs.items()
        }
        return bootstrap_cov_matrix

    def summary(self) -> dict:
        """Summary of event study regression (overrides the parent class method)

        Returns:
            dict of event study coefficients and their standard errors
        """
        if self.n_bootstraps > 0:
            summary_tables = {}
            for c in self.point_estimate.keys():
                point_estimate = self.point_estimate[c]
                se = np.sqrt(np.diag(self.vcov[c]))
                summary_tables[c] = pd.DataFrame(
                    np.c_[point_estimate, se],
                    columns=["point_estimate", "se"],
                    index=point_estimate.index,
                )
            return summary_tables
        return {"point_estimate": self.point_estimate}


################################################################################
class DuckDoubleDemeaning(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_var: str,
        treatment_var: str,
        unit_col: str,
        time_col: str,
        seed: int,
        n_bootstraps: int = 100,
        cluster_col: str = None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            **kwargs,
        )
        self.outcome_var = outcome_var
        self.treatment_var = treatment_var
        self.unit_col = unit_col
        self.time_col = time_col
        self.cluster_col = cluster_col

    def prepare_data(self):
        # Compute overall mean
        self.overall_mean_query = f"""
        CREATE TEMP TABLE overall_mean AS
        SELECT AVG({self.treatment_var}) AS mean_{self.treatment_var}
        FROM {self.table_name}
        """
        self.conn.execute(self.overall_mean_query)

        # Compute unit means
        self.unit_mean_query = f"""
        CREATE TEMP TABLE unit_means AS
        SELECT {self.unit_col}, AVG({self.treatment_var}) AS mean_{self.treatment_var}_unit
        FROM {self.table_name}
        GROUP BY {self.unit_col}
        """
        self.conn.execute(self.unit_mean_query)

        # Compute time means
        self.time_mean_query = f"""
        CREATE TEMP TABLE time_means AS
        SELECT {self.time_col}, AVG({self.treatment_var}) AS mean_{self.treatment_var}_time
        FROM {self.table_name}
        GROUP BY {self.time_col}
        """
        self.conn.execute(self.time_mean_query)

        # Create double-demeaned variables
        self.double_demean_query = f"""
        CREATE TEMP TABLE double_demeaned AS
        SELECT
            t.{self.unit_col},
            t.{self.time_col},
            {f"t.{self.cluster_col}," if self.cluster_col and self.cluster_col != self.unit_col else ""}
            t.{self.outcome_var},
            t.{self.treatment_var} - um.mean_{self.treatment_var}_unit - tm.mean_{self.treatment_var}_time + om.mean_{self.treatment_var} AS ddot_{self.treatment_var}
        FROM {self.table_name} t
        JOIN unit_means um ON t.{self.unit_col} = um.{self.unit_col}
        JOIN time_means tm ON t.{self.time_col} = tm.{self.time_col}
        CROSS JOIN overall_mean om
        """
        self.conn.execute(self.double_demean_query)

    def compress_data(self):
        self.compress_query = f"""
        SELECT
            ddot_{self.treatment_var},
            COUNT(*) as count,
            SUM({self.outcome_var}) as sum_{self.outcome_var}
        FROM double_demeaned
        GROUP BY ddot_{self.treatment_var}
        """
        self.df_compressed = self.conn.execute(self.compress_query).fetchdf()
        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )

    def collect_data(self, data: pd.DataFrame):
        X = data[f"ddot_{self.treatment_var}"].values
        X = np.c_[np.ones(X.shape[0]), X]
        y = data[f"mean_{self.outcome_var}"].values
        n = data["count"].values
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        return wls(X, y, n)

    def bootstrap(self):
        boot_coefs = np.zeros((self.n_bootstraps, 2))  # Intercept and treatment effect

        if self.cluster_col is None:
            total_units = self.conn.execute(
                f"SELECT COUNT(DISTINCT {self.unit_col}) FROM {self.table_name}"
            ).fetchone()[0]
            self.bootstrap_query = f"""
            SELECT
                ddot_{self.treatment_var},
                COUNT(*) as count,
                SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM double_demeaned
            WHERE {self.unit_col} IN (SELECT unnest((?)))
            GROUP BY ddot_{self.treatment_var}
            """
        else:
            unique_clusters = self.conn.execute(
                f"SELECT DISTINCT {self.cluster_col} FROM {self.table_name}"
            ).fetchall()
            unique_clusters = [c[0] for c in unique_clusters]
            self.bootstrap_query = f"""
            WITH resampled AS (
                SELECT cluster_id, COUNT(*) as mult
                FROM (SELECT unnest(?) as cluster_id)
                GROUP BY cluster_id
            ),
            grouped_data AS (
                SELECT
                    ddot_{self.treatment_var}, {self.cluster_col},
                    COUNT(*) as count,
                    SUM({self.outcome_var}) as sum_{self.outcome_var}
                FROM double_demeaned
                GROUP BY ddot_{self.treatment_var}, {self.cluster_col}
            )
            SELECT
                ddot_{self.treatment_var},
                SUM(gd.count * r.mult) as count,
                SUM(gd.sum_{self.outcome_var} * r.mult) as sum_{self.outcome_var}
            FROM grouped_data gd
            JOIN resampled r ON gd.{self.cluster_col} = r.cluster_id
            GROUP BY ddot_{self.treatment_var}
            """

        for b in tqdm(range(self.n_bootstraps)):
            if self.cluster_col is None:
                resampled_units = self.rng.choice(
                    total_units, size=total_units, replace=True
                )
            else:
                resampled_clusters = self.rng.choice(
                    unique_clusters, size=len(unique_clusters), replace=True
                )
                resampled_units = resampled_clusters

            df_boot = self.conn.execute(
                self.bootstrap_query, [resampled_units.tolist()]
            ).fetchdf()
            df_boot[f"mean_{self.outcome_var}"] = (
                df_boot[f"sum_{self.outcome_var}"] / df_boot["count"]
            )

            y, X, n = self.collect_data(data=df_boot)

            boot_coefs[b, :] = wls(X, y, n).flatten()

        return np.cov(boot_coefs.T)


################################################################################
