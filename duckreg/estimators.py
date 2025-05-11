import numpy as np
import pandas as pd
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
        event_study: bool = False,
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
        agg_expressions = ["COUNT(*) as count"]
        agg_expressions += [f"SUM({var}) as sum_{var}" for var in self.outcome_vars]
        agg_expressions += [
            f"SUM(POW({var}, 2)) as sum_{var}_sq" for var in self.outcome_vars
        ]
        group_by_cols = ", ".join(self.strata_cols)
        self.agg_query = f"""
        SELECT {group_by_cols}, {", ".join(agg_expressions)}
        FROM {self.table_name}
        GROUP BY {group_by_cols}
        """
        self.df_compressed = pd.DataFrame(self.conn.execute(self.agg_query).fetchall())
        self.df_compressed.columns = (
            self.strata_cols
            + ["count"]
            + [f"sum_{var}" for var in self.outcome_vars]
            + [f"sum_{var}_sq" for var in self.outcome_vars]
        )
        create_means = "\n".join(
            [f"mean_{var} = sum_{var}/count" for var in self.outcome_vars]
        )
        self.df_compressed.eval(create_means, inplace=True)

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
            unique_rows = total_rows
            self.bootstrap_query = f"""
            SELECT {", ".join(self.strata_cols)}, {", ".join(["COUNT(*) as count"] + [f"SUM({var}) as sum_{var}" for var in self.outcome_vars])}
            FROM {self.table_name}
            GROUP BY {", ".join(self.strata_cols)}
            """
        else:
            # Cluster bootstrap
            unique_groups = self.conn.execute(
                f"SELECT DISTINCT {self.cluster_col} FROM {self.table_name}"
            ).fetchall()
            unique_groups = [group[0] for group in unique_groups]
            unique_rows = len(unique_groups)
            self.bootstrap_query = f"""
            SELECT {", ".join(self.strata_cols)}, {", ".join(["COUNT(*) as count"] + [f"SUM({var}) as sum_{var}" for var in self.outcome_vars])}
            FROM {self.table_name}
            WHERE {self.cluster_col} IN (SELECT unnest((?)))
            GROUP BY {", ".join(self.strata_cols)}
            """

        for b in tqdm(range(self.n_bootstraps)):
            resampled_rows = self.rng.choice(
                unique_rows, size=unique_rows, replace=True
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
        self.compress_query = f"""
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
            total_clusters = self.conn.execute(
                f"SELECT COUNT(DISTINCT {self.cluster_col}) FROM {self.table_name}"
            ).fetchone()[0]
            self.bootstrap_query = f"""
            SELECT
                {", ".join([f"{cov}" for cov in self.covariates])},
                {", ".join([f"avg_{cov}_unit" for cov in self.covariates])}
                {", " + ", ".join([f"avg_{cov}_time" for cov in self.covariates]) if self.time_col is not None else ""},
                COUNT(*) as count,
                SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM design_matrix
            WHERE {self.cluster_col} IN (SELECT unnest((?)))
            GROUP BY {", ".join([f"{cov}" for cov in self.covariates])},
                        {", ".join([f"avg_{cov}_unit" for cov in self.covariates])}
                        {", " + ", ".join([f"avg_{cov}_time" for cov in self.covariates]) if self.time_col is not None else ""}
            """
            total_samples = total_clusters

        for b in tqdm(range(self.n_bootstraps)):
            resampled_samples = self.rng.choice(
                total_samples, size=total_samples, replace=True
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
        #   create_cohort_and_ever_treated_columns
        self.temp_table_query = f"""
            CREATE TEMP TABLE temp_{self.table_name} AS
            SELECT *, NULL::INTEGER AS cohort, NULL::INTEGER AS ever_treated
            FROM {self.table_name};
            UPDATE temp_{self.table_name} SET cohort = (
                SELECT MIN({self.time_col})
                FROM {self.table_name} AS p2
                WHERE p2.{self.unit_col} = temp_{self.table_name}.{self.unit_col} AND
                p2.{self.treatment_col} = 1
            );
            UPDATE temp_{self.table_name} SET cohort = NULL WHERE cohort = 2147483647;
            UPDATE temp_{self.table_name} SET ever_treated = CASE WHEN cohort IS NOT NULL THEN 1 ELSE 0 END;
        """
        self.conn.execute(self.temp_table_query)
        #  retrieve_num_periods_and_cohorts
        self.num_periods = self.conn.execute(
            f"SELECT MAX({self.time_col}) FROM temp_{self.table_name}"
        ).fetchone()[0]
        cohorts = self.conn.execute(
            f"SELECT DISTINCT cohort FROM temp_{self.table_name} WHERE cohort IS NOT NULL"
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

        #  create_transformed_query
        self.design_matrix_query = f"""
        CREATE TEMP TABLE transformed_panel_data AS
        SELECT
            p.{self.unit_col},
            p.{self.time_col},
            p.{self.treatment_col},
            p.{self.outcome_var},
            -- Intercept (constant term)
            1 AS intercept,
            -- cohort intercepts
            {self.cohort_intercepts},
            -- Time dummies for each period
            {self.time_dummies},
            -- Treated group interacted with treatment time dummies
            {self.treatment_dummies}
        FROM
            temp_{self.table_name} p;
        """
        self.conn.execute(self.design_matrix_query)

    def compress_data(self):
        self.rhs = f"""
        intercept,
        {", ".join([f"cohort_{cohort}" for cohort in self.cohorts])},
        {", ".join([f"time_{i}" for i in range(self.num_periods + 1)])},
        {", ".join([f"treatment_time_{cohort}_{i}" for cohort in self.cohorts for i in range(self.num_periods + 1)])};
        """
        self.compression_query = f"""
        CREATE TEMP TABLE compressed_panel_data AS
        SELECT
            {self.rhs.replace(";", "")},
            COUNT(*) AS count,
            SUM({self.outcome_var}) AS sum_{self.outcome_var}
        FROM
            transformed_panel_data
        GROUP BY
        {self.rhs}
        """
        self.conn.execute(self.compression_query)
        self.df_compressed = self.conn.execute(
            "SELECT * FROM compressed_panel_data"
        ).fetchdf()
        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )

    def collect_data(self, data):
        self._rhs_list = [x.strip().replace(";", "") for x in self.rhs.split(",")]
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
        total_clusters = self.conn.execute(
            f"SELECT COUNT(DISTINCT {self.cluster_col}) FROM transformed_panel_data"
        ).fetchone()[0]
        boot_coefs = {str(cohort): [] for cohort in self.cohorts}
        # bootstrap loop
        for _ in tqdm(range(self.n_bootstraps)):
            resampled_clusters = (
                self.conn.execute(
                    f"SELECT UNNEST(ARRAY(SELECT {self.cluster_col} FROM transformed_panel_data ORDER BY RANDOM() LIMIT {total_clusters}))"
                )
                .fetchdf()
                .values.flatten()
                .tolist()
            )

            self.conn.execute(
                f"""
                CREATE TEMP TABLE resampled_transformed_panel_data AS
                SELECT * FROM transformed_panel_data
                WHERE {self.cluster_col} IN ({", ".join(map(str, resampled_clusters))})
            """
            )

            self.conn.execute(
                f"""
                CREATE TEMP TABLE resampled_compressed_panel_data AS
                SELECT
                    {self.rhs.replace(";", "")},
                    COUNT(*) AS count,
                    SUM({self.outcome_var}) AS sum_{self.outcome_var}
                FROM
                    resampled_transformed_panel_data
                GROUP BY
                    {self.rhs.replace(";", "")}
            """
            )

            df_boot = self.conn.execute(
                "SELECT * FROM resampled_compressed_panel_data"
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

            self.conn.execute("DROP TABLE resampled_transformed_panel_data")
            self.conn.execute("DROP TABLE resampled_compressed_panel_data")
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

    def estimate_feols(self):
        raise NotImplementedError("feols solver not implemented for double-demeaning")

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
            total_clusters = self.conn.execute(
                f"SELECT COUNT(DISTINCT {self.cluster_col}) FROM {self.table_name}"
            ).fetchone()[0]
            self.bootstrap_query = f"""
            SELECT
                ddot_{self.treatment_var},
                COUNT(*) as count,
                SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM double_demeaned
            WHERE {self.cluster_col} IN (SELECT unnest((?)))
            GROUP BY ddot_{self.treatment_var}
            """

        for b in tqdm(range(self.n_bootstraps)):
            if self.cluster_col is None:
                resampled_units = self.rng.choice(
                    total_units, size=total_units, replace=True
                )
            else:
                resampled_clusters = self.rng.choice(
                    total_clusters, size=total_clusters, replace=True
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
