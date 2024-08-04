import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from .demean import demean
from .duckreg import DuckReg, wls


################################################################################


class DuckRegression(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        formula: str,
        cluster_col: str,
        n_bootstraps: int = 100,
        seed: int = 42,
        rowid_col: str = "rowid",
    ):
        super().__init__(db_name, table_name, n_bootstraps, seed)
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
        agg_expressions = ["COUNT(*) as count"] + [
            f"SUM({var}) as sum_{var}" for var in self.outcome_vars
        ]
        group_by_cols = ", ".join(self.strata_cols)
        self.agg_query = f"""
        SELECT {group_by_cols}, {', '.join(agg_expressions)}
        FROM {self.table_name}
        GROUP BY {group_by_cols}
        """
        self.df_compressed = pd.DataFrame(self.conn.execute(self.agg_query).fetchall())
        self.df_compressed.columns = (
            self.strata_cols + ["count"] + [f"sum_{var}" for var in self.outcome_vars]
        )
        create_means = "\n".join(
            [f"mean_{var} = sum_{var}/count" for var in self.outcome_vars]
        )
        self.df_compressed.eval(create_means, inplace=True)

    def collect_and_demean(self, data: pd.DataFrame) -> pd.DataFrame:

        y = data[f"mean_{self.outcome_vars[0]}"].values
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

        y, X, n = self.collect_and_demean(data=self.df_compressed)

        return wls(X, y, n)

    def bootstrap(self):
        if self.fevars:
            boot_coefs = np.zeros((self.n_bootstraps, len(self.covars)))
        else:
            boot_coefs = np.zeros((self.n_bootstraps, len(self.strata_cols) + 1))

        if not self.cluster_col:
            # IID bootstrap
            total_rows = self.conn.execute(
                f"SELECT COUNT(DISTINCT {self.rowid_col}) FROM {self.table_name}"
            ).fetchone()[0]
            unique_rows = total_rows
            self.bootstrap_query = f"""
            SELECT {', '.join(self.strata_cols)}, {', '.join(["COUNT(*) as count"] + [f"SUM({var}) as sum_{var}" for var in self.outcome_vars])}
            FROM {self.table_name}
            GROUP BY {', '.join(self.strata_cols)}
            """
        else:
            # Cluster bootstrap
            unique_groups = self.conn.execute(
                f"SELECT DISTINCT {self.cluster_col} FROM {self.table_name}"
            ).fetchall()
            unique_groups = [group[0] for group in unique_groups]
            unique_rows = len(unique_groups)
            self.bootstrap_query = f"""
            SELECT {', '.join(self.strata_cols)}, {', '.join(["COUNT(*) as count"] + [f"SUM({var}) as sum_{var}" for var in self.outcome_vars])}
            FROM {self.table_name}
            WHERE {self.cluster_col} IN (SELECT unnest((?)))
            GROUP BY {', '.join(self.strata_cols)}
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

            y, X, n = self.collect_and_demean(data=df_boot)

            boot_coefs[b, :] = wls(X, y, n).flatten()

            # else np.diag() fails if input is not at least 1-dim
            vcov = np.cov(boot_coefs.T)
            vcov = np.expand_dims(vcov, axis=0) if vcov.ndim == 0 else vcov

        return vcov


################################################################################


class DuckMundlak(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_var: str,
        covariates: list,
        unit_col: str,
        time_col: str = None,
        n_bootstraps: int = 100,
        seed: int = 42,
        cluster_col: str = None,
    ):
        super().__init__(db_name, table_name, n_bootstraps, seed)
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
               {', '.join([f'AVG({cov}) AS avg_{cov}_unit' for cov in self.covariates])}
        FROM {self.table_name}
        GROUP BY {self.unit_col}
        """
        self.conn.execute(self.unit_avg_query)

        # Step 2: Compute time averages (only if time_col is provided)
        if self.time_col is not None:
            self.time_avg_query = f"""
            CREATE TEMP TABLE time_avgs AS
            SELECT {self.time_col},
                   {', '.join([f'AVG({cov}) AS avg_{cov}_time' for cov in self.covariates])}
            FROM {self.table_name}
            GROUP BY {self.time_col}
            """
            self.conn.execute(self.time_avg_query)

        # Step 3: Create the design matrix
        self.design_matrix_query = f"""
        CREATE TEMP TABLE design_matrix AS
        SELECT
            t.{self.unit_col},
            {f't.{self.time_col},' if self.time_col is not None else ''}
            t.{self.outcome_var},
            {', '.join([f't.{cov}' for cov in self.covariates])},
            {', '.join([f'u.avg_{cov}_unit' for cov in self.covariates])}
            {', ' + ', '.join([f'tm.avg_{cov}_time' for cov in self.covariates]) if self.time_col is not None else ''}
        FROM {self.table_name} t
        JOIN unit_avgs u ON t.{self.unit_col} = u.{self.unit_col}
        {f"JOIN time_avgs tm ON t.{self.time_col} = tm.{self.time_col}" if self.time_col is not None else ""}
        """
        self.conn.execute(self.design_matrix_query)

    def compress_data(self):
        self.compress_query = f"""
        SELECT
            {', '.join([f'{cov}' for cov in self.covariates])},
            {', '.join([f'avg_{cov}_unit' for cov in self.covariates])}
            {', ' + ', '.join([f'avg_{cov}_time' for cov in self.covariates]) if self.time_col is not None else ''},
            COUNT(*) as count,
            SUM({self.outcome_var}) as sum_{self.outcome_var}
        FROM design_matrix
        GROUP BY {', '.join([f'{cov}' for cov in self.covariates])},
                    {', '.join([f'avg_{cov}_unit' for cov in self.covariates])}
                    {', ' + ', '.join([f'avg_{cov}_time' for cov in self.covariates]) if self.time_col is not None else ''}
        """
        self.df_compressed = self.conn.execute(self.compress_query).fetchdf()
        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )

    def estimate(self):
        rhs = (
            self.covariates
            + [f"avg_{cov}_unit" for cov in self.covariates]
            + (
                [f"avg_{cov}_time" for cov in self.covariates]
                if self.time_col is not None
                else []
            )
        )
        X = self.df_compressed[rhs].values
        X = np.c_[np.ones(X.shape[0]), X]
        y = self.df_compressed[f"mean_{self.outcome_var}"].values
        n = self.df_compressed["count"].values
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
                {', '.join([f'{cov}' for cov in self.covariates])},
                {', '.join([f'avg_{cov}_unit' for cov in self.covariates])}
                {', ' + ', '.join([f'avg_{cov}_time' for cov in self.covariates]) if self.time_col is not None else ''},
                COUNT(*) as count,
                SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM design_matrix
            GROUP BY {', '.join([f'{cov}' for cov in self.covariates])},
                        {', '.join([f'avg_{cov}_unit' for cov in self.covariates])}
                        {', ' + ', '.join([f'avg_{cov}_time' for cov in self.covariates]) if self.time_col is not None else ''}
            """
            total_samples = total_units
        else:
            # Cluster bootstrap
            total_clusters = self.conn.execute(
                f"SELECT COUNT(DISTINCT {self.cluster_col}) FROM {self.table_name}"
            ).fetchone()[0]
            self.bootstrap_query = f"""
            SELECT
                {', '.join([f'{cov}' for cov in self.covariates])},
                {', '.join([f'avg_{cov}_unit' for cov in self.covariates])}
                {', ' + ', '.join([f'avg_{cov}_time' for cov in self.covariates]) if self.time_col is not None else ''},
                COUNT(*) as count,
                SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM design_matrix
            WHERE {self.cluster_col} IN (SELECT unnest((?)))
            GROUP BY {', '.join([f'{cov}' for cov in self.covariates])},
                        {', '.join([f'avg_{cov}_unit' for cov in self.covariates])}
                        {', ' + ', '.join([f'avg_{cov}_time' for cov in self.covariates]) if self.time_col is not None else ''}
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

            X = df_boot[rhs].values
            X = np.c_[np.ones(X.shape[0]), X]
            y = df_boot[f"mean_{self.outcome_var}"].values
            n = df_boot["count"].values
            boot_coefs[b, :] = wls(X, y, n)

        return np.cov(boot_coefs.T)


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
        n_bootstraps: int = 100,
        seed: int = 42,
        cluster_col: str = None,
    ):
        super().__init__(db_name, table_name, n_bootstraps, seed)
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

    def estimate(self):
        X = self.df_compressed[f"ddot_{self.treatment_var}"].values
        X = np.c_[np.ones(X.shape[0]), X]
        y = self.df_compressed[f"mean_{self.outcome_var}"].values
        n = self.df_compressed["count"].values
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

            X = df_boot[f"ddot_{self.treatment_var}"].values
            X = np.c_[np.ones(X.shape[0]), X]
            y = df_boot[f"mean_{self.outcome_var}"].values
            n = df_boot["count"].values
            boot_coefs[b, :] = wls(X, y, n)

        return np.cov(boot_coefs.T)


######################################################################


def _convert_to_int(data: pd.DataFrame) -> pd.DataFrame:

    fval = np.zeros_like(data)
    for i, col in enumerate(data.columns):
        fval[:, i] = pd.factorize(data[col])[0]

    if fval.dtype != int:
        fval = fval.astype(int)

    fval = fval.reshape(-1, 1) if fval.ndim == 1 else fval

    return fval
