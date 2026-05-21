"""Backend-neutral estimators built from Ibis expressions.

The legacy ``Duck*`` estimators are kept for compatibility.  The classes in
this module build compression and design-matrix operations with the Ibis
expression API instead of backend-specific SQL strings, so the same logical
query can be compiled/executed by any Ibis backend that supports the required
relational operations.
"""

from __future__ import annotations

from typing import Any, Union

import ibis
import numpy as np
import pandas as pd
from tqdm import tqdm

from .duckreg import DuckReg, wls


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _pow2(expr):
    return expr * expr


def _execute_frame(backend: Any, expr) -> pd.DataFrame:
    """Execute an Ibis table expression and always return a DataFrame."""
    out = backend.execute(expr, limit=None)
    if isinstance(out, pd.Series):
        return out.to_frame()
    if isinstance(out, pd.DataFrame):
        return out
    return pd.DataFrame(out)


class DBReg(DuckReg):
    """Base class for Ibis-native database regression estimators."""

    def table_expr(self):
        return self.backend.table(self.table_name)

    def execute_expr(self, expr) -> pd.DataFrame:
        return _execute_frame(self.backend, expr)

    def scalar_expr(self, expr):
        return self.backend.execute(expr, limit=None)


class DBRegression(DBReg):
    """Compressed linear regression using Ibis group-by aggregations."""

    def __init__(
        self,
        db_name: str | None,
        table_name: str,
        formula: str,
        cluster_col: str | None,
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
        if "|" in rhs:
            raise NotImplementedError(
                "Fixed effects in DBRegression formulas are not supported. "
                "Use DBMundlak or DBDoubleDemeaning for panel fixed-effect designs."
            )
        self.outcome_vars = [x.strip() for x in lhs.split("+") if x.strip()]
        self.covars = [x.strip() for x in rhs.split("+") if x.strip()]
        self.strata_cols = self.covars
        if not self.outcome_vars:
            raise ValueError("No outcome variables found in the formula")

    def prepare_data(self):
        pass

    def compression_expr(self, table=None, include_cluster: bool = False):
        table = self.table_expr() if table is None else table
        group_cols = list(self.strata_cols)
        if include_cluster and self.cluster_col:
            group_cols.append(self.cluster_col)
        metrics = {"count": table.count()}
        for var in self.outcome_vars:
            metrics[f"sum_{var}"] = table[var].sum()
            metrics[f"sum_{var}_sq"] = _pow2(table[var]).sum()
        return table.group_by(group_cols).aggregate(**metrics)

    def compress_data(self):
        self.compression = self.compression_expr()
        self.df_compressed = self.execute_expr(self.compression)
        for var in self.outcome_vars:
            self.df_compressed[f"mean_{var}"] = (
                self.df_compressed[f"sum_{var}"] / self.df_compressed["count"]
            )

    def collect_data(self, data: pd.DataFrame):
        y = data.filter(
            regex=f"mean_{'(' + '|'.join(self.outcome_vars) + ')'}", axis=1
        ).values
        X = data[self.covars].values
        n = data["count"].values
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        X = np.c_[np.ones(X.shape[0]), X]
        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(self.df_compressed)
        return wls(X, y, n).flatten()

    def fit_vcov(self):
        self.se = "hc1"
        y, X, n = self.collect_data(self.df_compressed)
        betahat = wls(X, y, n).flatten()
        self.n_bootstraps = 0
        yprime = self.df_compressed[f"sum_{self.outcome_vars[0]}"].values.reshape(-1, 1)
        yprimeprime = self.df_compressed[f"sum_{self.outcome_vars[0]}_sq"].values.reshape(-1, 1)
        yhat = (X @ betahat).reshape(-1, 1)
        rss_g = (yhat**2) * n.reshape(-1, 1) - 2 * yhat * yprime + yprimeprime
        bread = np.linalg.inv(X.T @ np.diag(n.flatten()) @ X)
        meat = X.T @ np.diag(rss_g.flatten()) @ X
        n_nk = n.sum() / (n.sum() - X.shape[1])
        self.vcov = n_nk * (bread @ meat @ bread)

    def bootstrap(self):
        boot_coefs = np.zeros(
            (self.n_bootstraps, (len(self.strata_cols) + 1) * len(self.outcome_vars))
        )
        if not self.cluster_col:
            source = self.df_compressed
            for b in tqdm(range(self.n_bootstraps)):
                idx = self.rng.choice(len(source), size=len(source), replace=True)
                boot = source.iloc[idx].groupby(self.strata_cols, as_index=False).sum(numeric_only=True)
                for var in self.outcome_vars:
                    boot[f"mean_{var}"] = boot[f"sum_{var}"] / boot["count"]
                y, X, n = self.collect_data(boot)
                boot_coefs[b, :] = wls(X, y, n).flatten()
        else:
            grouped = self.execute_expr(self.compression_expr(include_cluster=True))
            clusters = self.execute_expr(
                self.table_expr().select(self.cluster_col)
            )[self.cluster_col].drop_duplicates().to_numpy()
            for b in tqdm(range(self.n_bootstraps)):
                sampled = pd.Series(self.rng.choice(clusters, size=len(clusters), replace=True))
                mult = sampled.value_counts().rename_axis(self.cluster_col).reset_index(name="mult")
                boot = grouped.merge(mult, on=self.cluster_col)
                boot["count"] = boot["count"] * boot["mult"]
                for var in self.outcome_vars:
                    boot[f"sum_{var}"] = boot[f"sum_{var}"] * boot["mult"]
                    boot[f"sum_{var}_sq"] = boot[f"sum_{var}_sq"] * boot["mult"]
                boot = boot.groupby(self.strata_cols, as_index=False).sum(numeric_only=True)
                for var in self.outcome_vars:
                    boot[f"mean_{var}"] = boot[f"sum_{var}"] / boot["count"]
                y, X, n = self.collect_data(boot)
                boot_coefs[b, :] = wls(X, y, n).flatten()
        vcov = np.cov(boot_coefs.T)
        return np.expand_dims(vcov, axis=0) if vcov.ndim == 0 else vcov

    def summary(self):
        if self.n_bootstraps > 0 or (hasattr(self, "se") and self.se == "hc1"):
            return {
                "point_estimate": self.point_estimate,
                "standard_error": np.sqrt(np.diag(self.vcov)),
            }
        return {"point_estimate": self.point_estimate}


class DBDML(DBReg):
    """Compressed leave-one-out DML partialling estimator using Ibis."""

    def __init__(
        self,
        db_name: str | None,
        table_name: str,
        outcome_var: str,
        treatment_var: Union[str, list[str]],
        discrete_covars: list[str],
        seed: int,
        n_bootstraps: int = 200,
        **kwargs,
    ):
        super().__init__(db_name, table_name, seed, n_bootstraps, **kwargs)
        self.outcome_var = outcome_var
        self.treatment_vars = _as_list(treatment_var)
        self.discrete_covars = list(discrete_covars)

    def prepare_data(self):
        pass

    def collect_data(self):
        pass

    def compression_expr(self):
        t = self.table_expr()
        y = self.outcome_var
        metrics = {"n_g": t.count(), "sum_y": t[y].sum(), "sum_y_sq": _pow2(t[y]).sum()}
        for x in self.treatment_vars:
            metrics[f"sum_{x}"] = t[x].sum()
            metrics[f"sum_{y}_{x}"] = (t[y] * t[x]).sum()
        for i, x1 in enumerate(self.treatment_vars):
            for x2 in self.treatment_vars[i:]:
                metrics[f"sum_{x1}_{x2}"] = (t[x1] * t[x2]).sum()
        grouped = t.group_by(self.discrete_covars).aggregate(**metrics)
        return grouped.filter(grouped.n_g > 1)

    def compress_data(self):
        self.compression = self.compression_expr()
        self.df_compressed = self.execute_expr(self.compression)

    def _calculate_beta_from_compressed(self, data: pd.DataFrame) -> np.ndarray:
        if data.empty:
            return np.full(len(self.treatment_vars), np.nan)
        df = data
        n_treat = len(self.treatment_vars)
        n_g = df["n_g"].values
        weight = n_g / (n_g - 1) ** 2
        s_x = np.stack([df[f"sum_{x}"] for x in self.treatment_vars], axis=1)
        s_y = df["sum_y"].values.reshape(-1, 1)
        s_xx = np.zeros((len(df), n_treat, n_treat))
        for i, x1 in enumerate(self.treatment_vars):
            for j, x2 in enumerate(self.treatment_vars):
                col = f"sum_{x1}_{x2}" if j >= i else f"sum_{x2}_{x1}"
                s_xx[:, i, j] = df[col]
        s_xy = np.zeros((len(df), n_treat, 1))
        for i, x in enumerate(self.treatment_vars):
            s_xy[:, i, 0] = df[f"sum_{self.outcome_var}_{x}"]
        w = weight.reshape(-1, 1, 1)
        n = n_g.reshape(-1, 1, 1)
        xty = (w * (n * s_xy - s_x.reshape(len(df), n_treat, 1) * s_y.reshape(len(df), 1, 1))).sum(axis=0)
        xtx = (w * (n * s_xx - np.einsum("bi,bj->bij", s_x, s_x))).sum(axis=0)
        try:
            return np.linalg.solve(xtx, xty).flatten()
        except np.linalg.LinAlgError:
            return np.full(n_treat, np.nan)

    def estimate(self):
        return self._calculate_beta_from_compressed(self.df_compressed)

    def bootstrap(self):
        n_groups = len(self.df_compressed)
        boot_coefs = np.zeros((self.n_bootstraps, len(self.treatment_vars)))
        for b in tqdm(range(self.n_bootstraps), desc="Bootstrapping"):
            idx = self.rng.choice(n_groups, size=n_groups, replace=True)
            boot_coefs[b, :] = self._calculate_beta_from_compressed(self.df_compressed.iloc[idx])
        vcov = np.cov(boot_coefs, rowvar=False)
        return np.expand_dims(vcov, axis=0) if vcov.ndim == 0 else vcov


class DBMundlak(DBReg):
    """Mundlak panel regression with an Ibis expression design matrix."""

    def __init__(
        self,
        db_name: str | None,
        table_name: str,
        outcome_var: str,
        covariates: list[str],
        seed: int,
        unit_col: str,
        time_col: str | None = None,
        n_bootstraps: int = 100,
        cluster_col: str | None = None,
        **kwargs,
    ):
        super().__init__(db_name, table_name, seed, n_bootstraps, **kwargs)
        self.outcome_var = outcome_var
        self.covariates = list(covariates)
        self.unit_col = unit_col
        self.time_col = time_col
        self.cluster_col = cluster_col

    @property
    def rhs(self) -> list[str]:
        return (
            self.covariates
            + [f"avg_{cov}_unit" for cov in self.covariates]
            + ([f"avg_{cov}_time" for cov in self.covariates] if self.time_col is not None else [])
        )

    def prepare_data(self):
        t = self.table_expr()
        unit_avgs = t.group_by(self.unit_col).aggregate(
            **{f"avg_{cov}_unit": t[cov].mean() for cov in self.covariates}
        )
        design = t.join(unit_avgs, self.unit_col)
        if self.time_col is not None:
            time_avgs = t.group_by(self.time_col).aggregate(
                **{f"avg_{cov}_time": t[cov].mean() for cov in self.covariates}
            )
            design = design.join(time_avgs, self.time_col)
        select_cols = [self.unit_col]
        if self.time_col is not None:
            select_cols.append(self.time_col)
        if self.cluster_col and self.cluster_col != self.unit_col:
            select_cols.append(self.cluster_col)
        select_cols += [self.outcome_var] + self.rhs
        self.design_matrix = design.select(select_cols)

    def compression_expr(self):
        return self.design_matrix.group_by(self.rhs).aggregate(
            count=self.design_matrix.count(),
            **{f"sum_{self.outcome_var}": self.design_matrix[self.outcome_var].sum()},
        )

    def compress_data(self):
        self.compression = self.compression_expr()
        self.df_compressed = self.execute_expr(self.compression)
        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )

    def collect_data(self, data: pd.DataFrame):
        X = np.c_[np.ones(len(data)), data[self.rhs].values]
        y = data[f"mean_{self.outcome_var}"].values.reshape(-1, 1)
        n = data["count"].values
        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(self.df_compressed)
        return wls(X, y, n)

    def bootstrap(self):
        if self.cluster_col is not None:
            grouped = self.execute_expr(
                self.design_matrix.group_by(self.rhs + [self.cluster_col]).aggregate(
                    count=self.design_matrix.count(),
                    **{f"sum_{self.outcome_var}": self.design_matrix[self.outcome_var].sum()},
                )
            )
            clusters = self.execute_expr(
                self.table_expr().select(self.cluster_col)
            )[self.cluster_col].drop_duplicates().to_numpy()
            boot_coefs = np.zeros((self.n_bootstraps, len(self.rhs) + 1))
            for b in tqdm(range(self.n_bootstraps)):
                sampled = pd.Series(self.rng.choice(clusters, size=len(clusters), replace=True))
                mult = sampled.value_counts().rename_axis(self.cluster_col).reset_index(name="mult")
                boot = grouped.merge(mult, on=self.cluster_col)
                boot["count"] *= boot["mult"]
                boot[f"sum_{self.outcome_var}"] *= boot["mult"]
                boot = boot.groupby(self.rhs, as_index=False).sum(numeric_only=True)
                boot[f"mean_{self.outcome_var}"] = boot[f"sum_{self.outcome_var}"] / boot["count"]
                y, X, n = self.collect_data(boot)
                boot_coefs[b, :] = wls(X, y, n).flatten()
            return np.cov(boot_coefs.T)
        raise NotImplementedError("IID bootstrap for DBMundlak is not implemented yet")


class DBDoubleDemeaning(DBReg):
    """Double-demeaned treatment regression using Ibis joins/aggregations."""

    def __init__(
        self,
        db_name: str | None,
        table_name: str,
        outcome_var: str,
        treatment_var: str,
        unit_col: str,
        time_col: str,
        seed: int,
        n_bootstraps: int = 100,
        cluster_col: str | None = None,
        **kwargs,
    ):
        super().__init__(db_name, table_name, seed, n_bootstraps, **kwargs)
        self.outcome_var = outcome_var
        self.treatment_var = treatment_var
        self.unit_col = unit_col
        self.time_col = time_col
        self.cluster_col = cluster_col

    def prepare_data(self):
        t = self.table_expr()
        unit_means = t.group_by(self.unit_col).aggregate(
            **{f"mean_{self.treatment_var}_unit": t[self.treatment_var].mean()}
        )
        time_means = t.group_by(self.time_col).aggregate(
            **{f"mean_{self.treatment_var}_time": t[self.treatment_var].mean()}
        )
        overall = t.aggregate(**{f"mean_{self.treatment_var}": t[self.treatment_var].mean()})
        design = t.join(unit_means, self.unit_col).join(time_means, self.time_col).cross_join(overall)
        ddot = (
            design[self.treatment_var]
            - design[f"mean_{self.treatment_var}_unit"]
            - design[f"mean_{self.treatment_var}_time"]
            + design[f"mean_{self.treatment_var}"]
        )
        cols = [self.unit_col, self.time_col]
        if self.cluster_col and self.cluster_col != self.unit_col:
            cols.append(self.cluster_col)
        self.design_matrix = design.mutate(**{f"ddot_{self.treatment_var}": ddot}).select(
            cols + [self.outcome_var, f"ddot_{self.treatment_var}"]
        )

    def compression_expr(self):
        ddot_col = f"ddot_{self.treatment_var}"
        return self.design_matrix.group_by(ddot_col).aggregate(
            count=self.design_matrix.count(),
            **{f"sum_{self.outcome_var}": self.design_matrix[self.outcome_var].sum()},
        )

    def compress_data(self):
        self.compression = self.compression_expr()
        self.df_compressed = self.execute_expr(self.compression)
        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )

    def collect_data(self, data: pd.DataFrame):
        x = data[f"ddot_{self.treatment_var}"].values
        X = np.c_[np.ones(len(data)), x]
        y = data[f"mean_{self.outcome_var}"].values.reshape(-1, 1)
        n = data["count"].values
        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(self.df_compressed)
        return wls(X, y, n)

    def bootstrap(self):
        raise NotImplementedError("Bootstrap for DBDoubleDemeaning is not implemented yet")


# Friendly mixed-case aliases for users who type Db* rather than DB*.
DbReg = DBReg
DbRegression = DBRegression
DbDML = DBDML
DbMundlak = DBMundlak
DbDoubleDemeaning = DBDoubleDemeaning
