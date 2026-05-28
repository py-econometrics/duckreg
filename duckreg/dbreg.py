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
from .estimators import (
    X_MIN_PILOT,
    _add_intercept,
    _multinomial_irls,
    _sigmoid,
    _softmax,
    _solve_step,
    _weighted_logistic_irls,
    _weighted_poisson_irls,
)


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


class DBMundlakEventStudy(DBReg):
    """Two-way Mundlak event study with an Ibis expression design matrix."""

    def __init__(
        self,
        db_name: str | None,
        table_name: str,
        outcome_var: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        cluster_col: str,
        seed: int,
        pre_treat_interactions: bool = True,
        n_bootstraps: int = 100,
        **kwargs,
    ):
        super().__init__(db_name, table_name, seed, n_bootstraps, **kwargs)
        self.outcome_var = outcome_var
        self.treatment_col = treatment_col
        self.unit_col = unit_col
        self.time_col = time_col
        self.cluster_col = cluster_col
        self.pre_treat_interactions = pre_treat_interactions

    @staticmethod
    def _reference_period(cohort):
        """Absolute time period used as the event-study normalization.

        Event-study coefficients are reported relative to period -1, i.e. the
        period immediately before treatment begins for each adoption cohort.
        With absolute time indexing, that reference period is ``cohort - 1``.
        """

        return cohort - 1

    def prepare_data(self):
        t = self.table_expr()
        treated = (
            t.filter(t[self.treatment_col] == 1)
            .group_by(self.unit_col)
            .aggregate(cohort=t[self.time_col].min())
        )
        cohort_data = t.left_join(treated, self.unit_col).mutate(
            ever_treated=lambda x: x.cohort.notnull().ifelse(1, 0)
        )
        self.num_periods = self.scalar_expr(t[self.time_col].max())
        cohort_frame = self.execute_expr(
            treated.select("cohort").distinct().order_by("cohort")
        )
        self.cohorts = cohort_frame["cohort"].dropna().tolist()

        cohort_cols = {
            f"cohort_{cohort}": (cohort_data.cohort == cohort).ifelse(1, 0)
            for cohort in self.cohorts
        }
        time_cols = {
            f"time_{i}": (cohort_data[self.time_col] == i).ifelse(1, 0)
            for i in range(int(self.num_periods) + 1)
        }
        treatment_cols = {}
        for cohort in self.cohorts:
            reference_period = self._reference_period(cohort)
            for i in range(int(self.num_periods) + 1):
                if i == reference_period:
                    continue
                condition = (cohort_data.cohort == cohort) & (
                    cohort_data[self.time_col] == i
                )
                if not self.pre_treat_interactions:
                    condition = condition & (cohort_data[self.treatment_col] == 1)
                treatment_cols[f"treatment_time_{cohort}_{i}"] = condition.ifelse(1, 0)

        design = cohort_data.mutate(
            intercept=ibis.literal(1),
            **cohort_cols,
            **time_cols,
            **treatment_cols,
        )
        self.rhs_cols = (
            ["intercept"]
            + list(cohort_cols)
            + list(time_cols)
            + list(treatment_cols)
        )
        select_cols = [
            self.unit_col,
            self.time_col,
            self.treatment_col,
            self.outcome_var,
        ]
        if self.cluster_col != self.unit_col:
            select_cols.append(self.cluster_col)
        select_cols += self.rhs_cols
        self.design_matrix = design.select(select_cols)

    def compression_expr(self, include_cluster: bool = False):
        group_cols = list(self.rhs_cols)
        if include_cluster:
            group_cols.append(self.cluster_col)
        return self.design_matrix.group_by(group_cols).aggregate(
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
        X = data[self.rhs_cols].values
        y = data[f"mean_{self.outcome_var}"].values
        n = data["count"].values
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return y, X, n

    def _event_study_from_compressed(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        y, X, n = self.collect_data(data)
        coef = wls(X, y, n)
        res = pd.DataFrame({"est": coef.squeeze()}, index=self.rhs_cols)
        event_study_coefs = {}
        for cohort in self.cohorts:
            cohort_name = str(cohort)
            reference_period = self._reference_period(cohort)
            estimates = []
            index = []
            for i in range(int(self.num_periods) + 1):
                name = f"treatment_time_{cohort_name}_{i}"
                index.append(name)
                if i == reference_period:
                    estimates.append(0.0)
                else:
                    estimates.append(float(res.loc[name, "est"]))
            event_study_coefs[cohort_name] = pd.DataFrame(
                {"est": estimates}, index=index
            )
        return event_study_coefs

    def estimate(self):
        return self._event_study_from_compressed(self.df_compressed)

    def bootstrap(self):
        grouped = self.execute_expr(self.compression_expr(include_cluster=True))
        clusters = self.execute_expr(self.design_matrix.select(self.cluster_col))[
            self.cluster_col
        ].drop_duplicates().to_numpy()
        boot_coefs = {str(cohort): [] for cohort in self.cohorts}
        for _ in tqdm(range(self.n_bootstraps)):
            sampled = pd.Series(self.rng.choice(clusters, size=len(clusters), replace=True))
            mult = sampled.value_counts().rename_axis(self.cluster_col).reset_index(name="mult")
            boot = grouped.merge(mult, on=self.cluster_col)
            boot["count"] *= boot["mult"]
            boot[f"sum_{self.outcome_var}"] *= boot["mult"]
            boot = boot.groupby(self.rhs_cols, as_index=False).sum(numeric_only=True)
            boot[f"mean_{self.outcome_var}"] = (
                boot[f"sum_{self.outcome_var}"] / boot["count"]
            )
            for cohort, coefs in self._event_study_from_compressed(boot).items():
                boot_coefs[cohort].append(coefs.values.flatten())
        return {
            cohort: np.cov(np.array(coefs).T) for cohort, coefs in boot_coefs.items()
        }

    def summary(self) -> dict:
        if self.n_bootstraps > 0:
            summary_tables = {}
            for cohort in self.point_estimate:
                point_estimate = self.point_estimate[cohort]
                se = np.sqrt(np.diag(self.vcov[cohort]))
                summary_tables[cohort] = pd.DataFrame(
                    np.c_[point_estimate, se],
                    columns=["point_estimate", "se"],
                    index=point_estimate.index,
                )
            return summary_tables
        return {"point_estimate": self.point_estimate}


class _DBCanonicalGLM(DBReg):
    """Compressed canonical-link GLM base class using Ibis expressions."""

    family = None

    def __init__(
        self,
        db_name: str | None,
        table_name: str,
        formula: str,
        seed: int,
        n_bootstraps: int = 0,
        method: str = "one_step",
        subsample_exponent: float = 5 / 9,
        max_iter: int = 100,
        tol: float = 1e-10,
        **kwargs,
    ):
        super().__init__(db_name, table_name, seed, n_bootstraps, **kwargs)
        if method not in {"one_step", "irls"}:
            raise ValueError("method must be 'one_step' or 'irls'")
        self.formula = formula
        self.method = method
        self.subsample_exponent = subsample_exponent
        self.max_iter = max_iter
        self.tol = tol
        self._parse_formula()

    def _parse_formula(self):
        lhs, rhs = self.formula.split("~")
        if "|" in rhs:
            raise NotImplementedError("GLM fixed effects are not implemented yet")
        self.outcome_var = lhs.strip()
        self.covars = [x.strip() for x in rhs.split("+") if x.strip()]

    def prepare_data(self):
        self.n_obs = self.scalar_expr(self.table_expr().count())

    def compression_expr(self):
        t = self.table_expr()
        return t.group_by(self.covars).aggregate(
            count=t.count(),
            sum_y=t[self.outcome_var].sum(),
        )

    def compress_data(self):
        self.compression = self.compression_expr()
        self.df_compressed = self.execute_expr(self.compression)

    def collect_data(self, data: pd.DataFrame):
        X = _add_intercept(data[self.covars].values)
        return X, data["sum_y"].values.astype(float), data["count"].values.astype(float)

    def _pilot_data(self):
        n = int(np.ceil(self.n_obs ** self.subsample_exponent))
        n = min(max(n, X_MIN_PILOT), self.n_obs)
        cols = [self.outcome_var] + self.covars
        fraction = min(1.0, n / max(self.n_obs, 1))
        df = self.execute_expr(
            self.table_expr().select(cols).sample(fraction, seed=self.seed).limit(n)
        )
        X = _add_intercept(df[self.covars].values)
        y = df[self.outcome_var].values.astype(float)
        return X, y, np.ones(len(df))

    def _score_info(self, beta: np.ndarray, data: pd.DataFrame | None = None):
        X, y_sum, totals = self.collect_data(data if data is not None else self.df_compressed)
        if self.family == "logistic":
            mu = np.clip(_sigmoid(X @ beta), 1e-9, 1 - 1e-9)
            var = mu * (1 - mu)
        elif self.family == "poisson":
            mu = np.exp(np.clip(X @ beta, -30, 30))
            var = mu
        else:
            raise ValueError("unknown family")
        score = X.T @ (y_sum - totals * mu)
        info = X.T @ ((totals * var)[:, None] * X)
        return score, info

    def estimate(self):
        X, y_sum, totals = self.collect_data(self.df_compressed)
        if self.method == "irls":
            if self.family == "logistic":
                return _weighted_logistic_irls(X, y_sum, totals, self.max_iter, self.tol)
            return _weighted_poisson_irls(X, y_sum, totals, self.max_iter, self.tol)

        X0, y0, n0 = self._pilot_data()
        if self.family == "logistic":
            beta0 = _weighted_logistic_irls(X0, y0, n0, self.max_iter, self.tol)
        else:
            beta0 = _weighted_poisson_irls(X0, y0, n0, self.max_iter, self.tol)
        score, info = self._score_info(beta0)
        return beta0 + _solve_step(info, score)

    def fit_vcov(self, robust: bool = False):
        _, info = self._score_info(self.point_estimate)
        bread = np.linalg.inv(info)
        if not robust:
            self.vcov = bread
        else:
            X, y_sum, totals = self.collect_data(self.df_compressed)
            if self.family == "logistic":
                mu = np.clip(_sigmoid(X @ self.point_estimate), 1e-9, 1 - 1e-9)
            else:
                mu = np.exp(np.clip(X @ self.point_estimate, -30, 30))
            grouped_scores = X * (y_sum - totals * mu)[:, None]
            meat = grouped_scores.T @ grouped_scores
            self.vcov = bread @ meat @ bread
        return self.vcov

    def bootstrap(self):
        raise NotImplementedError(
            "Bootstrap is not implemented for compressed DB GLM estimators yet. "
            "Use fit_vcov() with n_bootstraps=0."
        )

    def summary(self):
        out = {"point_estimate": self.point_estimate}
        if hasattr(self, "vcov"):
            out["standard_error"] = np.sqrt(np.diag(self.vcov))
        return out


class DBLogisticRegression(_DBCanonicalGLM):
    family = "logistic"


class DBPoissonRegression(_DBCanonicalGLM):
    family = "poisson"


class DBMultinomialLogisticRegression(DBReg):
    """Compressed multinomial logit using Ibis grouped class counts."""

    def __init__(
        self,
        db_name: str | None,
        table_name: str,
        formula: str,
        seed: int,
        n_bootstraps: int = 0,
        labels: list | None = None,
        baseline=None,
        max_iter: int = 100,
        tol: float = 1e-10,
        **kwargs,
    ):
        super().__init__(db_name, table_name, seed, n_bootstraps, **kwargs)
        self.formula = formula
        self.labels = labels
        self.baseline = baseline
        self.max_iter = max_iter
        self.tol = tol
        self._parse_formula()

    def _parse_formula(self):
        lhs, rhs = self.formula.split("~")
        if "|" in rhs:
            raise NotImplementedError("GLM fixed effects are not implemented yet")
        self.outcome_var = lhs.strip()
        self.covars = [x.strip() for x in rhs.split("+") if x.strip()]

    def prepare_data(self):
        if self.labels is None:
            labels = self.execute_expr(
                self.table_expr().select(self.outcome_var).distinct().order_by(self.outcome_var)
            )
            self.labels = labels[self.outcome_var].tolist()
        if self.baseline is None:
            self.baseline = self.labels[-1]
        self.labels = [x for x in self.labels if x != self.baseline] + [self.baseline]

    def compression_expr(self):
        t = self.table_expr()
        metrics = {"count": t.count()}
        for j, label in enumerate(self.labels):
            metrics[f"class_{j}"] = (t[self.outcome_var] == label).ifelse(1, 0).sum()
        return t.group_by(self.covars).aggregate(**metrics)

    def compress_data(self):
        self.compression = self.compression_expr()
        self.df_compressed = self.execute_expr(self.compression)

    def collect_data(self, data: pd.DataFrame):
        X = _add_intercept(data[self.covars].values)
        counts = data[[f"class_{j}" for j in range(len(self.labels))]].values.astype(float)
        return X, counts

    def estimate(self):
        X, counts = self.collect_data(self.df_compressed)
        return _multinomial_irls(X, counts, self.max_iter, self.tol)

    def fit_vcov(self):
        X, counts = self.collect_data(self.df_compressed)
        totals = counts.sum(axis=1)
        G, p = X.shape
        K1 = len(self.labels) - 1
        eta = X @ self.point_estimate.T
        probs = _softmax(np.c_[eta, np.zeros(G)])[:, :K1]
        info = np.zeros((K1 * p, K1 * p))
        for k in range(K1):
            for l in range(K1):
                w = totals * probs[:, k] * ((1.0 if k == l else 0.0) - probs[:, l])
                info[k * p : (k + 1) * p, l * p : (l + 1) * p] = X.T @ (w[:, None] * X)
        self.vcov = np.linalg.inv(info)
        return self.vcov

    def bootstrap(self):
        raise NotImplementedError(
            "Bootstrap is not implemented for compressed DB multinomial logit yet. "
            "Use fit_vcov() with n_bootstraps=0."
        )

    def summary(self):
        out = {
            "labels": self.labels[:-1],
            "baseline": self.baseline,
            "point_estimate": self.point_estimate,
        }
        if hasattr(self, "vcov"):
            out["standard_error"] = np.sqrt(np.diag(self.vcov)).reshape(
                len(self.labels) - 1, -1
            )
        return out


class DBPoissonMultinomialRegression(DBReg):
    """Many-label count model via label-wise Poisson regressions."""

    def __init__(
        self,
        db_name: str | None,
        table_name: str,
        count_col: str,
        label_col: str,
        covars: list[str],
        seed: int,
        n_bootstraps: int = 0,
        labels: list | None = None,
        max_iter: int = 100,
        tol: float = 1e-10,
        **kwargs,
    ):
        super().__init__(db_name, table_name, seed, n_bootstraps, **kwargs)
        self.count_col = count_col
        self.label_col = label_col
        self.covars = covars
        self.labels = labels
        self.max_iter = max_iter
        self.tol = tol

    def prepare_data(self):
        if self.labels is None:
            labels = self.execute_expr(
                self.table_expr().select(self.label_col).distinct().order_by(self.label_col)
            )
            self.labels = labels[self.label_col].tolist()

    def compression_expr(self):
        t = self.table_expr()
        return t.group_by([self.label_col] + self.covars).aggregate(
            rows=t.count(),
            sum_y=t[self.count_col].sum(),
        )

    def compress_data(self):
        self.compression = self.compression_expr()
        self.df_compressed = self.execute_expr(self.compression)

    def collect_data(self, label):
        data = self.df_compressed[self.df_compressed[self.label_col] == label]
        X = _add_intercept(data[self.covars].values)
        return X, data["sum_y"].values.astype(float), data["rows"].values.astype(float)

    def estimate(self):
        coefs = []
        for label in self.labels:
            X, y_sum, totals = self.collect_data(label)
            coefs.append(_weighted_poisson_irls(X, y_sum, totals, self.max_iter, self.tol))
        return pd.DataFrame(coefs, index=self.labels, columns=["Intercept"] + self.covars)

    def bootstrap(self):
        raise NotImplementedError(
            "Bootstrap is not implemented for the label-wise DB Poisson decomposition yet."
        )

    def summary(self):
        return {"point_estimate": self.point_estimate}
