import numpy as np
import pandas as pd
from typing import Union
from tqdm import tqdm
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
        if "|" in rhs:
            raise NotImplementedError(
                "Fixed effects in DuckRegression formulas are not supported. "
                "Use DuckMundlak or DuckDoubleDemeaning for panel fixed-effect designs."
            )

        self.outcome_vars = [x.strip() for x in lhs.split("+")]
        self.covars = [x.strip() for x in rhs.split("+")]
        self.strata_cols = self.covars

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

        # y and X need to be two-dimensional for the shared WLS helper.
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X
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

    def summary(self):
        if self.n_bootstraps > 0 or (hasattr(self, "se") and self.se == "hc1"):
            return {
                "point_estimate": self.point_estimate,
                "standard_error": np.sqrt(np.diag(self.vcov)),
            }
        return {"point_estimate": self.point_estimate}



################################################################################


def _add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    return np.c_[np.ones(X.shape[0]), X]


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    out = np.empty_like(eta, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    exp_eta = np.exp(eta[~pos])
    out[~pos] = exp_eta / (1.0 + exp_eta)
    return out


def _softmax(eta: np.ndarray) -> np.ndarray:
    eta = eta - np.max(eta, axis=1, keepdims=True)
    exp_eta = np.exp(eta)
    return exp_eta / exp_eta.sum(axis=1, keepdims=True)


def _sql_literal(value):
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def _solve_step(info: np.ndarray, score: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Solve a Newton/Fisher scoring step with a tiny ridge fallback."""
    try:
        return np.linalg.solve(info, score)
    except np.linalg.LinAlgError:
        return np.linalg.solve(info + ridge * np.eye(info.shape[0]), score)


def _weighted_logistic_irls(
    X: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    successes = successes.astype(float)
    totals = totals.astype(float)
    for _ in range(max_iter):
        p = np.clip(_sigmoid(X @ beta), 1e-9, 1 - 1e-9)
        score = X.T @ (successes - totals * p)
        info = X.T @ ((totals * p * (1 - p))[:, None] * X)
        step = _solve_step(info, score)
        max_step = np.max(np.abs(step))
        if max_step > 5:
            step = step * (5 / max_step)
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            return beta_new
        beta = beta_new
    return beta


def _weighted_poisson_irls(
    X: np.ndarray,
    y_sum: np.ndarray,
    totals: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    mean_y = np.maximum(y_sum.sum() / max(totals.sum(), 1.0), 1e-8)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(mean_y)
    y_sum = y_sum.astype(float)
    totals = totals.astype(float)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = np.exp(eta)
        score = X.T @ (y_sum - totals * mu)
        info = X.T @ ((totals * mu)[:, None] * X)
        step = _solve_step(info, score)
        max_step = np.max(np.abs(step))
        if max_step > 5:
            step = step * (5 / max_step)
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            return beta_new
        beta = beta_new
    return beta


def _multinomial_irls(
    X: np.ndarray,
    class_counts: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """Baseline-category multinomial logit IRLS on grouped class counts."""
    G, p = X.shape
    K = class_counts.shape[1]
    K1 = K - 1
    beta = np.zeros((K1, p))
    totals = class_counts.sum(axis=1)

    for _ in range(max_iter):
        eta = X @ beta.T
        probs = _softmax(np.c_[eta, np.zeros(G)])[:, :K1]
        score = (class_counts[:, :K1] - totals[:, None] * probs).T @ X
        info = np.zeros((K1 * p, K1 * p))
        for k in range(K1):
            for l in range(K1):
                w = totals * probs[:, k] * ((1.0 if k == l else 0.0) - probs[:, l])
                block = X.T @ (w[:, None] * X)
                info[k * p : (k + 1) * p, l * p : (l + 1) * p] = block
        step = _solve_step(info, score.reshape(-1)).reshape(K1, p)
        max_step = np.max(np.abs(step))
        if max_step > 5:
            step = step * (5 / max_step)
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            return beta_new
        beta = beta_new
    return beta


X_MIN_PILOT = 200


class _DuckCanonicalGLM(DuckReg):
    """Compressed canonical-link GLM base class.

    The exact estimator runs IRLS on grouped sufficient statistics. The
    `one_step` estimator follows Lumley's large-data trick: fit a pilot model on
    a subsample and take one full-data Fisher scoring step using compressed
    score and information.
    """

    family = None

    def __init__(
        self,
        db_name: str,
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
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            **kwargs,
        )
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
        self.n_obs = self.conn.execute(
            f"SELECT COUNT(*) FROM {self.table_name}"
        ).fetchone()[0]

    def compress_data(self):
        group_by_cols = ", ".join(self.covars)
        self.agg_query = f"""
        SELECT {group_by_cols}, COUNT(*) AS count, SUM({self.outcome_var}) AS sum_y
        FROM {self.table_name}
        GROUP BY {group_by_cols}
        """
        self.df_compressed = self.conn.execute(self.agg_query).fetchdf()

    def collect_data(self, data: pd.DataFrame):
        X = _add_intercept(data[self.covars].values)
        return X, data["sum_y"].values.astype(float), data["count"].values.astype(float)

    def _pilot_data(self):
        n = int(np.ceil(self.n_obs ** self.subsample_exponent))
        n = min(max(n, X_MIN_PILOT), self.n_obs)
        cols = [self.outcome_var] + self.covars
        fraction = min(1.0, n / max(self.n_obs, 1))
        table = self.conn.table(self.table_name)
        df = self.conn.to_pandas(
            table.select(cols).sample(fraction, seed=self.seed).limit(n)
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
            "Bootstrap is not implemented for compressed GLM estimators yet. "
            "Use fit_vcov() with n_bootstraps=0."
        )

    def summary(self):
        out = {"point_estimate": self.point_estimate}
        if hasattr(self, "vcov"):
            out["standard_error"] = np.sqrt(np.diag(self.vcov))
        return out


class DuckLogisticRegression(_DuckCanonicalGLM):
    family = "logistic"


class DuckPoissonRegression(_DuckCanonicalGLM):
    family = "poisson"


class DuckMultinomialLogisticRegression(DuckReg):
    """Compressed multinomial logit for moderate numbers of labels."""

    def __init__(
        self,
        db_name: str,
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
            self.labels = [
                x[0]
                for x in self.conn.execute(
                    f"SELECT DISTINCT {self.outcome_var} FROM {self.table_name} ORDER BY {self.outcome_var}"
                ).fetchall()
            ]
        if self.baseline is None:
            self.baseline = self.labels[-1]
        self.labels = [x for x in self.labels if x != self.baseline] + [self.baseline]

    def compress_data(self):
        group_by_cols = ", ".join(self.covars)
        count_exprs = ", ".join(
            [
                f"SUM(CASE WHEN {self.outcome_var} = {_sql_literal(label)} THEN 1 ELSE 0 END) AS class_{j}"
                for j, label in enumerate(self.labels)
            ]
        )
        self.agg_query = f"""
        SELECT {group_by_cols}, COUNT(*) AS count, {count_exprs}
        FROM {self.table_name}
        GROUP BY {group_by_cols}
        """
        self.df_compressed = self.conn.execute(self.agg_query).fetchdf()

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
            "Bootstrap is not implemented for compressed multinomial logit yet. "
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


class DuckPoissonMultinomialRegression(DuckReg):
    """Many-label multinomial/count model via label-wise Poisson regressions."""

    def __init__(
        self,
        db_name: str,
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
            self.labels = [
                x[0]
                for x in self.conn.execute(
                    f"SELECT DISTINCT {self.label_col} FROM {self.table_name} ORDER BY {self.label_col}"
                ).fetchall()
            ]

    def compress_data(self):
        group_by_cols = ", ".join([self.label_col] + self.covars)
        self.agg_query = f"""
        SELECT {group_by_cols}, COUNT(*) AS rows, SUM({self.count_col}) AS sum_y
        FROM {self.table_name}
        GROUP BY {group_by_cols}
        """
        self.df_compressed = self.conn.execute(self.agg_query).fetchdf()

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
            "Bootstrap is not implemented for the label-wise Poisson decomposition yet."
        )

    def summary(self):
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
            f"SUM(POW({y}, 2)) as sum_y_sq",
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

    def fit_vcov(self):
        """
        Compute analytic HC1 standard errors using compressed data.
        Assumes homoskedasticity within strata to approximate the meat matrix.
        """
        self.se = "hc1"
        df = self.df_compressed
        if df.empty:
            self.vcov = np.full((len(self.treatment_vars), len(self.treatment_vars)), np.nan)
            return

        n_treat = len(self.treatment_vars)
        n_g = df["n_g"].values
        
        # --- 1. Reconstruct S matrices ---
        S_X = np.stack([df[f"sum_{x}"] for x in self.treatment_vars], axis=1)
        S_Y = df[f"sum_y"].values.reshape(-1, 1)
        S_Y_sq = df["sum_y_sq"].values.reshape(-1, 1)
        
        S_XX = np.zeros((len(df), n_treat, n_treat))
        for i, x1 in enumerate(self.treatment_vars):
            for j, x2 in enumerate(self.treatment_vars):
                if j >= i:
                    col = f"sum_{x1}_{x2}"
                    S_XX[:, i, j] = df[col]
                    S_XX[:, j, i] = df[col]

        S_XY = np.zeros((len(df), n_treat, 1))
        for i, x in enumerate(self.treatment_vars):
            col = f"sum_{self.outcome_var}_{x}"
            S_XY[:, i, 0] = df[col]

        # --- 2. Compute Residual Cross-Products Q ---
        # Factor from LOO derivation: N_g / (N_g - 1)^2
        # Note: Q_WW is what we called M_XX_g before, but without the weight w_g?
        # No, previously M_XX_g = w * (N * S_XX - S_X S_X'). 
        # w = N / (N-1)^2.
        # So Q_WW^(g) = sum_{i in g} tilde_W_i tilde_W_i' IS exactly M_XX_g.
        
        weight = n_g / (n_g - 1) ** 2
        w_reshaped = weight.reshape(-1, 1, 1)
        n_g_reshaped = n_g.reshape(-1, 1, 1)

        # Q_WW (G, K, K) = sum tilde_W tilde_W'
        S_X_outer = np.einsum("bi,bj->bij", S_X, S_X)
        Q_WW = w_reshaped * (n_g_reshaped * S_XX - S_X_outer)
        
        # Q_XY (G, K, 1) = sum tilde_W tilde_Y
        S_X_expanded = S_X.reshape(len(df), n_treat, 1)
        S_Y_expanded = S_Y.reshape(len(df), 1, 1)
        Q_XY = w_reshaped * (n_g_reshaped * S_XY - S_X_expanded * S_Y_expanded)
        
        # Q_YY (G, 1, 1) = sum tilde_Y^2
        # Formula: w * (N * S_YY - S_Y^2)
        S_Y_sq_expanded = S_Y_sq.reshape(len(df), 1, 1)
        Q_YY = w_reshaped * (n_g_reshaped * S_Y_sq_expanded - S_Y_expanded**2)
        
        # --- 3. Point Estimate ---
        total_XTX = Q_WW.sum(axis=0)
        total_XTY = Q_XY.sum(axis=0)
        
        try:
            bread = np.linalg.inv(total_XTX)
            beta_hat = bread @ total_XTY
        except np.linalg.LinAlgError:
            self.vcov = np.full((n_treat, n_treat), np.nan)
            return
        
        self.point_estimate = beta_hat.flatten()

        # --- 4. Compute Meat (Approximation) ---
        # SSR_g = sum_{i in g} (tilde_Y_i - beta' tilde_W_i)^2
        #       = Q_YY - 2 beta' Q_WY + beta' Q_WW beta
        
        beta_reshaped = beta_hat.reshape(1, n_treat, 1) # (1, K, 1)
        
        # Term 2: 2 * beta' * Q_XY
        # Q_XY is (G, K, 1). beta is (1, K, 1).
        # dot: (G, 1, K) @ (G, K, 1) -> (G, 1, 1) ?
        # transpose Q_XY to (G, 1, K)
        term2 = 2 * np.matmul(np.transpose(Q_XY, (0, 2, 1)), beta_reshaped) # (G, 1, 1)
        
        # Term 3: beta' * Q_WW * beta
        # Q_WW is (G, K, K)
        # beta' Q_WW -> (G, 1, K)
        temp = np.matmul(np.transpose(beta_reshaped, (0, 2, 1)), Q_WW)
        term3 = np.matmul(temp, beta_reshaped) # (G, 1, 1)
        
        SSR_g = Q_YY - term2 + term3 # (G, 1, 1)
        
        # Estimate sigma_g^2 = SSR_g / N_g
        # (assuming homoskedasticity within group)
        # Avoid division by zero if N_g=0 (unlikely due to having count>1)
        sigma_sq_g = SSR_g / n_g_reshaped
        
        # Meat = sum_g (sigma_g^2 * Q_WW)
        # Broadcast sigma_sq_g (G, 1, 1) against Q_WW (G, K, K)
        weighted_Q_WW = sigma_sq_g * Q_WW
        meat = weighted_Q_WW.sum(axis=0)
        
        # --- 5. Sandwich ---
        self.vcov = bread @ meat @ bread
        
        # Finite sample correction n / (n-k)
        N = n_g.sum()
        K = n_treat
        self.vcov *= (N / (N - K))

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

    def summary(self):
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
