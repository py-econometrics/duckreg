"""Bitmap Mundlak estimators for dense low-cardinality panels.

Use this module for estimation:

```
BitmapMundlak.from_long(data).fit()
BitmapMundlakEventStudy.from_long(data).fit()
```

The lower-level bitmap layout and demo helpers live in :mod:`duckreg.bitmap_utils`.
The current SQL implementation targets DuckDB because Ibis does not currently
expose a backend-neutral popcount expression.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .bitmap_utils import (
    COHORT_TIME_SUMS_SQL as _COHORT_TIME_SUMS_SQL,
    STATIC_MUNDLAK_DIRECT_SQL as _STATIC_MUNDLAK_DIRECT_SQL,
    BitmapPanel as _BitmapPanel,
    LongPanelBitPacker as _LongPanelBitPacker,
    _bitmap_connection,
    _event_study_rhs,
    _event_study_row,
    _split_packer_kwargs,
    _with_popcount_function,
)
from .duckreg import wls

__all__ = ["BitmapMundlak", "BitmapMundlakEventStudy"]


class _BitmapEstimator:
    def _init_bootstrap(self, seed: int, n_bootstraps: int):
        if n_bootstraps < 0:
            raise ValueError("n_bootstraps must be nonnegative")
        self.seed = seed
        self.n_bootstraps = n_bootstraps
        self.rng = np.random.default_rng(seed)

    def fit(self):
        self.prepare_data()
        self.compress_data()
        self.point_estimate = self.estimate()
        if self.n_bootstraps > 0:
            if self.n_bootstraps < 2:
                raise ValueError("n_bootstraps must be at least 2 when bootstrapping")
            self.vcov = self.bootstrap()
        return self

    def prepare_data(self):
        pass

    def compress_data(self):
        pass

    def bootstrap(self):
        pass

    def summary(self) -> dict[str, Any]:
        return {"point_estimate": self.point_estimate}


class BitmapMundlak(_BitmapEstimator):
    """Static two-way Mundlak LPM from direct bitmap moments."""

    def __init__(
        self,
        panel: _BitmapPanel,
        treatment_col: str | None = None,
        popcount_function: str = "bit_count",
        seed: int = 42,
        n_bootstraps: int = 0,
    ):
        self._init_bootstrap(seed=seed, n_bootstraps=n_bootstraps)
        self.panel = panel
        self.treatment_col = treatment_col or panel.treatment_col
        self.popcount_function = popcount_function

    @classmethod
    def from_long(cls, data: pd.DataFrame, **kwargs):
        """Build the bitmap layout from a balanced long panel."""

        packer_kwargs = _split_packer_kwargs(kwargs)
        panel = _LongPanelBitPacker(**packer_kwargs).fit_transform(data)
        estimator_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in packer_kwargs and key != "data"
        }
        return cls(panel, **estimator_kwargs)

    @property
    def rhs(self) -> list[str]:
        return [
            self.treatment_col,
            f"avg_{self.treatment_col}_unit",
            f"avg_{self.treatment_col}_time",
        ]

    def compress_data(self):
        sql = _with_popcount_function(
            _STATIC_MUNDLAK_DIRECT_SQL, self.popcount_function
        )
        with _bitmap_connection(self.panel) as con:
            stats = con.execute(sql).fetchone()
        columns = [
            "sum_w",
            "sum_c2",
            "sum_y",
            "sum_wy",
            "sum_cy",
            "sum_d2",
            "sum_dy",
        ]
        self.moments = pd.Series(map(float, stats), index=columns)
        self._cache_unit_arrays()

    def estimate(self) -> pd.Series:
        return self._estimate_from_moments(self.moments, float(self.panel.n_units))

    def _estimate_from_moments(
        self,
        moments: pd.Series,
        n_units: float,
    ) -> pd.Series:
        sum_w = moments["sum_w"]
        sum_c2 = moments["sum_c2"]
        sum_y = moments["sum_y"]
        sum_wy = moments["sum_wy"]
        sum_cy = moments["sum_cy"]
        sum_d2 = moments["sum_d2"]
        sum_dy = moments["sum_dy"]
        n_times = float(self.panel.n_times)
        n_obs = n_units * n_times
        xtx = np.array(
            [
                [n_obs, sum_w, sum_w, sum_w],
                [sum_w, sum_w, sum_c2 / n_times, sum_d2 / n_units],
                [sum_w, sum_c2 / n_times, sum_c2 / n_times, sum_w**2 / n_obs],
                [sum_w, sum_d2 / n_units, sum_w**2 / n_obs, sum_d2 / n_units],
            ],
            dtype=float,
        )
        xty = np.array(
            [
                sum_y,
                sum_wy,
                sum_cy / n_times,
                sum_dy / n_units,
            ],
            dtype=float,
        )
        beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
        return pd.Series(
            beta,
            index=["Intercept", *self.rhs],
        )

    def _cache_unit_arrays(self):
        w = self.panel.w.astype(float, copy=False)
        y = self.panel.y.astype(float, copy=False)
        self._unit_w = w
        self._unit_y_matrix = y
        self._unit_w_sum = w.sum(axis=1)
        self._unit_y_sum = y.sum(axis=1)
        self._unit_wy_sum = (w * y).sum(axis=1)
        self._unit_w2_sum = self._unit_w_sum * self._unit_w_sum
        self._unit_w_y_sum = self._unit_w_sum * self._unit_y_sum

    def _moments_from_unit_weights(self, weights: np.ndarray) -> pd.Series:
        d_t = weights @ self._unit_w
        y_t = weights @ self._unit_y_matrix
        values = {
            "sum_w": weights @ self._unit_w_sum,
            "sum_c2": weights @ self._unit_w2_sum,
            "sum_y": weights @ self._unit_y_sum,
            "sum_wy": weights @ self._unit_wy_sum,
            "sum_cy": weights @ self._unit_w_y_sum,
            "sum_d2": d_t @ d_t,
            "sum_dy": d_t @ y_t,
        }
        return pd.Series(values, dtype=float)

    def bootstrap(self) -> np.ndarray:
        n_units = self.panel.n_units
        boot_coefs = np.zeros((self.n_bootstraps, len(self.point_estimate)))
        for boot_index in range(self.n_bootstraps):
            weights = _draw_cluster_weights(self.rng, n_units)
            moments = self._moments_from_unit_weights(weights)
            boot_coefs[boot_index, :] = self._estimate_from_moments(
                moments,
                n_units=float(weights.sum()),
            ).to_numpy()
        return np.cov(boot_coefs.T)

    def summary(self) -> dict[str, Any]:
        if self.n_bootstraps > 0:
            return {
                "point_estimate": self.point_estimate,
                "standard_error": pd.Series(
                    np.sqrt(np.diag(self.vcov)),
                    index=self.point_estimate.index,
                ),
            }
        return {"point_estimate": self.point_estimate}


class BitmapMundlakEventStudy(_BitmapEstimator):
    """Mundlak event-study design from bitmap cohort-time moments."""

    def __init__(
        self,
        panel: _BitmapPanel,
        pre_treat_interactions: bool = True,
        popcount_function: str = "bit_count",
        seed: int = 42,
        n_bootstraps: int = 0,
    ):
        self._init_bootstrap(seed=seed, n_bootstraps=n_bootstraps)
        self.panel = panel
        self.pre_treat_interactions = pre_treat_interactions
        self.popcount_function = popcount_function

    @classmethod
    def from_long(cls, data: pd.DataFrame, **kwargs):
        """Build the bitmap layout from a balanced long panel."""

        packer_kwargs = _split_packer_kwargs(kwargs)
        panel = _LongPanelBitPacker(**packer_kwargs).fit_transform(data)
        estimator_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in packer_kwargs and key != "data"
        }
        return cls(panel, **estimator_kwargs)

    def prepare_data(self):
        self.cohorts = self.panel.treated_cohorts
        self.num_periods = self.panel.n_times - 1

    def compress_data(self):
        sql = _with_popcount_function(_COHORT_TIME_SUMS_SQL, self.popcount_function)
        with _bitmap_connection(self.panel) as con:
            self.cohort_time = con.execute(sql).fetchdf()
        self._cache_event_arrays()

    def estimate(self) -> dict[str, pd.DataFrame]:
        return self._estimate_from_cohort_time(self.cohort_time)

    def _estimate_from_cohort_time(
        self,
        cohort_time: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        rhs, rhs_index = _event_study_rhs(self.cohorts, self.panel.n_times)
        y_lookup = {
            (int(row.cohort), int(row.time_index)): float(row.sum_y)
            for row in cohort_time.itertuples(index=False)
        }
        n_lookup = {
            int(row.cohort): float(row.n_units)
            for row in cohort_time.drop_duplicates("cohort").itertuples(index=False)
        }

        rows = []
        y_means = []
        weights = []
        for cohort in self.panel.all_cohorts:
            n_units = n_lookup[cohort]
            for time_index in range(self.panel.n_times):
                rows.append(
                    _event_study_row(
                        cohort,
                        time_index,
                        self.cohorts,
                        rhs_index,
                        self.pre_treat_interactions,
                    )
                )
                if n_units > 0:
                    y_means.append(y_lookup[(cohort, time_index)] / n_units)
                else:
                    y_means.append(0.0)
                weights.append(n_units)

        coef = wls(
            np.vstack(rows),
            np.asarray(y_means, dtype=float).reshape(-1, 1),
            np.asarray(weights, dtype=float),
        ).reshape(-1)
        estimates = pd.Series(coef, index=rhs)

        result = {}
        for cohort in self.cohorts:
            values = []
            index = []
            for time_index in range(self.panel.n_times):
                name = f"treatment_time_{cohort}_{time_index}"
                index.append(name)
                if time_index == cohort - 1:
                    values.append(0.0)
                else:
                    values.append(float(estimates[name]))
            result[str(cohort)] = pd.DataFrame({"est": values}, index=index)
        return result

    def _cache_event_arrays(self):
        self._cohort_values = np.asarray(self.panel.all_cohorts, dtype=int)
        self._cohort_to_code = {
            cohort: code for code, cohort in enumerate(self._cohort_values)
        }
        self._unit_cohort_codes = np.asarray(
            [self._cohort_to_code[int(cohort)] for cohort in self.panel.cohorts],
            dtype=int,
        )
        self._unit_y_matrix = self.panel.y.astype(float, copy=False)

    def _cohort_time_from_unit_weights(self, weights: np.ndarray) -> pd.DataFrame:
        n_cohorts = len(self._cohort_values)
        n_units = np.bincount(
            self._unit_cohort_codes,
            weights=weights,
            minlength=n_cohorts,
        )
        records = []
        for time_index in range(self.panel.n_times):
            sum_y = np.bincount(
                self._unit_cohort_codes,
                weights=weights * self._unit_y_matrix[:, time_index],
                minlength=n_cohorts,
            )
            for code, cohort in enumerate(self._cohort_values):
                records.append(
                    {
                        "cohort": int(cohort),
                        "time_index": time_index,
                        "sum_y": float(sum_y[code]),
                        "n_units": float(n_units[code]),
                    }
                )
        return pd.DataFrame.from_records(records)

    def bootstrap(self) -> dict[str, np.ndarray]:
        n_units = self.panel.n_units
        boot_coefs = {str(cohort): [] for cohort in self.cohorts}
        for _ in range(self.n_bootstraps):
            weights = _draw_cluster_weights(self.rng, n_units)
            cohort_time = self._cohort_time_from_unit_weights(weights)
            estimates = self._estimate_from_cohort_time(cohort_time)
            for cohort, table in estimates.items():
                boot_coefs[cohort].append(table["est"].to_numpy())
        return {
            cohort: np.cov(np.asarray(coefs).T)
            for cohort, coefs in boot_coefs.items()
        }

    def summary(self) -> dict[str, Any]:
        if self.n_bootstraps > 0:
            summary_tables = {}
            for cohort, point_estimate in self.point_estimate.items():
                se = np.sqrt(np.diag(self.vcov[cohort]))
                summary_tables[cohort] = pd.DataFrame(
                    {
                        "point_estimate": point_estimate["est"].to_numpy(),
                        "se": se,
                    },
                    index=point_estimate.index,
                )
            return summary_tables
        return {"point_estimate": self.point_estimate}


def _draw_cluster_weights(rng: np.random.Generator, n_clusters: int) -> np.ndarray:
    sampled = rng.integers(0, n_clusters, size=n_clusters)
    return np.bincount(sampled, minlength=n_clusters).astype(float)
