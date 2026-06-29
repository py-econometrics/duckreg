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
    def fit(self):
        self.prepare_data()
        self.compress_data()
        self.point_estimate = self.estimate()
        return self

    def prepare_data(self):
        pass

    def compress_data(self):
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
    ):
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

    def estimate(self) -> pd.Series:
        sum_w = self.moments["sum_w"]
        sum_c2 = self.moments["sum_c2"]
        sum_y = self.moments["sum_y"]
        sum_wy = self.moments["sum_wy"]
        sum_cy = self.moments["sum_cy"]
        sum_d2 = self.moments["sum_d2"]
        sum_dy = self.moments["sum_dy"]

        n_units = float(self.panel.n_units)
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


class BitmapMundlakEventStudy(_BitmapEstimator):
    """Mundlak event-study design from bitmap cohort-time moments."""

    def __init__(
        self,
        panel: _BitmapPanel,
        pre_treat_interactions: bool = True,
        popcount_function: str = "bit_count",
    ):
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

    def estimate(self) -> dict[str, pd.DataFrame]:
        rhs, rhs_index = _event_study_rhs(self.cohorts, self.panel.n_times)
        y_lookup = {
            (int(row.cohort), int(row.time_index)): float(row.sum_y)
            for row in self.cohort_time.itertuples(index=False)
        }
        n_lookup = {
            int(row.cohort): float(row.n_units)
            for row in self.cohort_time.drop_duplicates("cohort").itertuples(
                index=False
            )
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
                y_means.append(y_lookup[(cohort, time_index)] / n_units)
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

    def summary(self) -> dict[str, Any]:
        return {"point_estimate": self.point_estimate}
