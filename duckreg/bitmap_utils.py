"""Support code for bitmap panel layout and moment queries.

The public estimator surface is in :mod:`duckreg.bitpacking`.  This module
contains the data layout, SQL fragments, synthetic examples, and formatting
helpers used by those estimators and the Quarto memo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd


CHUNK_BITS = 64
DEFAULT_MAX_OUTCOME_CARDINALITY = 64

__all__ = [
    "BitmapPanel",
    "CHUNK_BITS",
    "COHORT_TIME_SUMS_SQL",
    "DEFAULT_MAX_OUTCOME_CARDINALITY",
    "LongPanelBitPacker",
    "STATIC_MUNDLAK_DIRECT_SQL",
    "bitmap_translation_table",
    "make_five_period_demo_long_panel",
    "make_one_shot_binary_adoption_panel",
    "make_staggered_binary_adoption_panel",
    "pack_long_panel_to_bitmaps",
]


STATIC_MUNDLAK_DIRECT_SQL = """
WITH unit_treatment AS (
    SELECT
        unit_index,
        SUM(bit_count(w_bits & valid_mask))::DOUBLE AS c
    FROM unit_bits
    GROUP BY unit_index
),
unit_y AS (
    SELECT
        u.unit_index,
        SUM(y.outcome_value::DOUBLE * bit_count(y.y_bits & u.valid_mask)) AS y_sum,
        SUM(y.outcome_value::DOUBLE * bit_count(y.y_bits & u.w_bits & u.valid_mask)) AS wy_sum
    FROM unit_bits AS u
    JOIN unit_outcome_bits AS y USING (unit_index, chunk)
    GROUP BY u.unit_index
),
unit_stats AS (
    SELECT
        t.unit_index,
        t.c,
        COALESCE(y.y_sum, 0.0) AS y_sum,
        COALESCE(y.wy_sum, 0.0) AS wy_sum
    FROM unit_treatment AS t
    LEFT JOIN unit_y AS y USING (unit_index)
),
time_treatment AS (
    SELECT
        time_index,
        SUM(bit_count(w_bits & valid_mask))::DOUBLE AS d
    FROM time_bits
    GROUP BY time_index
),
time_y AS (
    SELECT
        t.time_index,
        SUM(y.outcome_value::DOUBLE * bit_count(y.y_bits & t.valid_mask)) AS y_sum
    FROM time_bits AS t
    JOIN time_outcome_bits AS y USING (time_index, chunk)
    GROUP BY t.time_index
),
time_stats AS (
    SELECT
        t.time_index,
        t.d,
        COALESCE(y.y_sum, 0.0) AS y_sum
    FROM time_treatment AS t
    LEFT JOIN time_y AS y USING (time_index)
)
SELECT
    (SELECT SUM(c) FROM unit_stats) AS sum_w,
    (SELECT SUM(c * c) FROM unit_stats) AS sum_c2,
    (SELECT SUM(y_sum) FROM unit_stats) AS sum_y,
    (SELECT SUM(wy_sum) FROM unit_stats) AS sum_wy,
    (SELECT SUM(c * y_sum) FROM unit_stats) AS sum_cy,
    (SELECT SUM(d * d) FROM time_stats) AS sum_d2,
    (SELECT SUM(d * y_sum) FROM time_stats) AS sum_dy
"""


COHORT_TIME_SUMS_SQL = """
WITH cohort_time_n AS (
    SELECT
        m.cohort::INTEGER AS cohort,
        t.time_index::INTEGER AS time_index,
        SUM(bit_count(m.mask & t.valid_mask))::DOUBLE AS n_units
    FROM time_bits AS t
    JOIN cohort_masks AS m USING (chunk)
    GROUP BY m.cohort, t.time_index
),
cohort_time_y AS (
    SELECT
        m.cohort::INTEGER AS cohort,
        t.time_index::INTEGER AS time_index,
        SUM(y.outcome_value::DOUBLE * bit_count(y.y_bits & m.mask & t.valid_mask)) AS sum_y
    FROM time_bits AS t
    JOIN cohort_masks AS m USING (chunk)
    JOIN time_outcome_bits AS y USING (time_index, chunk)
    GROUP BY m.cohort, t.time_index
)
SELECT
    n.cohort,
    n.time_index,
    COALESCE(y.sum_y, 0.0) AS sum_y,
    n.n_units
FROM cohort_time_n AS n
LEFT JOIN cohort_time_y AS y USING (cohort, time_index)
ORDER BY n.cohort, n.time_index
"""


@dataclass
class BitmapPanel:
    """Packed representation of a dense balanced panel."""

    unit_ids: np.ndarray
    time_ids: np.ndarray
    y: np.ndarray
    w: np.ndarray
    outcome_values: np.ndarray
    unit_bits: pd.DataFrame
    time_bits: pd.DataFrame
    unit_outcome_bits: pd.DataFrame
    time_outcome_bits: pd.DataFrame
    cohorts: np.ndarray
    cohort_masks: pd.DataFrame
    unit_col: str
    time_col: str
    outcome_col: str
    treatment_col: str

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def n_times(self) -> int:
        return len(self.time_ids)

    @property
    def treated_cohorts(self) -> list[int]:
        return sorted(int(c) for c in np.unique(self.cohorts) if c >= 0)

    @property
    def all_cohorts(self) -> list[int]:
        return sorted(int(c) for c in np.unique(self.cohorts))

    @property
    def outcome_cardinality(self) -> int:
        return len(self.outcome_values)


class LongPanelBitPacker:
    """Convert a balanced long panel into treatment and outcome bitmaps.

    The treatment must be binary.  Outcomes may have low finite cardinality; one
    bitplane is stored per nonzero outcome value by default.
    """

    def __init__(
        self,
        unit_col: str = "unit_id",
        time_col: str = "time_id",
        outcome_col: str = "Y_it",
        treatment_col: str = "W_it",
        max_outcome_cardinality: int = DEFAULT_MAX_OUTCOME_CARDINALITY,
        store_zero_outcome: bool = False,
    ):
        self.unit_col = unit_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.max_outcome_cardinality = max_outcome_cardinality
        self.store_zero_outcome = store_zero_outcome

    def fit_transform(self, data: pd.DataFrame) -> BitmapPanel:
        return self.transform(data)

    def transform(self, data: pd.DataFrame) -> BitmapPanel:
        required = {
            self.unit_col,
            self.time_col,
            self.outcome_col,
            self.treatment_col,
        }
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if data.duplicated([self.unit_col, self.time_col]).any():
            raise ValueError("The long panel has duplicate unit-time rows")

        unit_ids = np.sort(data[self.unit_col].unique())
        time_ids = np.sort(data[self.time_col].unique())
        expected_times = np.arange(len(time_ids))
        if not np.array_equal(time_ids, expected_times):
            raise ValueError("time_col must be zero-based consecutive integers")

        y_wide = (
            data.pivot(
                index=self.unit_col,
                columns=self.time_col,
                values=self.outcome_col,
            )
            .reindex(index=unit_ids, columns=time_ids)
            .to_numpy()
        )
        w_wide = (
            data.pivot(
                index=self.unit_col,
                columns=self.time_col,
                values=self.treatment_col,
            )
            .reindex(index=unit_ids, columns=time_ids)
            .to_numpy()
        )
        if np.isnan(y_wide).any() or np.isnan(w_wide).any():
            raise ValueError("The panel must be balanced over the observed unit and time ids")
        if not np.isin(w_wide, [0, 1]).all():
            raise ValueError("The treatment must be binary")

        outcome_values = np.sort(np.unique(y_wide))
        if len(outcome_values) > self.max_outcome_cardinality:
            raise ValueError(
                "The outcome has "
                f"{len(outcome_values)} distinct values, which exceeds "
                f"max_outcome_cardinality={self.max_outcome_cardinality}"
            )

        y = y_wide.astype(float)
        w = w_wide.astype(np.uint8)
        stored_outcome_values = (
            outcome_values if self.store_zero_outcome else outcome_values[outcome_values != 0]
        )
        cohorts = _first_treated_period(w)
        return BitmapPanel(
            unit_ids=unit_ids,
            time_ids=time_ids,
            y=y,
            w=w,
            outcome_values=outcome_values,
            unit_bits=_pack_treatment_unit_chunks(w),
            time_bits=_pack_treatment_time_chunks(w),
            unit_outcome_bits=_pack_outcome_unit_chunks(y, stored_outcome_values),
            time_outcome_bits=_pack_outcome_time_chunks(y, stored_outcome_values),
            cohorts=cohorts,
            cohort_masks=_pack_cohort_masks(cohorts),
            unit_col=self.unit_col,
            time_col=self.time_col,
            outcome_col=self.outcome_col,
            treatment_col=self.treatment_col,
        )


def pack_long_panel_to_bitmaps(
    data: pd.DataFrame,
    unit_col: str = "unit_id",
    time_col: str = "time_id",
    outcome_col: str = "Y_it",
    treatment_col: str = "W_it",
    max_outcome_cardinality: int = DEFAULT_MAX_OUTCOME_CARDINALITY,
) -> BitmapPanel:
    """Translate a balanced long panel into bitmap tables."""

    return LongPanelBitPacker(
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        max_outcome_cardinality=max_outcome_cardinality,
    ).fit_transform(data)


def bitmap_translation_table(panel: BitmapPanel) -> pd.DataFrame:
    """Format a small panel's unit-major treatment and outcome layout."""

    rows = []
    for row in panel.unit_bits.itertuples(index=False):
        start = int(row.chunk) * CHUNK_BITS
        stop = min(start + CHUNK_BITS, panel.n_times)
        width = stop - start
        unit_index = int(row.unit_index)
        values = panel.y[unit_index, start:stop]
        if _is_binary_support(panel.outcome_values):
            y_display_key = "Y_bitmap"
            y_display = _bitmap_string(_mask_from_bits(values == 1), width)
        else:
            y_display_key = "Y_values"
            y_display = "".join(_format_outcome_value(value) for value in values)
        rows.append(
            {
                "unit_index": unit_index,
                "chunk": int(row.chunk),
                y_display_key: y_display,
                "W_bitmap": _bitmap_string(row.w_bits, width),
                "valid_mask": _bitmap_string(row.valid_mask, width),
            }
        )
    return pd.DataFrame(rows)


def make_five_period_demo_long_panel() -> pd.DataFrame:
    """Small long-panel example used to show the bitmap translation."""

    y = np.array(
        [
            [0, 1, 1, 0, 1],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    w = np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return _arrays_to_long_panel(y, w)


def make_one_shot_binary_adoption_panel(
    num_units: int = 5_000,
    num_periods: int = 30,
    num_treated: int = 2_500,
    treatment_start: int = 15,
    seed: int = 20260629,
) -> pd.DataFrame:
    """Binary-outcome version of the one-shot adoption design."""

    rng = np.random.default_rng(seed)
    treated_units = rng.choice(num_units, size=num_treated, replace=False)
    w = np.zeros((num_units, num_periods), dtype=np.uint8)
    w[treated_units, treatment_start:] = 1

    effect = np.zeros(num_periods)
    post_effect = 0.07 * np.log(2.0 * np.arange(1, num_periods - treatment_start + 1))
    post_effect[8:] = 0.0
    effect[treatment_start:] = post_effect

    unit_score = rng.normal(scale=0.55, size=(num_units, 1))
    time_score = 0.25 * np.sin(np.linspace(0.0, 2.5 * np.pi, num_periods))[None, :]
    baseline_prob = _logistic(-0.8 + unit_score + time_score)
    prob = np.clip(baseline_prob + w * effect[None, :], 0.01, 0.99)
    y = rng.binomial(1, prob).astype(np.uint8)
    return _arrays_to_long_panel(y, w)


def make_staggered_binary_adoption_panel(
    num_units: int = 6_000,
    num_periods: int = 30,
    treatment_start_cohorts: tuple[int, ...] = (10, 15, 20),
    num_treated: tuple[int, ...] = (1_000, 2_000, 1_000),
    seed: int = 20260630,
) -> pd.DataFrame:
    """Binary-outcome version of the staggered adoption event-study design."""

    if len(treatment_start_cohorts) != len(num_treated):
        raise ValueError("treatment_start_cohorts and num_treated must have equal length")
    if sum(num_treated) > num_units:
        raise ValueError("sum(num_treated) cannot exceed num_units")

    rng = np.random.default_rng(seed)
    w = np.zeros((num_units, num_periods), dtype=np.uint8)
    treatment_effect = np.zeros((num_units, num_periods), dtype=float)
    available_units = np.arange(num_units)

    effect_vectors = [
        _cohort_effect_vector(num_periods - cohort, index)
        for index, cohort in enumerate(treatment_start_cohorts)
    ]

    for cohort, n_treated, effect in zip(
        treatment_start_cohorts, num_treated, effect_vectors
    ):
        cohort_units = rng.choice(available_units, size=n_treated, replace=False)
        available_units = np.setdiff1d(available_units, cohort_units, assume_unique=False)
        w[cohort_units, cohort:] = 1
        treatment_effect[cohort_units, cohort:] = effect

    unit_score = rng.normal(scale=0.55, size=(num_units, 1))
    time_score = 0.20 * np.sin(np.linspace(0.0, 3.0 * np.pi, num_periods))[None, :]
    baseline_prob = _logistic(-0.85 + unit_score + time_score)
    prob = np.clip(baseline_prob + treatment_effect, 0.01, 0.99)
    y = rng.binomial(1, prob).astype(np.uint8)
    return _arrays_to_long_panel(y, w)


def _front_loaded_effect(length: int, tail_periods: int = 10) -> np.ndarray:
    tail = min(max(length, 0), tail_periods)
    lead = max(length - tail, 0)
    if lead == 0:
        return np.repeat(0.0, tail)
    return np.r_[np.linspace(0.16, 0.02, lead), np.repeat(0.0, tail)]


def _cohort_effect_vector(length: int, index: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=float)
    pattern = index % 3
    if pattern == 0:
        return _front_loaded_effect(length)
    if pattern == 1:
        effect = 0.06 * np.log(2.0 * np.arange(1, length + 1))
        effect[8:] = 0.0
        return effect
    return 0.06 * np.sin(np.arange(1, length + 1))


def _split_packer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    names = {
        "unit_col",
        "time_col",
        "outcome_col",
        "treatment_col",
        "max_outcome_cardinality",
        "store_zero_outcome",
    }
    return {key: kwargs[key] for key in names if key in kwargs}


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _arrays_to_long_panel(y: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    unit_id = np.repeat(np.arange(y.shape[0], dtype=np.int32), y.shape[1])
    time_id = np.tile(np.arange(y.shape[1], dtype=np.int16), y.shape[0])
    return pd.DataFrame(
        {
            "unit_id": unit_id,
            "time_id": time_id,
            "Y_it": y.reshape(-1).astype(np.int8),
            "W_it": w.reshape(-1).astype(np.int8),
        }
    )


def _mask_from_bits(bits: np.ndarray) -> np.uint64:
    value = 0
    for offset, bit in enumerate(bits):
        if int(bit):
            value |= 1 << offset
    return np.uint64(value)


def _bitmap_string(value: int, width: int) -> str:
    return "".join("1" if (int(value) >> offset) & 1 else "0" for offset in range(width))


def _first_treated_period(w: np.ndarray) -> np.ndarray:
    any_treated = w.any(axis=1)
    first = np.argmax(w == 1, axis=1)
    return np.where(any_treated, first, -1).astype(int)


def _valid_mask(width: int) -> np.uint64:
    return np.uint64((1 << width) - 1)


def _pack_treatment_unit_chunks(w: np.ndarray) -> pd.DataFrame:
    records = []
    n_units, n_times = w.shape
    n_chunks = int(np.ceil(n_times / CHUNK_BITS))
    for unit_index in range(n_units):
        for chunk in range(n_chunks):
            start = chunk * CHUNK_BITS
            stop = min(start + CHUNK_BITS, n_times)
            width = stop - start
            records.append(
                {
                    "unit_index": unit_index,
                    "chunk": chunk,
                    "w_bits": _mask_from_bits(w[unit_index, start:stop]),
                    "valid_mask": _valid_mask(width),
                }
            )
    return pd.DataFrame(records)


def _pack_treatment_time_chunks(w: np.ndarray) -> pd.DataFrame:
    records = []
    n_units, n_times = w.shape
    n_chunks = int(np.ceil(n_units / CHUNK_BITS))
    for time_index in range(n_times):
        for chunk in range(n_chunks):
            start = chunk * CHUNK_BITS
            stop = min(start + CHUNK_BITS, n_units)
            width = stop - start
            records.append(
                {
                    "time_index": time_index,
                    "chunk": chunk,
                    "w_bits": _mask_from_bits(w[start:stop, time_index]),
                    "valid_mask": _valid_mask(width),
                }
            )
    return pd.DataFrame(records)


def _pack_outcome_unit_chunks(y: np.ndarray, outcome_values: np.ndarray) -> pd.DataFrame:
    records = []
    n_units, n_times = y.shape
    n_chunks = int(np.ceil(n_times / CHUNK_BITS))
    for outcome_value in outcome_values:
        for unit_index in range(n_units):
            for chunk in range(n_chunks):
                start = chunk * CHUNK_BITS
                stop = min(start + CHUNK_BITS, n_times)
                mask = _mask_from_bits(y[unit_index, start:stop] == outcome_value)
                if int(mask) != 0:
                    records.append(
                        {
                            "unit_index": unit_index,
                            "chunk": chunk,
                            "outcome_value": float(outcome_value),
                            "y_bits": mask,
                        }
                    )
    return _outcome_bits_frame(records, ["unit_index", "chunk", "outcome_value", "y_bits"])


def _pack_outcome_time_chunks(y: np.ndarray, outcome_values: np.ndarray) -> pd.DataFrame:
    records = []
    n_units, n_times = y.shape
    n_chunks = int(np.ceil(n_units / CHUNK_BITS))
    for outcome_value in outcome_values:
        for time_index in range(n_times):
            for chunk in range(n_chunks):
                start = chunk * CHUNK_BITS
                stop = min(start + CHUNK_BITS, n_units)
                mask = _mask_from_bits(y[start:stop, time_index] == outcome_value)
                if int(mask) != 0:
                    records.append(
                        {
                            "time_index": time_index,
                            "chunk": chunk,
                            "outcome_value": float(outcome_value),
                            "y_bits": mask,
                        }
                    )
    return _outcome_bits_frame(records, ["time_index", "chunk", "outcome_value", "y_bits"])


def _outcome_bits_frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=columns)
    if frame.empty:
        for column in columns:
            if column == "outcome_value":
                frame[column] = pd.Series(dtype=np.float64)
            elif column == "y_bits":
                frame[column] = pd.Series(dtype=np.uint64)
            else:
                frame[column] = pd.Series(dtype=np.int64)
        return frame
    int_columns = [column for column in columns if column not in {"outcome_value", "y_bits"}]
    frame[int_columns] = frame[int_columns].astype(np.int64)
    frame["outcome_value"] = frame["outcome_value"].astype(np.float64)
    frame["y_bits"] = frame["y_bits"].astype(np.uint64)
    return frame


def _pack_cohort_masks(cohorts: np.ndarray) -> pd.DataFrame:
    records = []
    n_units = len(cohorts)
    n_chunks = int(np.ceil(n_units / CHUNK_BITS))
    for cohort in sorted(int(c) for c in np.unique(cohorts)):
        for chunk in range(n_chunks):
            start = chunk * CHUNK_BITS
            stop = min(start + CHUNK_BITS, n_units)
            records.append(
                {
                    "cohort": cohort,
                    "chunk": chunk,
                    "mask": _mask_from_bits(cohorts[start:stop] == cohort),
                }
            )
    return pd.DataFrame(records)


def _bitmap_connection(panel: BitmapPanel):
    con = duckdb.connect()
    con.register("unit_bits", panel.unit_bits)
    con.register("time_bits", panel.time_bits)
    con.register("unit_outcome_bits", panel.unit_outcome_bits)
    con.register("time_outcome_bits", panel.time_outcome_bits)
    con.register("cohort_masks", panel.cohort_masks)
    return con


def _with_popcount_function(sql: str, popcount_function: str) -> str:
    if popcount_function == "bit_count":
        return sql
    return sql.replace("bit_count(", f"{popcount_function}(")


def _event_study_rhs(
    treated_cohorts: list[int], n_times: int
) -> tuple[list[str], dict[str, int]]:
    cohort_cols = [f"cohort_{cohort}" for cohort in treated_cohorts]
    time_cols = [f"time_{time_index}" for time_index in range(n_times)]
    treatment_cols = [
        f"treatment_time_{cohort}_{time_index}"
        for cohort in treated_cohorts
        for time_index in range(n_times)
        if time_index != cohort - 1
    ]
    rhs = ["intercept"] + cohort_cols + time_cols + treatment_cols
    return rhs, {name: idx for idx, name in enumerate(rhs)}


def _event_study_row(
    cohort: int,
    time_index: int,
    treated_cohorts: list[int],
    rhs_index: dict[str, int],
    pre_treat_interactions: bool,
) -> np.ndarray:
    row = np.zeros(len(rhs_index), dtype=float)
    row[rhs_index["intercept"]] = 1.0
    row[rhs_index[f"time_{time_index}"]] = 1.0
    if cohort >= 0:
        row[rhs_index[f"cohort_{cohort}"]] = 1.0
        treatment_name = f"treatment_time_{cohort}_{time_index}"
        if treatment_name in rhs_index and (
            pre_treat_interactions or time_index >= cohort
        ):
            row[rhs_index[treatment_name]] = 1.0
    return row


def _is_binary_support(values: np.ndarray) -> bool:
    return set(values.tolist()).issubset({0, 1, 0.0, 1.0})


def _format_outcome_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"
