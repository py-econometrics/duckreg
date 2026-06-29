"""Direct bitmap moment estimators for dense binary panel Mundlak examples.

The routines in this file assume a balanced panel with a binary treatment and
binary outcome.  They avoid materializing unit-time rows during estimation by
packing outcomes and treatments into 64-bit words, then extracting the low
dimensional moments needed for the Mundlak and event-study designs.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from duckreg.duckreg import wls


CHUNK_BITS = 64


STATIC_MUNDLAK_DIRECT_SQL = """
WITH unit_stats AS (
    SELECT
        unit_index,
        SUM(bit_count(w_bits & valid_mask))::DOUBLE AS c,
        SUM(bit_count(y_bits & valid_mask))::DOUBLE AS y_sum,
        SUM(bit_count(y_bits & w_bits & valid_mask))::DOUBLE AS wy_sum
    FROM unit_bits
    GROUP BY unit_index
),
time_stats AS (
    SELECT
        time_index,
        SUM(bit_count(w_bits & valid_mask))::DOUBLE AS d,
        SUM(bit_count(y_bits & valid_mask))::DOUBLE AS y_sum
    FROM time_bits
    GROUP BY time_index
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
SELECT
    m.cohort::INTEGER AS cohort,
    t.time_index::INTEGER AS time_index,
    SUM(bit_count(t.y_bits & m.mask))::DOUBLE AS sum_y,
    SUM(bit_count(m.mask))::DOUBLE AS n_units
FROM time_bits AS t
JOIN cohort_masks AS m USING (chunk)
GROUP BY m.cohort, t.time_index
ORDER BY m.cohort, t.time_index
"""


@dataclass
class BitmapPanel:
    y: np.ndarray
    w: np.ndarray
    unit_ids: np.ndarray
    time_ids: np.ndarray
    unit_bits: pd.DataFrame
    time_bits: pd.DataFrame
    cohorts: np.ndarray
    cohort_masks: pd.DataFrame

    @property
    def n_units(self) -> int:
        return self.y.shape[0]

    @property
    def n_times(self) -> int:
        return self.y.shape[1]

    @property
    def treated_cohorts(self) -> list[int]:
        return sorted(int(c) for c in np.unique(self.cohorts) if c >= 0)

    @property
    def all_cohorts(self) -> list[int]:
        return sorted(int(c) for c in np.unique(self.cohorts))


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _mask_from_bits(bits: np.ndarray) -> np.uint64:
    value = 0
    for offset, bit in enumerate(bits):
        if int(bit):
            value |= 1 << offset
    return np.uint64(value)


def _bitmap_string(value: int, width: int) -> str:
    """Return a human-readable bitmap with time/unit position 0 on the left."""

    return "".join("1" if (int(value) >> offset) & 1 else "0" for offset in range(width))


def _first_treated_period(w: np.ndarray) -> np.ndarray:
    any_treated = w.any(axis=1)
    first = np.argmax(w == 1, axis=1)
    return np.where(any_treated, first, -1).astype(int)


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
    unit_id = np.repeat(np.arange(y.shape[0]), y.shape[1])
    time_id = np.tile(np.arange(y.shape[1]), y.shape[0])
    return pd.DataFrame(
        {
            "unit_id": unit_id,
            "time_id": time_id,
            "Y_it": y.reshape(-1).astype(int),
            "W_it": w.reshape(-1).astype(int),
        }
    )


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

    rng = np.random.default_rng(seed)
    w = np.zeros((num_units, num_periods), dtype=np.uint8)
    treatment_effect = np.zeros((num_units, num_periods), dtype=float)
    available_units = np.arange(num_units)

    effect_vectors = [
        np.r_[
            np.linspace(0.16, 0.02, num_periods - treatment_start_cohorts[0] - 10),
            np.repeat(0.0, 10),
        ],
        np.r_[
            0.06
            * np.log(2.0 * np.arange(1, num_periods - treatment_start_cohorts[1] + 1))
        ],
        0.06 * np.sin(np.arange(1, num_periods - treatment_start_cohorts[2] + 1)),
    ]
    effect_vectors[1][8:] = 0.0

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


def _arrays_to_long_panel(y: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    unit_id = np.repeat(np.arange(y.shape[0]), y.shape[1])
    time_id = np.tile(np.arange(y.shape[1]), y.shape[0])
    return pd.DataFrame(
        {
            "unit_id": unit_id,
            "time_id": time_id,
            "Y_it": y.reshape(-1).astype(int),
            "W_it": w.reshape(-1).astype(int),
        }
    )


def pack_long_panel_to_bitmaps(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    time_col: str = "time_id",
    outcome_col: str = "Y_it",
    treatment_col: str = "W_it",
) -> BitmapPanel:
    """Translate a balanced binary long panel into unit- and time-major bitmaps."""

    duplicate_rows = df.duplicated([unit_col, time_col]).any()
    if duplicate_rows:
        raise ValueError("The long panel has duplicate unit-time rows")

    unit_ids = np.sort(df[unit_col].unique())
    time_ids = np.sort(df[time_col].unique())
    expected_times = np.arange(len(time_ids))
    if not np.array_equal(time_ids, expected_times):
        raise ValueError("This proof of concept expects zero-based consecutive time ids")

    y_wide = (
        df.pivot(index=unit_col, columns=time_col, values=outcome_col)
        .reindex(index=unit_ids, columns=time_ids)
        .to_numpy()
    )
    w_wide = (
        df.pivot(index=unit_col, columns=time_col, values=treatment_col)
        .reindex(index=unit_ids, columns=time_ids)
        .to_numpy()
    )
    if np.isnan(y_wide).any() or np.isnan(w_wide).any():
        raise ValueError("The panel must be balanced over the observed unit and time ids")
    if not np.isin(y_wide, [0, 1]).all():
        raise ValueError("The outcome must be binary")
    if not np.isin(w_wide, [0, 1]).all():
        raise ValueError("The treatment must be binary")

    y = y_wide.astype(np.uint8)
    w = w_wide.astype(np.uint8)
    cohorts = _first_treated_period(w)
    unit_bits = _pack_unit_chunks(y, w)
    time_bits = _pack_time_chunks(y, w)
    cohort_masks = _pack_cohort_masks(cohorts)
    return BitmapPanel(
        y=y,
        w=w,
        unit_ids=unit_ids,
        time_ids=time_ids,
        unit_bits=unit_bits,
        time_bits=time_bits,
        cohorts=cohorts,
        cohort_masks=cohort_masks,
    )


def _pack_unit_chunks(y: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    records = []
    n_units, n_times = y.shape
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
                    "y_bits": _mask_from_bits(y[unit_index, start:stop]),
                    "w_bits": _mask_from_bits(w[unit_index, start:stop]),
                    "valid_mask": np.uint64((1 << width) - 1),
                }
            )
    return pd.DataFrame(records)


def _pack_time_chunks(y: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    records = []
    n_units, n_times = y.shape
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
                    "y_bits": _mask_from_bits(y[start:stop, time_index]),
                    "w_bits": _mask_from_bits(w[start:stop, time_index]),
                    "valid_mask": np.uint64((1 << width) - 1),
                }
            )
    return pd.DataFrame(records)


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


def bitmap_translation_table(panel: BitmapPanel) -> pd.DataFrame:
    """Format the unit-major bitmap table for small examples."""

    rows = []
    for row in panel.unit_bits.itertuples(index=False):
        width = panel.n_times if row.chunk == 0 else CHUNK_BITS
        rows.append(
            {
                "unit_index": row.unit_index,
                "chunk": row.chunk,
                "Y_bitmap": _bitmap_string(row.y_bits, min(width, panel.n_times)),
                "W_bitmap": _bitmap_string(row.w_bits, min(width, panel.n_times)),
                "valid_mask": _bitmap_string(row.valid_mask, min(width, panel.n_times)),
            }
        )
    return pd.DataFrame(rows)


def _register_bitmap_tables(con: duckdb.DuckDBPyConnection, panel: BitmapPanel) -> None:
    con.register("unit_bits", panel.unit_bits)
    con.register("time_bits", panel.time_bits)
    con.register("cohort_masks", panel.cohort_masks)


def fit_static_mundlak_bitmap(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    time_col: str = "time_id",
    outcome_col: str = "Y_it",
    treatment_col: str = "W_it",
) -> pd.Series:
    """Fit the static two-way Mundlak LPM from direct bitmap moments."""

    panel = pack_long_panel_to_bitmaps(df, unit_col, time_col, outcome_col, treatment_col)
    return fit_static_mundlak_bitmap_panel(panel, treatment_col=treatment_col)


def fit_static_mundlak_bitmap_panel(
    panel: BitmapPanel,
    treatment_col: str = "W_it",
) -> pd.Series:
    """Fit the static two-way Mundlak LPM from an already packed panel."""

    con = duckdb.connect()
    _register_bitmap_tables(con, panel)
    stats = con.execute(STATIC_MUNDLAK_DIRECT_SQL).fetchone()
    sum_w, sum_c2, sum_y, sum_wy, sum_cy, sum_d2, sum_dy = map(float, stats)

    n_units = float(panel.n_units)
    n_times = float(panel.n_times)
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
        index=[
            "Intercept",
            treatment_col,
            f"avg_{treatment_col}_unit",
            f"avg_{treatment_col}_time",
        ],
    )


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


def fit_event_study_bitmap(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    time_col: str = "time_id",
    outcome_col: str = "Y_it",
    treatment_col: str = "W_it",
    pre_treat_interactions: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fit the Mundlak event-study design from bitmap cohort-time moments."""

    panel = pack_long_panel_to_bitmaps(df, unit_col, time_col, outcome_col, treatment_col)
    return fit_event_study_bitmap_panel(
        panel, pre_treat_interactions=pre_treat_interactions
    )


def fit_event_study_bitmap_panel(
    panel: BitmapPanel,
    pre_treat_interactions: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fit the Mundlak event-study design from an already packed panel."""

    con = duckdb.connect()
    _register_bitmap_tables(con, panel)
    cohort_time = con.execute(COHORT_TIME_SUMS_SQL).fetchdf()

    treated_cohorts = panel.treated_cohorts
    rhs, rhs_index = _event_study_rhs(treated_cohorts, panel.n_times)
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
    for cohort in panel.all_cohorts:
        n_units = n_lookup[cohort]
        for time_index in range(panel.n_times):
            rows.append(
                _event_study_row(
                    cohort,
                    time_index,
                    treated_cohorts,
                    rhs_index,
                    pre_treat_interactions,
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
    for cohort in treated_cohorts:
        values = []
        index = []
        for time_index in range(panel.n_times):
            name = f"treatment_time_{cohort}_{time_index}"
            index.append(name)
            if time_index == cohort - 1:
                values.append(0.0)
            else:
                values.append(float(estimates[name]))
        result[str(cohort)] = pd.DataFrame({"est": values}, index=index)
    return result


def run_poc(n_units: int, n_times: int, seed: int) -> None:
    one_shot = make_one_shot_binary_adoption_panel(
        num_units=n_units,
        num_periods=n_times,
        num_treated=n_units // 2,
        treatment_start=n_times // 2,
        seed=seed,
    )

    start = time.perf_counter()
    static_beta = fit_static_mundlak_bitmap(one_shot)
    static_seconds = time.perf_counter() - start

    start = time.perf_counter()
    event_study = fit_event_study_bitmap(one_shot, pre_treat_interactions=True)
    event_seconds = time.perf_counter() - start

    panel = pack_long_panel_to_bitmaps(one_shot)
    print(f"DuckDB version: {duckdb.__version__}")
    print(f"Panel shape: {panel.n_units:,} units x {panel.n_times:,} time periods")
    print(f"Long panel rows: {len(one_shot):,}")
    print(f"Unit-major bitmap rows: {len(panel.unit_bits):,}")
    print(f"Time-major bitmap rows: {len(panel.time_bits):,}")
    print(f"Static Mundlak bitmap seconds: {static_seconds:.4f}")
    print(f"Event-study bitmap seconds: {event_seconds:.4f}")
    print("Static Mundlak coefficients:")
    print(static_beta.to_string())
    first_cohort = next(iter(event_study))
    print(f"First event-study cohort: {first_cohort}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct bitmap Mundlak estimators on a synthetic binary panel."
    )
    parser.add_argument("--units", type=int, default=2_000)
    parser.add_argument("--times", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_poc(n_units=args.units, n_times=args.times, seed=args.seed)
