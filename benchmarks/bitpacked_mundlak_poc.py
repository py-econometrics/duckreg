"""DuckDB proof of concept for bit-packed binary panel Mundlak compression.

The example keeps a balanced binary panel in one row per unit and chunk.  It
then uses DuckDB's `bit_count` and bitwise operators to produce the same
compressed Mundlak sufficient statistics as a materialized long-panel query.
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


MATERIALIZED_COMPRESSION_SQL = """
WITH unit_counts AS (
    SELECT unit, SUM(w)::INTEGER AS c
    FROM long_panel
    GROUP BY unit
),
time_counts AS (
    SELECT time_index, SUM(w)::INTEGER AS d
    FROM long_panel
    GROUP BY time_index
)
SELECT
    p.w::INTEGER AS w,
    u.c::INTEGER AS c,
    t.d::INTEGER AS d,
    COUNT(*)::BIGINT AS count,
    SUM(p.y)::BIGINT AS sum_y
FROM long_panel AS p
JOIN unit_counts AS u USING (unit)
JOIN time_counts AS t USING (time_index)
GROUP BY p.w, u.c, t.d
ORDER BY w, c, d
"""


BITPACKED_COMPRESSION_SQL = """
WITH unit_counts AS (
    SELECT
        unit,
        SUM(bit_count(w_bits & valid_mask))::INTEGER AS c
    FROM unit_bits
    GROUP BY unit
),
packed_cells AS (
    SELECT
        1::INTEGER AS w,
        u.c::INTEGER AS c,
        m.d::INTEGER AS d,
        SUM(bit_count(b.w_bits & m.mask))::BIGINT AS count,
        SUM(bit_count(b.y_bits & b.w_bits & m.mask))::BIGINT AS sum_y
    FROM unit_bits AS b
    JOIN unit_counts AS u USING (unit)
    JOIN time_masks AS m USING (chunk)
    GROUP BY u.c, m.d

    UNION ALL

    SELECT
        0::INTEGER AS w,
        u.c::INTEGER AS c,
        m.d::INTEGER AS d,
        SUM(bit_count((~b.w_bits) & m.mask))::BIGINT AS count,
        SUM(bit_count(b.y_bits & (~b.w_bits) & m.mask))::BIGINT AS sum_y
    FROM unit_bits AS b
    JOIN unit_counts AS u USING (unit)
    JOIN time_masks AS m USING (chunk)
    GROUP BY u.c, m.d
)
SELECT w, c, d, count, sum_y
FROM packed_cells
WHERE count > 0
ORDER BY w, c, d
"""


DIRECT_MOMENTS_SQL = """
WITH unit_stats AS (
    SELECT
        unit,
        SUM(bit_count(w_bits & valid_mask))::DOUBLE AS c,
        SUM(bit_count(y_bits & valid_mask))::DOUBLE AS y_sum,
        SUM(bit_count(y_bits & w_bits & valid_mask))::DOUBLE AS wy_sum
    FROM unit_bits
    GROUP BY unit
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


@dataclass
class PanelData:
    y: np.ndarray
    w: np.ndarray

    @property
    def n_units(self) -> int:
        return self.y.shape[0]

    @property
    def n_times(self) -> int:
        return self.y.shape[1]


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_binary_panel(n_units: int, n_times: int, seed: int) -> PanelData:
    rng = np.random.default_rng(seed)
    unit_score = rng.normal(scale=0.9, size=(n_units, 1))
    time_score = 0.35 * np.sin(np.linspace(0.0, 4.0 * np.pi, n_times))[None, :]
    w_prob = _logistic(-0.4 + unit_score + time_score)
    w = rng.binomial(1, w_prob).astype(np.uint8)

    unit_effect = rng.normal(scale=0.4, size=(n_units, 1))
    time_effect = 0.25 * np.cos(np.linspace(0.0, 2.0 * np.pi, n_times))[None, :]
    y_prob = _logistic(-1.2 + 0.8 * w + unit_effect + time_effect)
    y = rng.binomial(1, y_prob).astype(np.uint8)
    return PanelData(y=y, w=w)


def _mask_from_bits(bits: np.ndarray) -> np.uint64:
    value = 0
    for offset, bit in enumerate(bits):
        if int(bit):
            value |= 1 << offset
    return np.uint64(value)


def pack_unit_chunks(panel: PanelData) -> pd.DataFrame:
    records = []
    n_chunks = int(np.ceil(panel.n_times / CHUNK_BITS))
    for unit in range(panel.n_units):
        for chunk in range(n_chunks):
            start = chunk * CHUNK_BITS
            stop = min(start + CHUNK_BITS, panel.n_times)
            width = stop - start
            valid_mask = np.uint64((1 << width) - 1)
            records.append(
                {
                    "unit": unit,
                    "chunk": chunk,
                    "y_bits": _mask_from_bits(panel.y[unit, start:stop]),
                    "w_bits": _mask_from_bits(panel.w[unit, start:stop]),
                    "valid_mask": valid_mask,
                }
            )
    return pd.DataFrame(records)


def pack_time_chunks(panel: PanelData) -> pd.DataFrame:
    records = []
    n_chunks = int(np.ceil(panel.n_units / CHUNK_BITS))
    for time_index in range(panel.n_times):
        for chunk in range(n_chunks):
            start = chunk * CHUNK_BITS
            stop = min(start + CHUNK_BITS, panel.n_units)
            width = stop - start
            valid_mask = np.uint64((1 << width) - 1)
            records.append(
                {
                    "time_index": time_index,
                    "chunk": chunk,
                    "y_bits": _mask_from_bits(panel.y[start:stop, time_index]),
                    "w_bits": _mask_from_bits(panel.w[start:stop, time_index]),
                    "valid_mask": valid_mask,
                }
            )
    return pd.DataFrame(records)


def make_time_masks(panel: PanelData) -> pd.DataFrame:
    time_counts = panel.w.sum(axis=0).astype(np.int64)
    records = []
    n_chunks = int(np.ceil(panel.n_times / CHUNK_BITS))
    for chunk in range(n_chunks):
        start = chunk * CHUNK_BITS
        stop = min(start + CHUNK_BITS, panel.n_times)
        for d in np.unique(time_counts[start:stop]):
            bits = time_counts[start:stop] == d
            records.append(
                {
                    "chunk": chunk,
                    "d": int(d),
                    "mask": _mask_from_bits(bits),
                }
            )
    return pd.DataFrame(records)


def make_long_panel(panel: PanelData) -> pd.DataFrame:
    unit = np.repeat(np.arange(panel.n_units, dtype=np.int64), panel.n_times)
    time_index = np.tile(np.arange(panel.n_times, dtype=np.int64), panel.n_units)
    return pd.DataFrame(
        {
            "unit": unit,
            "time_index": time_index,
            "y": panel.y.reshape(-1).astype(np.int64),
            "w": panel.w.reshape(-1).astype(np.int64),
        }
    )


def materialized_compression(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(MATERIALIZED_COMPRESSION_SQL).fetchdf()


def bitpacked_compression(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(BITPACKED_COMPRESSION_SQL).fetchdf()


def fit_mundlak_from_compression(df: pd.DataFrame, n_units: int, n_times: int) -> np.ndarray:
    ordered = df.sort_values(["w", "c", "d"]).reset_index(drop=True)
    x = np.column_stack(
        [
            np.ones(len(ordered)),
            ordered["w"].to_numpy(dtype=float),
            ordered["c"].to_numpy(dtype=float) / n_times,
            ordered["d"].to_numpy(dtype=float) / n_units,
        ]
    )
    y = (ordered["sum_y"] / ordered["count"]).to_numpy(dtype=float).reshape(-1, 1)
    n = ordered["count"].to_numpy(dtype=float)
    return wls(x, y, n).reshape(-1)


def fit_mundlak_from_bitpacked_moments(
    con: duckdb.DuckDBPyConnection, n_units: int, n_times: int
) -> np.ndarray:
    stats = con.execute(DIRECT_MOMENTS_SQL).fetchone()

    sum_w, sum_c2, sum_y, sum_wy, sum_cy, sum_d2, sum_dy = map(float, stats)
    n_units = float(n_units)
    n_times = float(n_times)
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
    return np.linalg.lstsq(xtx, xty, rcond=None)[0]


def run_poc(n_units: int, n_times: int, seed: int) -> None:
    panel = make_binary_panel(n_units=n_units, n_times=n_times, seed=seed)
    con = duckdb.connect()

    start = time.perf_counter()
    long_panel = make_long_panel(panel)
    unit_bits = pack_unit_chunks(panel)
    time_masks = make_time_masks(panel)
    time_bits = pack_time_chunks(panel)
    setup_seconds = time.perf_counter() - start

    con.register("long_panel", long_panel)
    con.register("unit_bits", unit_bits)
    con.register("time_masks", time_masks)
    con.register("time_bits", time_bits)

    start = time.perf_counter()
    row_compressed = materialized_compression(con)
    row_seconds = time.perf_counter() - start

    start = time.perf_counter()
    bit_compressed = bitpacked_compression(con)
    bit_seconds = time.perf_counter() - start

    pd.testing.assert_frame_equal(
        row_compressed.reset_index(drop=True),
        bit_compressed.reset_index(drop=True),
        check_dtype=False,
    )

    row_beta = fit_mundlak_from_compression(row_compressed, n_units, n_times)
    bit_beta = fit_mundlak_from_compression(bit_compressed, n_units, n_times)
    np.testing.assert_allclose(row_beta, bit_beta, rtol=0.0, atol=1e-12)

    start = time.perf_counter()
    direct_beta = fit_mundlak_from_bitpacked_moments(con, n_units, n_times)
    direct_seconds = time.perf_counter() - start
    np.testing.assert_allclose(row_beta, direct_beta, rtol=0.0, atol=1e-11)

    n_chunks = int(np.ceil(n_times / CHUNK_BITS))
    transposed_chunks = int(np.ceil(n_units / CHUNK_BITS))
    print(f"DuckDB version: {duckdb.__version__}")
    print(f"Panel shape: {n_units:,} units x {n_times:,} time periods")
    print(f"Packed representation: {n_units * n_chunks:,} unit-chunk rows")
    print(f"Transposed packed representation: {n_times * transposed_chunks:,} time-chunk rows")
    print(f"Materialized representation: {n_units * n_times:,} panel rows")
    print(f"Compressed cells: {len(bit_compressed):,}")
    print(f"Representation setup seconds: {setup_seconds:.4f}")
    print(f"Materialized compression seconds: {row_seconds:.4f}")
    print(f"Bitpacked compression seconds: {bit_seconds:.4f}")
    print(f"Direct bitpacked moment seconds: {direct_seconds:.4f}")
    print("Mundlak coefficients from compressed cells:")
    print(
        pd.Series(
            bit_beta,
            index=["intercept", "w", "avg_w_unit", "avg_w_time"],
        ).to_string()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare long-panel and bit-packed DuckDB Mundlak compression."
    )
    parser.add_argument("--units", type=int, default=2_000)
    parser.add_argument("--times", type=int, default=365)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_poc(n_units=args.units, n_times=args.times, seed=args.seed)
