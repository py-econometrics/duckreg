"""Run the bitmap Mundlak proof-of-concept benchmark."""

from __future__ import annotations

import argparse
import time

import duckdb

from duckreg.bitpacking import BitmapMundlak, BitmapMundlakEventStudy
from duckreg.bitmap_utils import (
    BitmapPanel,
    COHORT_TIME_SUMS_SQL,
    LongPanelBitPacker,
    STATIC_MUNDLAK_DIRECT_SQL,
    bitmap_translation_table,
    make_five_period_demo_long_panel,
    make_one_shot_binary_adoption_panel,
    make_staggered_binary_adoption_panel,
    pack_long_panel_to_bitmaps,
)

__all__ = [
    "BitmapMundlak",
    "BitmapMundlakEventStudy",
    "BitmapPanel",
    "COHORT_TIME_SUMS_SQL",
    "LongPanelBitPacker",
    "STATIC_MUNDLAK_DIRECT_SQL",
    "bitmap_translation_table",
    "make_five_period_demo_long_panel",
    "make_one_shot_binary_adoption_panel",
    "make_staggered_binary_adoption_panel",
    "pack_long_panel_to_bitmaps",
    "parse_args",
    "run_poc",
]


def run_poc(n_units: int, n_times: int, seed: int) -> None:
    one_shot = make_one_shot_binary_adoption_panel(
        num_units=n_units,
        num_periods=n_times,
        num_treated=n_units // 2,
        treatment_start=n_times // 2,
        seed=seed,
    )
    panel = pack_long_panel_to_bitmaps(one_shot)

    start = time.perf_counter()
    static_beta = BitmapMundlak(panel).fit().point_estimate
    static_seconds = time.perf_counter() - start

    start = time.perf_counter()
    event_study = BitmapMundlakEventStudy(
        panel,
        pre_treat_interactions=True,
    ).fit().point_estimate
    event_seconds = time.perf_counter() - start

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
