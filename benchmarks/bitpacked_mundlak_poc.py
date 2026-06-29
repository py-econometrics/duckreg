"""Compatibility wrapper for the bitmap Mundlak proof-of-concept benchmark."""

from duckreg.bitpacking import (
    BitmapMundlak,
    BitmapMundlakEventStudy,
    BitmapPanel,
    COHORT_TIME_SUMS_SQL,
    LongPanelBitPacker,
    STATIC_MUNDLAK_DIRECT_SQL,
    bitmap_translation_table,
    fit_event_study_bitmap,
    fit_event_study_bitmap_panel,
    fit_static_mundlak_bitmap,
    fit_static_mundlak_bitmap_panel,
    make_five_period_demo_long_panel,
    make_one_shot_binary_adoption_panel,
    make_staggered_binary_adoption_panel,
    pack_long_panel_to_bitmaps,
    parse_args,
    run_poc,
)

__all__ = [
    "BitmapMundlak",
    "BitmapMundlakEventStudy",
    "BitmapPanel",
    "COHORT_TIME_SUMS_SQL",
    "LongPanelBitPacker",
    "STATIC_MUNDLAK_DIRECT_SQL",
    "bitmap_translation_table",
    "fit_event_study_bitmap",
    "fit_event_study_bitmap_panel",
    "fit_static_mundlak_bitmap",
    "fit_static_mundlak_bitmap_panel",
    "make_five_period_demo_long_panel",
    "make_one_shot_binary_adoption_panel",
    "make_staggered_binary_adoption_panel",
    "pack_long_panel_to_bitmaps",
    "parse_args",
    "run_poc",
]


if __name__ == "__main__":
    args = parse_args()
    run_poc(n_units=args.units, n_times=args.times, seed=args.seed)
