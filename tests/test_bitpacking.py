import ibis
import numpy as np
import pandas as pd

import duckreg
from duckreg.bitpacking import (
    BitmapMundlak,
    BitmapMundlakEventStudy,
)
from duckreg.bitmap_utils import (
    LongPanelBitPacker,
    bitmap_translation_table,
    make_one_shot_binary_adoption_panel,
    make_staggered_binary_adoption_panel,
)
from duckreg.dbreg import DBMundlak, DBMundlakEventStudy


def _make_con(tmp_path, data, table="data"):
    con = ibis.duckdb.connect(tmp_path / "bitpacking.db")
    con.create_table(table, data, overwrite=True)
    return con


def test_bitpacking_exports():
    assert duckreg.BitmapMundlak is BitmapMundlak
    assert duckreg.BitmapMundlakEventStudy is BitmapMundlakEventStudy


def test_long_panel_bitpacker_supports_low_cardinality_outcomes():
    data = pd.DataFrame(
        {
            "unit_id": np.repeat(np.arange(4), 5),
            "time_id": np.tile(np.arange(5), 4),
            "Y_it": [
                0,
                1,
                2,
                3,
                0,
                1,
                1,
                0,
                2,
                3,
                2,
                0,
                3,
                1,
                0,
                3,
                2,
                1,
                0,
                2,
            ],
            "W_it": [
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    )
    panel = LongPanelBitPacker(max_outcome_cardinality=4).fit_transform(data)

    np.testing.assert_array_equal(panel.outcome_values, np.array([0, 1, 2, 3]))
    assert set(panel.unit_outcome_bits["outcome_value"]) == {1.0, 2.0, 3.0}

    table = bitmap_translation_table(panel)
    assert "Y_values" in table.columns
    assert table.loc[0, "Y_values"] == "01230"


def test_bitmap_mundlak_matches_dbmundlak_for_binary_outcome(tmp_path):
    data = make_one_shot_binary_adoption_panel(
        num_units=120,
        num_treated=60,
        num_periods=12,
        treatment_start=5,
        seed=123,
    )
    bitmap = BitmapMundlak.from_long(data).fit()

    con = _make_con(tmp_path, data)
    db = DBMundlak(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y_it",
        covariates=["W_it"],
        unit_col="unit_id",
        time_col="time_id",
        cluster_col="unit_id",
        seed=42,
        n_bootstraps=0,
    )
    db.fit()

    np.testing.assert_allclose(
        bitmap.point_estimate.to_numpy(),
        db.point_estimate.reshape(-1),
        rtol=1e-10,
        atol=1e-10,
    )


def test_bitmap_mundlak_matches_dbmundlak_for_low_cardinality_outcome(tmp_path):
    rng = np.random.default_rng(456)
    n_units = 90
    n_times = 8
    unit = np.repeat(np.arange(n_units), n_times)
    time = np.tile(np.arange(n_times), n_units)
    treatment = ((unit % 3 == 0) & (time >= 3)).astype(int)
    y = rng.integers(0, 4, size=len(unit))
    data = pd.DataFrame(
        {
            "unit_id": unit,
            "time_id": time,
            "Y_it": y,
            "W_it": treatment,
        }
    )
    panel = LongPanelBitPacker(max_outcome_cardinality=4).fit_transform(data)
    bitmap = BitmapMundlak(panel).fit()

    con = _make_con(tmp_path, data)
    db = DBMundlak(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y_it",
        covariates=["W_it"],
        unit_col="unit_id",
        time_col="time_id",
        cluster_col="unit_id",
        seed=42,
        n_bootstraps=0,
    )
    db.fit()

    np.testing.assert_allclose(
        bitmap.point_estimate.to_numpy(),
        db.point_estimate.reshape(-1),
        rtol=1e-10,
        atol=1e-10,
    )


def test_bitmap_event_study_matches_dbmundlak_event_study(tmp_path):
    data = make_staggered_binary_adoption_panel(
        num_units=120,
        num_periods=12,
        treatment_start_cohorts=(4, 7),
        num_treated=(35, 45),
        seed=789,
    )
    bitmap = BitmapMundlakEventStudy.from_long(
        data,
        pre_treat_interactions=True,
    ).fit()

    con = _make_con(tmp_path, data)
    db = DBMundlakEventStudy(
        db_name=None,
        connection=con,
        table_name="data",
        outcome_var="Y_it",
        treatment_col="W_it",
        unit_col="unit_id",
        time_col="time_id",
        cluster_col="unit_id",
        seed=42,
        pre_treat_interactions=True,
        n_bootstraps=0,
    )
    db.fit()

    assert bitmap.point_estimate.keys() == db.point_estimate.keys()
    for cohort in bitmap.point_estimate:
        np.testing.assert_allclose(
            bitmap.point_estimate[cohort]["est"].to_numpy(),
            db.point_estimate[cohort]["est"].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
