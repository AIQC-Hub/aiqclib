"""Unit tests for the density inversion and temp-to-psal QC item classes.

Covers the Phase 5 items: RTQC14 density inversion (sigma-0 comparison at
consecutive levels in both directions, joint temp+psal flagging) and the
temperature-to-salinity flag propagation rule. The density expectations
are anchored to hand-checked sigma-0 values: for psal 35, temp
[10, 20, 12] gives sigma-0 [26.953, 24.764, 26.590], so the first pair
inverts by ~2.19 kg/m3 while temp [10, 10.05, 10.02] inverts by only
~0.008 kg/m3.
"""

from datetime import datetime

import polars as pl

from aiqclib.common.loader.feature_registry import FEATURE_REGISTRY
from aiqclib.prepare.features.qc_density_inversion import QCDensityInversion
from aiqclib.prepare.features.qc_temp_to_psal import QCTempToPsal

VALID_TS = datetime(2023, 6, 1, 12, 0)


def make_profile(
    temp: list,
    psal: list = None,
    pres: list = None,
    profile_no: int = 1,
) -> pl.DataFrame:
    """Build one synthetic profile with the standard input columns."""
    n = len(temp)
    psal = psal if psal is not None else [35.0] * n
    pres = pres if pres is not None else [float(10 * i) for i in range(1, n + 1)]
    return pl.DataFrame(
        {
            "platform_code": ["P1"] * n,
            "profile_no": [profile_no] * n,
            "observation_no": list(range(1, n + 1)),
            "profile_timestamp": pl.Series([VALID_TS] * n, dtype=pl.Datetime("ms")),
            "longitude": [15.0] * n,
            "latitude": [55.0] * n,
            "pres": pres,
            "temp": temp,
            "psal": psal,
        }
    )


def run_item(cls, df: pl.DataFrame, feature_info: dict = None) -> pl.DataFrame:
    """Instantiate a QC item in full-frame mode and return its features."""
    ds = cls(feature_info=feature_info, filtered_input=df)
    ds.extract_features()
    return ds.features


# ---------------------------------------------------------------------------
# RTQC14: density inversion
# ---------------------------------------------------------------------------


class TestDensityInversion:
    """RTQC14 density inversion test."""

    def test_stable_profile_passes(self):
        """Density increasing with depth (cooling profile) passes."""
        df = make_profile(temp=[20.0, 15.0, 10.0])
        flags = run_item(QCDensityInversion, df)
        assert flags["temp_qc_density_inversion"].to_list() == [1, 1, 1]
        assert flags["psal_qc_density_inversion"].to_list() == [1, 1, 1]

    def test_inversion_flags_both_levels_of_the_pair(self):
        """A warm intrusion inverts the first pair: both levels flagged.

        sigma-0 = [26.953, 24.764, 26.590]: level 2 is lighter than level 1
        (top-to-bottom fail), and level 1 is denser than level 2
        (bottom-to-top fail). Level 3 is denser than level 2 and last, so
        it passes.
        """
        df = make_profile(temp=[10.0, 20.0, 12.0])
        flags = run_item(QCDensityInversion, df)
        assert flags["temp_qc_density_inversion"].to_list() == [4, 4, 1]
        assert flags["psal_qc_density_inversion"].to_list() == [4, 4, 1]

    def test_temp_and_psal_flags_identical(self):
        """The density flag applies to both variables jointly."""
        df = make_profile(temp=[10.0, 20.0, 12.0])
        flags = run_item(QCDensityInversion, df)
        assert (
            flags["temp_qc_density_inversion"].to_list()
            == flags["psal_qc_density_inversion"].to_list()
        )

    def test_small_inversion_below_threshold_allowed(self):
        """A ~0.008 kg/m3 inversion stays under the default 0.03 threshold."""
        df = make_profile(temp=[10.0, 10.05, 10.02])
        flags = run_item(QCDensityInversion, df)
        assert flags["temp_qc_density_inversion"].to_list() == [1, 1, 1]

    def test_threshold_param_override(self):
        """The same small inversion fails with a tighter threshold."""
        df = make_profile(temp=[10.0, 10.05, 10.02])
        flags = run_item(QCDensityInversion, df, {"params": {"threshold": 0.005}})
        assert flags["temp_qc_density_inversion"].to_list() == [4, 4, 1]

    def test_null_values_untestable(self):
        """Levels with missing inputs cannot be density-checked and pass."""
        df = make_profile(temp=[10.0, None, 12.0])
        flags = run_item(QCDensityInversion, df)
        assert flags["temp_qc_density_inversion"].to_list() == [1, 1, 1]

    def test_profiles_independent(self):
        """Densities are not compared across profile boundaries."""
        df = pl.concat(
            [
                make_profile(temp=[10.0, 9.0], profile_no=1),
                make_profile(temp=[25.0, 24.0], profile_no=2),
            ]
        )
        flags = run_item(QCDensityInversion, df)
        assert flags["temp_qc_density_inversion"].to_list() == [1, 1, 1, 1]

    def test_col_names_limit_output(self):
        """An explicit col_names list restricts the produced columns."""
        df = make_profile(temp=[10.0, 20.0, 12.0])
        flags = run_item(QCDensityInversion, df, {"col_names": ["temp"]})
        assert "temp_qc_density_inversion" in flags.columns
        assert "psal_qc_density_inversion" not in flags.columns

    def test_no_sigma_column_leaks(self):
        """The internal sigma-0 helper column is not part of the output."""
        df = make_profile(temp=[10.0, 20.0, 12.0])
        flags = run_item(QCDensityInversion, df)
        assert flags.columns == [
            "platform_code",
            "profile_no",
            "observation_no",
            "temp_qc_density_inversion",
            "psal_qc_density_inversion",
        ]


# ---------------------------------------------------------------------------
# Temp -> psal flag propagation
# ---------------------------------------------------------------------------


class TestTempToPsal:
    """Temperature-to-salinity flag propagation."""

    def _with_temp_flag(self, temp_flags: list) -> pl.DataFrame:
        """A frame carrying an aggregated temperature flag column."""
        df = make_profile(temp=[5.0] * len(temp_flags))
        return df.with_columns(pl.Series("temp_nrt_flag", temp_flags, dtype=pl.Int64))

    def test_propagates_severity(self):
        """Flags 3 and 4 propagate with their severity; 1 does not."""
        df = self._with_temp_flag([1, 3, 4, None])
        flags = run_item(QCTempToPsal, df)
        assert flags["psal_qc_temp_to_psal"].to_list() == [1, 3, 4, 1]

    def test_output_never_null(self):
        df = self._with_temp_flag([None, None])
        flags = run_item(QCTempToPsal, df)
        assert flags["psal_qc_temp_to_psal"].null_count() == 0

    def test_source_column_override(self):
        """The aggregated-flag column name is configurable."""
        df = make_profile(temp=[5.0, 5.0]).with_columns(
            pl.Series("temp_final", [4, 1], dtype=pl.Int64)
        )
        flags = run_item(QCTempToPsal, df, {"params": {"source_column": "temp_final"}})
        assert flags["psal_qc_temp_to_psal"].to_list() == [4, 1]

    def test_target_variable_override(self):
        """The propagated column follows the configured target variable."""
        df = self._with_temp_flag([4])
        flags = run_item(QCTempToPsal, df, {"params": {"target_variable": "cndc"}})
        assert flags["cndc_qc_temp_to_psal"].to_list() == [4]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistryEntries:
    """The Phase 5 items are registered under qc_-prefixed names."""

    def test_registry_entries(self):
        assert FEATURE_REGISTRY["qc_density_inversion"] is QCDensityInversion
        assert FEATURE_REGISTRY["qc_temp_to_psal"] is QCTempToPsal
