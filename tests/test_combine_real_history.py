"""Tests for the combined real World Cup history (1990-2026) loader."""

import pytest
import pandas as pd

from src.data.combine_real_history import load_full_real_history, HISTORICAL_FILES


class TestLoadFullRealHistory:
    def test_loads_without_error(self):
        combined = load_full_real_history()
        assert len(combined) > 0

    def test_includes_all_nine_historical_tournaments(self):
        combined = load_full_real_history(include_2026=False)
        years_present = set(combined["year"].unique())
        expected_years = {1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022}
        assert expected_years.issubset(years_present)

    def test_total_historical_match_count_matches_real_record(self):
        """
        Verified real fact: 1990 and 1994 had 52 matches each (24-team era),
        1998-2022 had 64 matches each (32-team era). Total = 52*2 + 64*7 = 552.
        """
        combined = load_full_real_history(include_2026=False)
        assert len(combined) == 552

    def test_sorted_chronologically(self):
        combined = load_full_real_history()
        assert combined["date"].is_monotonic_increasing

    def test_can_exclude_2026(self):
        with_2026 = load_full_real_history(include_2026=True)
        without_2026 = load_full_real_history(include_2026=False)
        assert len(with_2026) > len(without_2026)

    def test_1990_opener_present_in_combined_data(self):
        """Verified real fact: 1990 World Cup opened with Argentina-Cameroon."""
        combined = load_full_real_history(include_2026=False)
        year_1990 = combined[combined["year"] == 1990].sort_values("date")
        first_match = year_1990.iloc[0]
        assert first_match["date"] == pd.Timestamp("1990-06-08")

    def test_2022_final_present(self):
        """Verified real fact: 2022 final was Argentina vs France, 3-3."""
        combined = load_full_real_history(include_2026=False)
        final_2022 = combined[
            (combined["year"] == 2022) & (combined["stage"] == "Final")
        ]
        assert len(final_2022) == 1
        match = final_2022.iloc[0]
        assert {match["home_team"], match["away_team"]} == {"Argentina", "France"}
        assert match["home_score"] == 3
        assert match["away_score"] == 3

    def test_no_match_has_missing_year(self):
        """
        Regression test: every real match must have a populated year value.
        Catches the class of bug where wc2026_real_results.csv rows were
        written with a blank year field (silently became NaN on load).
        """
        combined = load_full_real_history()
        assert combined["year"].notna().all()

    def test_2026_matches_all_tagged_2026(self):
        combined = load_full_real_history(include_2026=True)
        year_2026 = combined[combined["date"] >= pd.Timestamp("2026-01-01")]
        assert (year_2026["year"] == 2026).all()

    def test_all_expected_files_exist(self):
        from pathlib import Path
        raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
        for filename in HISTORICAL_FILES:
            assert (raw_dir / filename).exists(), f"Missing {filename}"
