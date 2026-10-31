"""Tests for the data quality monitoring module."""

import pytest
import pandas as pd

from src.data.data_quality import (
    check_no_nulls_in_required_fields,
    check_no_negative_scores,
    check_no_team_plays_itself,
    check_no_duplicate_matches,
    check_expected_match_count_per_tournament,
    check_dates_within_real_tournament_windows,
    check_data_freshness,
    run_all_quality_checks,
)


@pytest.fixture
def clean_df():
    return pd.DataFrame([
        {"date": "1990-06-08", "home_team": "Italy", "away_team": "Austria",
         "home_score": 1, "away_score": 0, "year": 1990},
        {"date": "1990-06-09", "home_team": "Czechoslovakia", "away_team": "USA",
         "home_score": 5, "away_score": 1, "year": 1990},
    ])


class TestNoNullsCheck:
    def test_passes_on_clean_data(self, clean_df):
        result = check_no_nulls_in_required_fields(clean_df)
        assert result.passed

    def test_fails_when_nulls_present(self, clean_df):
        dirty = clean_df.copy()
        dirty.loc[0, "home_score"] = None
        result = check_no_nulls_in_required_fields(dirty)
        assert not result.passed
        assert result.affected_rows > 0


class TestNegativeScoresCheck:
    def test_passes_on_clean_data(self, clean_df):
        result = check_no_negative_scores(clean_df)
        assert result.passed

    def test_fails_on_negative_score(self, clean_df):
        dirty = clean_df.copy()
        dirty.loc[0, "home_score"] = -1
        result = check_no_negative_scores(dirty)
        assert not result.passed
        assert result.affected_rows == 1


class TestSelfMatchCheck:
    def test_passes_on_clean_data(self, clean_df):
        result = check_no_team_plays_itself(clean_df)
        assert result.passed

    def test_fails_when_team_plays_itself(self, clean_df):
        dirty = clean_df.copy()
        dirty.loc[0, "away_team"] = dirty.loc[0, "home_team"]
        result = check_no_team_plays_itself(dirty)
        assert not result.passed


class TestDuplicateCheck:
    def test_passes_on_clean_data(self, clean_df):
        result = check_no_duplicate_matches(clean_df)
        assert result.passed

    def test_fails_on_exact_duplicate(self, clean_df):
        dirty = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
        result = check_no_duplicate_matches(dirty)
        assert not result.passed
        assert result.affected_rows == 1

    def test_fails_on_swapped_home_away_duplicate(self, clean_df):
        swapped = clean_df.iloc[[0]].copy()
        swapped["home_team"], swapped["away_team"] = swapped["away_team"].values, swapped["home_team"].values
        dirty = pd.concat([clean_df, swapped], ignore_index=True)
        result = check_no_duplicate_matches(dirty)
        assert not result.passed

    def test_catches_real_name_variant_duplicate(self):
        """
        Regression test for a real bug found in this project's own data:
        'Portugal vs Congo DR' and 'Portugal vs DR Congo' were both present
        as if they were two different matches, when they're the same real
        country under two valid spellings.
        """
        df = pd.DataFrame([
            {"date": "2026-06-17", "home_team": "Portugal", "away_team": "Congo DR",
             "home_score": 1, "away_score": 1, "year": 2026},
            {"date": "2026-06-17", "home_team": "Portugal", "away_team": "DR Congo",
             "home_score": 1, "away_score": 1, "year": 2026},
        ])
        result = check_no_duplicate_matches(df)
        assert not result.passed
        assert result.affected_rows == 1


class TestExpectedMatchCountCheck:
    def test_fails_when_tournament_has_wrong_count(self):
        """1990 should have 52 matches; giving it only 2 should fail."""
        df = pd.DataFrame([
            {"date": "1990-06-08", "home_team": "A", "away_team": "B", "year": 1990},
            {"date": "1990-06-09", "home_team": "C", "away_team": "D", "year": 1990},
        ])
        result = check_expected_match_count_per_tournament(df)
        assert not result.passed

    def test_passes_with_real_full_dataset(self):
        from src.data.combine_real_history import load_full_real_history
        df = load_full_real_history(include_2026=False)
        df["date"] = df["date"].astype(str)
        result = check_expected_match_count_per_tournament(df)
        assert result.passed


class TestDateWindowCheck:
    def test_fails_on_date_outside_real_window(self):
        df = pd.DataFrame([
            {"date": "1990-12-25", "home_team": "A", "away_team": "B", "year": 1990},  # impossible: WC90 was June-July
        ])
        result = check_dates_within_real_tournament_windows(df)
        assert not result.passed

    def test_passes_with_real_full_dataset(self):
        from src.data.combine_real_history import load_full_real_history
        df = load_full_real_history(include_2026=False)
        df["date"] = df["date"].astype(str)
        result = check_dates_within_real_tournament_windows(df)
        assert result.passed


class TestDataFreshnessCheck:
    def test_fails_when_no_2026_data(self):
        df = pd.DataFrame([{"date": "1990-06-08", "home_team": "A", "away_team": "B", "year": 1990}])
        result = check_data_freshness(df)
        assert not result.passed

    def test_passes_with_recent_real_2026_data(self):
        from src.data.combine_real_history import load_full_real_history
        df = load_full_real_history(include_2026=True)
        df["date"] = df["date"].astype(str)
        result = check_data_freshness(df, max_staleness_days=30)
        assert result.passed  # generous window since the tournament is actively running


class TestRunAllQualityChecks:
    def test_returns_dataframe_with_all_checks(self):
        report = run_all_quality_checks()
        assert len(report) == 7

    def test_expected_columns_present(self):
        report = run_all_quality_checks()
        expected = {"check_name", "category", "passed", "detail", "affected_rows"}
        assert expected.issubset(set(report.columns))

    def test_real_dataset_passes_all_structural_checks(self):
        """The real, current dataset should pass every check we ship with."""
        report = run_all_quality_checks()
        failing = report[~report["passed"]]
        assert len(failing) == 0, f"Unexpected quality failures: {failing[['check_name','detail']].to_dict('records')}"
