"""Tests for the feature engineering pipeline."""

import pandas as pd
import numpy as np
import pytest

from src.data.build_features import (
    _result_label,
    compute_recent_form,
    compute_rest_days,
    compute_head_to_head,
    build_feature_set,
)


@pytest.fixture
def sample_matches():
    return pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B",
         "home_score": 2, "away_score": 1, "neutral": False, "tournament": "Friendly"},
        {"date": pd.Timestamp("2020-01-15"), "home_team": "B", "away_team": "C",
         "home_score": 0, "away_score": 0, "neutral": False, "tournament": "Friendly"},
        {"date": pd.Timestamp("2020-02-01"), "home_team": "A", "away_team": "C",
         "home_score": 1, "away_score": 3, "neutral": True, "tournament": "World Cup"},
        {"date": pd.Timestamp("2020-02-15"), "home_team": "C", "away_team": "A",
         "home_score": 2, "away_score": 2, "neutral": False, "tournament": "Friendly"},
    ])


class TestResultLabel:
    def test_home_win(self):
        assert _result_label(2, 0) == 2

    def test_away_win(self):
        assert _result_label(0, 2) == 0

    def test_draw(self):
        assert _result_label(1, 1) == 1


class TestRecentForm:
    def test_first_match_uses_default_form(self, sample_matches):
        result = compute_recent_form(sample_matches, window=5)
        assert result.iloc[0]["home_form"] == 1.0
        assert result.iloc[0]["away_form"] == 1.0

    def test_form_updates_after_match(self, sample_matches):
        result = compute_recent_form(sample_matches, window=5)
        # team A won their first match (3 pts), should reflect in their next appearance
        a_third_match_form = result.iloc[2]["home_form"]
        assert a_third_match_form > 0

    def test_no_data_leakage_first_match_is_neutral(self, sample_matches):
        result = compute_recent_form(sample_matches, window=5)
        # no team should know the future on their very first match
        assert result.iloc[0]["home_form"] == result.iloc[0]["away_form"] == 1.0


class TestRestDays:
    def test_first_appearance_gets_default_rest(self, sample_matches):
        result = compute_rest_days(sample_matches)
        assert result.iloc[0]["home_rest_days"] == 60
        assert result.iloc[0]["away_rest_days"] == 60

    def test_rest_days_computed_correctly(self, sample_matches):
        result = compute_rest_days(sample_matches)
        # B plays Jan 1 and Jan 15 -> 14 days rest before second match
        b_second_match_rest = result.iloc[1]["home_rest_days"]
        assert b_second_match_rest == 14

    def test_rest_days_non_negative(self, sample_matches):
        result = compute_rest_days(sample_matches)
        assert (result["home_rest_days"] >= 0).all()
        assert (result["away_rest_days"] >= 0).all()


class TestHeadToHead:
    def test_first_meeting_defaults_to_fifty_percent(self, sample_matches):
        result = compute_head_to_head(sample_matches)
        assert result.iloc[0]["h2h_home_win_rate"] == 0.5

    def test_h2h_updates_after_meeting(self, sample_matches):
        result = compute_head_to_head(sample_matches)
        # A beat C 1-3 (away win) in match 3, the rematch in match 4 should reflect history
        rematch_h2h = result.iloc[3]["h2h_home_win_rate"]
        assert rematch_h2h != 0.5


class TestBuildFeatureSet:
    def test_produces_expected_columns(self, sample_matches):
        features, elo = build_feature_set(sample_matches)
        expected_cols = {
            "elo_diff", "home_elo_pre", "away_elo_pre", "form_diff",
            "home_form", "away_form", "rest_diff", "h2h_home_win_rate",
            "is_neutral", "is_knockout", "result",
        }
        assert expected_cols.issubset(set(features.columns))

    def test_no_missing_values_in_features(self, sample_matches):
        features, elo = build_feature_set(sample_matches)
        feature_cols = ["elo_diff", "home_form", "away_form", "h2h_home_win_rate"]
        assert features[feature_cols].isnull().sum().sum() == 0

    def test_result_label_matches_score(self, sample_matches):
        features, elo = build_feature_set(sample_matches)
        first_match = features.iloc[0]
        assert first_match["result"] == 2  # A won 2-1 at home

    def test_neutral_flag_preserved(self, sample_matches):
        features, elo = build_feature_set(sample_matches)
        neutral_match = features[
            (features["home_team"] == "A") & (features["away_team"] == "C")
        ]
        assert neutral_match.iloc[0]["is_neutral"] == 1

    def test_returns_fitted_elo_system(self, sample_matches):
        features, elo = build_feature_set(sample_matches)
        assert elo.get_rating("A") != 1500.0  # should have moved from base after matches
