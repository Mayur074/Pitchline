"""Tests for the upset/anomaly detector."""

import pytest
import pandas as pd

from src.models.upset_detector import (
    _elo_diff_to_win_prob,
    detect_upsets,
    summarize_upsets,
    UPSET_PROBABILITY_THRESHOLD,
    STATEMENT_WIN_GOAL_DIFF,
)


class TestEloDiffToWinProb:
    def test_zero_diff_is_fifty_percent(self):
        assert _elo_diff_to_win_prob(0) == pytest.approx(0.5, abs=1e-6)

    def test_large_positive_diff_near_certain(self):
        assert _elo_diff_to_win_prob(800) > 0.95

    def test_large_negative_diff_near_zero(self):
        assert _elo_diff_to_win_prob(-800) < 0.05

    def test_symmetric(self):
        assert _elo_diff_to_win_prob(300) + _elo_diff_to_win_prob(-300) == pytest.approx(1.0, abs=1e-6)


class TestDetectUpsets:
    def test_returns_dataframe_for_2026(self):
        result = detect_upsets(real_results_year=2026)
        assert len(result) > 0

    def test_expected_columns_present(self):
        result = detect_upsets(real_results_year=2026)
        expected = {
            "date", "home_team", "away_team", "home_score", "away_score",
            "home_win_prob_pre_match", "away_win_prob_pre_match",
            "winner", "winner_pre_match_prob", "goal_diff",
            "is_upset", "upset_type", "favorite_held_to_draw", "is_statement_win",
        }
        assert expected.issubset(set(result.columns))

    def test_win_probabilities_sum_to_one(self):
        result = detect_upsets(real_results_year=2026)
        sums = result["home_win_prob_pre_match"] + result["away_win_prob_pre_match"]
        assert all(abs(s - 1.0) < 1e-6 for s in sums)

    def test_draws_have_no_winner(self):
        result = detect_upsets(real_results_year=2026)
        draws = result[result["home_score"] == result["away_score"]]
        assert draws["winner"].isna().all()

    def test_statement_win_flagged_correctly(self):
        """Verified real fact: Germany beat Curacao 7-1, a 6-goal margin."""
        result = detect_upsets(real_results_year=2026)
        germany_match = result[
            (result["home_team"] == "Germany") & (result["away_team"] == "Curacao")
        ]
        assert len(germany_match) == 1
        assert germany_match.iloc[0]["goal_diff"] == 6
        assert germany_match.iloc[0]["is_statement_win"] == True

    def test_statement_win_threshold_respected(self):
        result = detect_upsets(real_results_year=2026)
        statements = result[result["is_statement_win"]]
        assert (statements["goal_diff"] >= STATEMENT_WIN_GOAL_DIFF).all()

    def test_upset_threshold_respected(self):
        result = detect_upsets(real_results_year=2026)
        underdog_wins = result[result["upset_type"] == "underdog_win"]
        if len(underdog_wins):
            assert (underdog_wins["winner_pre_match_prob"] < UPSET_PROBABILITY_THRESHOLD).all()

    def test_filters_to_requested_year_only(self):
        result = detect_upsets(real_results_year=2026)
        assert (result["year"] == 2026).all()

    def test_real_known_draw_upset_is_caught(self):
        """
        Regression test for a real blind spot found in this detector: Spain
        (a top-5 ranked team) was held to a 0-0 draw by Cape Verde (a
        tournament debutant) — a genuine shock result that the original
        version of this detector missed entirely because it only checked
        for clear win/loss upsets and silently excluded all draws.
        """
        result = detect_upsets(real_results_year=2026)
        match = result[
            (result["home_team"] == "Spain") & (result["away_team"] == "Cape Verde")
        ]
        assert len(match) == 1
        row = match.iloc[0]
        assert row["is_upset"] == True
        assert row["upset_type"] == "favorite_held_to_draw"
        assert row["favorite_held_to_draw"] == "Spain"

    def test_draw_upset_only_flagged_when_favorite_clearly_expected_to_win(self):
        """An evenly-matched draw should NOT be flagged as an upset."""
        result = detect_upsets(real_results_year=2026)
        draws = result[result["home_score"] == result["away_score"]]
        evenly_matched_draws = draws[
            draws[["home_win_prob_pre_match", "away_win_prob_pre_match"]].max(axis=1) < 0.6
        ]
        assert not evenly_matched_draws["is_upset"].any()


class TestSummarizeUpsets:
    def test_returns_expected_keys(self):
        df = detect_upsets(real_results_year=2026)
        summary = summarize_upsets(df)
        expected_keys = {
            "total_matches", "upsets_detected", "upset_rate", "underdog_wins",
            "favorites_held_to_draw", "statement_wins", "biggest_upset", "biggest_draw_upset",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_total_matches_matches_input_length(self):
        df = detect_upsets(real_results_year=2026)
        summary = summarize_upsets(df)
        assert summary["total_matches"] == len(df)

    def test_handles_empty_dataframe(self):
        empty = pd.DataFrame(columns=["is_upset", "is_statement_win", "winner_pre_match_prob", "upset_type"])
        summary = summarize_upsets(empty)
        assert summary["total_matches"] == 0
        assert summary["upset_rate"] == 0.0

    def test_draw_upsets_counted_in_summary(self):
        """Real fact: Spain vs Cape Verde should contribute to favorites_held_to_draw."""
        df = detect_upsets(real_results_year=2026)
        summary = summarize_upsets(df)
        assert summary["favorites_held_to_draw"] >= 1
        assert summary["biggest_draw_upset"] is not None
