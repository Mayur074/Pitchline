"""Tests for the model-vs-real-market validation module."""

import pytest

from src.models.validate_vs_market import elo_diff_to_implied_prob, compare_model_to_market


class TestEloDiffToImpliedProb:
    def test_zero_diff_gives_fifty_percent(self):
        assert elo_diff_to_implied_prob(0) == pytest.approx(0.5, abs=1e-6)

    def test_positive_diff_above_fifty(self):
        assert elo_diff_to_implied_prob(200) > 0.5

    def test_negative_diff_below_fifty(self):
        assert elo_diff_to_implied_prob(-200) < 0.5

    def test_large_positive_diff_approaches_one(self):
        assert elo_diff_to_implied_prob(1000) > 0.95

    def test_output_always_valid_probability(self):
        for diff in [-2000, -500, 0, 500, 2000]:
            prob = elo_diff_to_implied_prob(diff)
            assert 0.0 <= prob <= 1.0


class TestCompareModelToMarket:
    def test_returns_dataframe_with_results(self):
        result = compare_model_to_market()
        assert len(result) > 0

    def test_expected_columns_present(self):
        result = compare_model_to_market()
        expected = {
            "home_team", "away_team", "elo_implied_home_win_pct",
            "market_home_win_pct", "abs_difference", "same_favorite",
        }
        assert expected.issubset(set(result.columns))

    def test_agreement_rate_is_reasonable(self):
        """
        Sanity check: a model built from real FIFA rankings and real match
        results should agree with the real betting market on the favored
        team most of the time, even without access to market data.
        """
        result = compare_model_to_market()
        agreement_rate = result["same_favorite"].mean()
        assert agreement_rate >= 0.5  # should beat a coin flip at minimum

    def test_implied_probabilities_in_valid_range(self):
        result = compare_model_to_market()
        assert (result["elo_implied_home_win_pct"] >= 0).all()
        assert (result["elo_implied_home_win_pct"] <= 100).all()
