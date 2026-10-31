"""Tests for the ELO rating system."""

import pandas as pd
import pytest

from src.data.elo_ratings import EloRatingSystem, _expected_score, _goal_diff_multiplier


class TestExpectedScore:
    def test_equal_ratings_gives_fifty_percent(self):
        assert _expected_score(1500, 1500) == pytest.approx(0.5, abs=1e-6)

    def test_higher_rating_favored(self):
        assert _expected_score(1700, 1500) > 0.5

    def test_lower_rating_disfavored(self):
        assert _expected_score(1300, 1500) < 0.5

    def test_symmetry(self):
        a_vs_b = _expected_score(1600, 1400)
        b_vs_a = _expected_score(1400, 1600)
        assert a_vs_b + b_vs_a == pytest.approx(1.0, abs=1e-6)


class TestGoalDiffMultiplier:
    def test_one_goal_diff_baseline(self):
        assert _goal_diff_multiplier(1) == 1.0

    def test_two_goal_diff_increased(self):
        assert _goal_diff_multiplier(2) == 1.5

    def test_large_diff_scales_up(self):
        assert _goal_diff_multiplier(5) > _goal_diff_multiplier(2)


class TestEloRatingSystem:
    def test_new_team_starts_at_base_rating(self):
        system = EloRatingSystem(base_rating=1500.0)
        assert system.get_rating("NewTeam") == 1500.0

    def test_winner_rating_increases(self):
        system = EloRatingSystem()
        before = system.get_rating("TeamA")
        system.process_match("TeamA", "TeamB", home_score=2, away_score=0)
        after = system.get_rating("TeamA")
        assert after > before

    def test_loser_rating_decreases(self):
        system = EloRatingSystem()
        before = system.get_rating("TeamB")
        system.process_match("TeamA", "TeamB", home_score=2, away_score=0)
        after = system.get_rating("TeamB")
        assert after < before

    def test_draw_with_equal_ratings_no_change(self):
        system = EloRatingSystem()
        system.process_match("TeamA", "TeamB", home_score=1, away_score=1, neutral_venue=True)
        # both started equal, draw on neutral venue should roughly cancel out
        assert abs(system.get_rating("TeamA") - 1500.0) < 5.0

    def test_home_advantage_applied(self):
        system_neutral = EloRatingSystem()
        system_home = EloRatingSystem()

        snap_neutral = system_neutral.process_match("A", "B", 1, 1, neutral_venue=True)
        snap_home = system_home.process_match("A", "B", 1, 1, neutral_venue=False)

        # home team draw should look like a slight "loss" relative to expectation
        # when home advantage was baked into the expected score
        assert snap_home["expected_home_result"] > snap_neutral["expected_home_result"]

    def test_history_records_each_match(self):
        system = EloRatingSystem()
        system.process_match("A", "B", 2, 1)
        system.process_match("C", "D", 0, 0)
        assert len(system.history) == 2

    def test_build_ratings_from_history_preserves_order(self):
        matches = pd.DataFrame([
            {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 1, "neutral": False},
            {"date": pd.Timestamp("2020-01-08"), "home_team": "B", "away_team": "C",
             "home_score": 0, "away_score": 0, "neutral": False},
        ])
        system = EloRatingSystem()
        result = system.build_ratings_from_history(matches)
        assert len(result) == 2
        assert result.iloc[0]["home_team"] == "A"

    def test_current_ratings_table_sorted_descending(self):
        system = EloRatingSystem()
        system.process_match("A", "B", 5, 0)  # A should now be well above B
        table = system.current_ratings_table()
        assert table.iloc[0]["elo_rating"] >= table.iloc[1]["elo_rating"]

    def test_rating_conservation(self):
        """Zero-sum property: what one team gains, the other loses exactly."""
        system = EloRatingSystem()
        rating_a_before = system.get_rating("A")
        rating_b_before = system.get_rating("B")

        system.process_match("A", "B", 2, 0)

        delta_a = system.get_rating("A") - rating_a_before
        delta_b = system.get_rating("B") - rating_b_before

        assert delta_a == pytest.approx(-delta_b, abs=1e-6)
