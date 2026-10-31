"""Tests for real World Cup 2026 data loading and ELO seeding from real FIFA rankings."""

import pytest
import pandas as pd

from src.data.real_data import (
    rank_to_seed_elo,
    build_seed_elo_ratings,
    load_real_fifa_rankings,
    load_real_world_cup_results,
    load_real_upcoming_fixtures,
    RANK_TO_ELO_TOP,
    RANK_TO_ELO_BOTTOM,
)


class TestRankToSeedElo:
    def test_rank_one_gets_top_rating(self):
        assert rank_to_seed_elo(1) == pytest.approx(RANK_TO_ELO_TOP, abs=1.0)

    def test_rank_fifty_gets_bottom_rating(self):
        assert rank_to_seed_elo(50) == pytest.approx(RANK_TO_ELO_BOTTOM, abs=1.0)

    def test_rank_ordering_preserved(self):
        """Better (lower) rank number should always map to a higher rating."""
        assert rank_to_seed_elo(1) > rank_to_seed_elo(10)
        assert rank_to_seed_elo(10) > rank_to_seed_elo(30)

    def test_rank_beyond_max_clamped(self):
        """Ranks worse than our max (e.g. unranked minnows) shouldn't go below the floor."""
        assert rank_to_seed_elo(200) == rank_to_seed_elo(50)


class TestLoadRealFifaRankings:
    def test_loads_without_error(self):
        rankings = load_real_fifa_rankings()
        assert len(rankings) > 0

    def test_argentina_ranked_first(self):
        """Verified real fact: Argentina entered the 2026 World Cup ranked #1."""
        rankings = load_real_fifa_rankings()
        top_team = rankings.sort_values("fifa_rank").iloc[0]
        assert top_team["team"] == "Argentina"
        assert top_team["fifa_rank"] == 1

    def test_expected_top_five_present(self):
        """Verified real fact: Spain, France, England, Portugal round out the real top 5."""
        rankings = load_real_fifa_rankings()
        top_five = set(rankings.sort_values("fifa_rank").head(6)["team"])
        for team in ["Argentina", "Spain", "France", "England", "Portugal", "Brazil"]:
            assert team in top_five


class TestBuildSeedEloRatings:
    def test_returns_dict(self):
        ratings = build_seed_elo_ratings()
        assert isinstance(ratings, dict)

    def test_argentina_has_highest_seed_rating(self):
        ratings = build_seed_elo_ratings()
        assert ratings["Argentina"] == max(ratings.values())

    def test_all_teams_have_positive_ratings(self):
        ratings = build_seed_elo_ratings()
        assert all(r > 0 for r in ratings.values())


class TestLoadRealWorldCupResults:
    def test_loads_without_error(self):
        results = load_real_world_cup_results()
        assert len(results) > 0

    def test_expected_columns_present(self):
        results = load_real_world_cup_results()
        expected = {"date", "home_team", "away_team", "home_score", "away_score"}
        assert expected.issubset(set(results.columns))

    def test_sorted_chronologically(self):
        results = load_real_world_cup_results()
        assert results["date"].is_monotonic_increasing

    def test_real_opening_match_present(self):
        """Verified real fact: the tournament opener was Mexico 2-0 South Africa."""
        results = load_real_world_cup_results()
        opener = results.iloc[0]
        assert opener["home_team"] == "Mexico"
        assert opener["away_team"] == "South Africa"
        assert opener["home_score"] == 2
        assert opener["away_score"] == 0

    def test_scores_are_non_negative(self):
        results = load_real_world_cup_results()
        assert (results["home_score"] >= 0).all()
        assert (results["away_score"] >= 0).all()


class TestLoadRealUpcomingFixtures:
    def test_loads_without_error(self):
        fixtures = load_real_upcoming_fixtures()
        assert len(fixtures) > 0

    def test_expected_columns_present(self):
        fixtures = load_real_upcoming_fixtures()
        expected = {"date", "home_team", "away_team", "stage"}
        assert expected.issubset(set(fixtures.columns))

    def test_no_team_plays_itself(self):
        fixtures = load_real_upcoming_fixtures()
        assert (fixtures["home_team"] != fixtures["away_team"]).all()
