"""Tests for the real-data match summary / momentum timeline module."""

import pytest

from src.data.match_summary import (
    _parse_minute,
    _build_momentum_timeline,
    _extract_goals,
    _resolve_team_name,
    GoalEvent,
    fetch_match_summary,
    list_available_completed_matches,
)


class TestParseMinute:
    def test_regular_minute(self):
        assert _parse_minute("67") == 67.0

    def test_stoppage_time_minute(self):
        assert _parse_minute("90+7") == 90.7

    def test_first_half_stoppage(self):
        assert _parse_minute("45+5") == 45.5

    def test_malformed_minute_defaults_to_zero(self):
        assert _parse_minute("unknown") == 0.0


class TestResolveTeamName:
    def test_plain_spelling_resolves_to_special_character_version(self):
        assert _resolve_team_name("Curacao") == "Curaçao"

    def test_unmapped_name_passes_through(self):
        assert _resolve_team_name("Brazil") == "Brazil"


class TestExtractGoals:
    def test_extracts_basic_goal(self):
        raw = [{"name": "Erling Haaland", "minute": "29"}]
        events = _extract_goals(raw, "Norway")
        assert len(events) == 1
        assert events[0].scorer == "Erling Haaland"
        assert events[0].team == "Norway"
        assert not events[0].is_penalty

    def test_extracts_penalty_flag(self):
        raw = [{"name": "Harry Kane", "minute": "12", "penalty": True}]
        events = _extract_goals(raw, "England")
        assert events[0].is_penalty

    def test_extracts_own_goal_flag(self):
        raw = [{"name": "Some Player", "minute": "75", "owngoal": True}]
        events = _extract_goals(raw, "TeamA")
        assert events[0].is_own_goal

    def test_empty_goals_list(self):
        assert _extract_goals([], "TeamA") == []


class TestBuildMomentumTimeline:
    def test_starts_at_zero_zero(self):
        timeline = _build_momentum_timeline([], "Home")
        assert timeline[0]["home_score"] == 0
        assert timeline[0]["away_score"] == 0

    def test_running_score_updates_correctly(self):
        events = [
            GoalEvent("10", 10.0, "Home", "Player A"),
            GoalEvent("20", 20.0, "Away", "Player B"),
            GoalEvent("30", 30.0, "Home", "Player C"),
        ]
        timeline = _build_momentum_timeline(events, "Home")
        assert timeline[-1]["home_score"] == 2
        assert timeline[-1]["away_score"] == 1

    def test_timeline_sorted_by_minute_even_if_input_unsorted(self):
        events = [
            GoalEvent("80", 80.0, "Away", "Player B"),
            GoalEvent("10", 10.0, "Home", "Player A"),
        ]
        timeline = _build_momentum_timeline(events, "Home")
        minutes = [p["minute"] for p in timeline]
        assert minutes == sorted(minutes)

    def test_stoppage_time_goal_sorted_after_regular_time(self):
        events = [
            GoalEvent("90+3", 90.3, "Home", "Player A"),
            GoalEvent("85", 85.0, "Away", "Player B"),
        ]
        timeline = _build_momentum_timeline(events, "Home")
        assert timeline[1]["minute"] == 85.0
        assert timeline[2]["minute"] == 90.3


class TestFetchMatchSummary:
    def test_real_known_high_scoring_match(self):
        """Verified real fact: Germany beat Curacao 7-1 with HT 3-1."""
        summary = fetch_match_summary("Germany", "Curacao")
        assert summary is not None
        assert summary.home_score == 7
        assert summary.away_score == 1
        assert summary.half_time_score == (3, 1)

    def test_order_independent_lookup(self):
        """Should find the match regardless of which team is passed first."""
        a = fetch_match_summary("Germany", "Curacao")
        b = fetch_match_summary("Curacao", "Germany")
        assert a is not None and b is not None
        assert a.home_score == b.home_score or a.home_score == b.away_score

    def test_goal_events_sorted_chronologically(self):
        summary = fetch_match_summary("Germany", "Curacao")
        minutes = [e.minute_sort for e in summary.goal_events]
        assert minutes == sorted(minutes)

    def test_nonexistent_match_returns_none(self):
        summary = fetch_match_summary("NotARealTeam", "AlsoNotReal")
        assert summary is None

    def test_momentum_timeline_ends_at_final_score(self):
        summary = fetch_match_summary("Germany", "Curacao")
        last_point = summary.momentum_timeline[-1]
        assert last_point["home_score"] == 7
        assert last_point["away_score"] == 1


class TestListAvailableCompletedMatches:
    def test_returns_dataframe_with_matches(self):
        df = list_available_completed_matches()
        assert len(df) > 0

    def test_expected_columns_present(self):
        df = list_available_completed_matches()
        expected = {"date", "home_team", "away_team", "home_score", "away_score", "stage"}
        assert expected.issubset(set(df.columns))

    def test_all_listed_matches_have_real_scores(self):
        df = list_available_completed_matches()
        assert (df["home_score"] >= 0).all()
        assert (df["away_score"] >= 0).all()
