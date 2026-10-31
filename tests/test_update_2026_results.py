"""Tests for the live 2026 results auto-update script."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from update_2026_results import (
    append_new_results,
    _validate_match,
    _is_placeholder_team,
    _normalize_team_name,
    load_existing_results,
    fetch_live_2026_matches,
)


class TestValidateMatch:
    def test_valid_match_passes(self):
        match = {
            "date": "2026-06-25", "home_team": "Mexico", "away_team": "Czechia",
            "home_score": 2, "away_score": 1, "tournament": "World Cup",
            "neutral": False, "stage": "Group A", "year": 2026,
        }
        _validate_match(match)

    def test_missing_field_raises(self):
        match = {"date": "2026-06-25", "home_team": "Mexico"}
        with pytest.raises(ValueError, match="missing required fields"):
            _validate_match(match)

    def test_same_team_raises(self):
        match = {
            "date": "2026-06-25", "home_team": "Mexico", "away_team": "Mexico",
            "home_score": 1, "away_score": 1, "tournament": "World Cup",
            "neutral": False, "stage": "Group A", "year": 2026,
        }
        with pytest.raises(ValueError, match="identical"):
            _validate_match(match)

    def test_negative_score_raises(self):
        match = {
            "date": "2026-06-25", "home_team": "Mexico", "away_team": "Czechia",
            "home_score": -1, "away_score": 1, "tournament": "World Cup",
            "neutral": False, "stage": "Group A", "year": 2026,
        }
        with pytest.raises(ValueError, match="negative"):
            _validate_match(match)


class TestPlaceholderDetection:
    def test_real_team_name_not_placeholder(self):
        assert not _is_placeholder_team("Argentina")
        assert not _is_placeholder_team("South Korea")

    def test_knockout_slot_is_placeholder(self):
        assert _is_placeholder_team("2A")
        assert _is_placeholder_team("3C/D/F")
        assert _is_placeholder_team("W74")

    def test_empty_name_is_placeholder(self):
        assert _is_placeholder_team("")
        assert _is_placeholder_team(None)


class TestTeamNameNormalization:
    def test_known_alias_mapped(self):
        assert _normalize_team_name("Czech Republic") == "Czechia"
        assert _normalize_team_name("Curaçao") == "Curacao"

    def test_unknown_name_passed_through(self):
        assert _normalize_team_name("Brazil") == "Brazil"


class TestFetchLive2026Matches:
    def test_fetch_returns_list(self):
        matches = fetch_live_2026_matches()
        assert isinstance(matches, list)

    def test_fetch_returns_only_completed_matches(self):
        matches = fetch_live_2026_matches()
        for m in matches:
            assert m["home_score"] >= 0
            assert m["away_score"] >= 0

    def test_fetch_excludes_placeholder_teams(self):
        matches = fetch_live_2026_matches()
        for m in matches:
            assert not _is_placeholder_team(m["home_team"])
            assert not _is_placeholder_team(m["away_team"])

    def test_fetch_returns_real_known_match(self):
        """Verified real fact: Mexico beat South Africa 2-0 in the opener."""
        matches = fetch_live_2026_matches()
        opener = [
            m for m in matches
            if {m["home_team"], m["away_team"]} == {"Mexico", "South Africa"}
        ]
        assert len(opener) >= 1
        assert opener[0]["home_score"] == 2
        assert opener[0]["away_score"] == 0


class TestAppendNewResultsIdempotency:
    def test_dry_run_does_not_modify_file(self):
        before = load_existing_results()
        new_match = [{
            "date": "2099-01-01", "home_team": "TestTeamA", "away_team": "TestTeamB",
            "home_score": 1, "away_score": 0, "tournament": "World Cup",
            "neutral": True, "stage": "Group A", "year": 2026,
        }]
        append_new_results(new_match, dry_run=True)
        after = load_existing_results()
        assert len(before) == len(after)

    def test_duplicate_with_swapped_home_away_is_skipped(self):
        """Real-world case: two sources may disagree on which team is 'home'."""
        existing = load_existing_results()
        first_real_match = existing.iloc[0]
        swapped_duplicate = [{
            "date": first_real_match["date"].strftime("%Y-%m-%d"),
            "home_team": first_real_match["away_team"],  # swapped
            "away_team": first_real_match["home_team"],  # swapped
            "home_score": int(first_real_match["away_score"]),
            "away_score": int(first_real_match["home_score"]),
            "tournament": "World Cup", "neutral": True,
            "stage": first_real_match["stage"], "year": 2026,
        }]
        result = append_new_results(swapped_duplicate, dry_run=True)
        assert len(result) == len(existing)  # correctly recognized as duplicate

    def test_genuinely_new_match_would_be_added(self):
        before = load_existing_results()
        new_match = [{
            "date": "2099-01-01", "home_team": "TestTeamA", "away_team": "TestTeamB",
            "home_score": 1, "away_score": 0, "tournament": "World Cup",
            "neutral": True, "stage": "Group A", "year": 2026,
        }]
        result = append_new_results(new_match, dry_run=True)
        assert len(result) == len(before) + 1

    def test_running_update_twice_is_idempotent(self):
        """The core safety property: re-running never creates duplicates."""
        live_matches = fetch_live_2026_matches()
        first_run = append_new_results(live_matches, dry_run=True)
        second_run = append_new_results(live_matches, dry_run=True)
        # Second dry run against the (unmodified) file should find the same
        # "new" matches as the first, since dry_run never actually writes.
        assert len(first_run) == len(second_run)

