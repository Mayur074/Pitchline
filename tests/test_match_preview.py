"""Tests for match preview generation (fallback logic, no API key required)."""

import pytest

from src.api.match_preview import _fallback_preview


class TestFallbackPreview:
    def test_clear_favorite_identified(self):
        preview = _fallback_preview("TeamA", "TeamB", prob_home_win=0.75, prob_draw=0.15, prob_away_win=0.10)
        assert "TeamA" in preview
        assert "75%" in preview

    def test_close_match_flagged_as_open(self):
        preview = _fallback_preview("TeamA", "TeamB", prob_home_win=0.35, prob_draw=0.33, prob_away_win=0.32)
        assert "open" in preview.lower() or "coin flip" in preview.lower()

    def test_draw_favored_handled(self):
        preview = _fallback_preview("TeamA", "TeamB", prob_home_win=0.30, prob_draw=0.40, prob_away_win=0.30)
        assert "draw" in preview.lower()

    def test_away_team_favorite_identified(self):
        preview = _fallback_preview("TeamA", "TeamB", prob_home_win=0.15, prob_draw=0.15, prob_away_win=0.70)
        assert "TeamB" in preview

    def test_no_exception_on_equal_probabilities(self):
        # should not raise even in a perfectly tied edge case
        preview = _fallback_preview("TeamA", "TeamB", prob_home_win=0.333, prob_draw=0.334, prob_away_win=0.333)
        assert isinstance(preview, str)
        assert len(preview) > 0

    def test_preview_always_returns_string(self):
        preview = _fallback_preview("X", "Y", 0.5, 0.3, 0.2)
        assert isinstance(preview, str)
        assert len(preview) > 0
