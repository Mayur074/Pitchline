"""
Builds the model-ready feature set using REAL World Cup data spanning
1990-2026.

This module now has genuine match volume to work with: 552 real World Cup
matches from 1990-2022, plus every real match played at the 2026 tournament
so far. Team ELO ratings are seeded from the real June 11, 2026 FIFA World
Ranking (for any team without enough historical World Cup history to have
built up a rating organically), then updated chronologically through all
590 real matches in date order.

The resulting fixture predictions in src/models/predict.py are generated
from this real, continuously-updating rating system — not from synthetic
historical data. The synthetic generator in scripts/download_sample_data.py
still exists separately for anyone who wants to stress-test the modeling
pipeline with a larger volume of data than the real World Cup alone provides.
"""

import pandas as pd
from pathlib import Path

from src.data.elo_ratings import EloRatingSystem
from src.data.real_data import (
    build_seed_elo_ratings,
    load_real_upcoming_fixtures,
)
from src.data.combine_real_history import load_full_real_history

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_real_world_cup_elo_system() -> EloRatingSystem:
    """
    Returns an ELO system seeded from real FIFA rankings (as a sensible
    starting point for teams with sparse World Cup history) and updated
    through all 590 real World Cup matches from 1990 through the 2026
    tournament so far, processed in chronological order.
    """
    seed_ratings = build_seed_elo_ratings()
    full_history = load_full_real_history()

    elo = EloRatingSystem(seed_ratings=seed_ratings)
    elo.build_ratings_from_history(full_history)
    return elo


def build_real_fixture_predictions_input() -> pd.DataFrame:
    """
    Builds the feature rows needed to predict the real upcoming fixtures,
    using current (real, tournament-updated) ELO ratings for every team.
    """
    elo = build_real_world_cup_elo_system()
    fixtures = load_real_upcoming_fixtures()

    rows = []
    for _, fixture in fixtures.iterrows():
        home, away = fixture["home_team"], fixture["away_team"]
        home_elo = elo.get_rating(home)
        away_elo = elo.get_rating(away)

        rows.append({
            "date": fixture["date"],
            "home_team": home,
            "away_team": away,
            "stage": fixture["stage"],
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "elo_diff": round(home_elo - away_elo, 1),
            "market_home_win_pct": fixture.get("market_home_win_pct"),
            "market_draw_pct": fixture.get("market_draw_pct"),
            "market_away_win_pct": fixture.get("market_away_win_pct"),
        })

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "real_fixture_elo_features.csv"
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = build_real_fixture_predictions_input()
    print(f"Built ELO-based features for {len(df)} real upcoming fixtures\n")
    print(df[["home_team", "away_team", "home_elo", "away_elo", "elo_diff",
              "market_home_win_pct", "market_away_win_pct"]].to_string(index=False))
