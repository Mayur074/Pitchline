"""
Feature engineering for the match outcome prediction model.

Builds the feature set used to predict win/draw/loss outcomes:
- ELO rating differential (home vs away, pre-match)
- Recent form (points won in last 5 matches, per team)
- Rest days since each team's previous match
- Head-to-head historical win rate
- Tournament stage indicator (group stage vs knockout)
- Neutral venue flag
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data.elo_ratings import EloRatingSystem

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _result_label(home_score: int, away_score: int) -> int:
    """0 = away win, 1 = draw, 2 = home win. Used as the multiclass target."""
    if home_score > away_score:
        return 2
    elif home_score == away_score:
        return 1
    return 0


def compute_recent_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    For every match, computes each team's points won (3/1/0) across their
    previous `window` matches, using only information available before the
    current match date (no data leakage).
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    team_points_history: dict[str, list[tuple[pd.Timestamp, int]]] = {}

    home_form, away_form = [], []

    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]

        home_hist = team_points_history.get(home, [])
        away_hist = team_points_history.get(away, [])

        home_recent = [pts for _, pts in home_hist[-window:]]
        away_recent = [pts for _, pts in away_hist[-window:]]

        home_form.append(sum(home_recent) / len(home_recent) if home_recent else 1.0)
        away_form.append(sum(away_recent) / len(away_recent) if away_recent else 1.0)

        if row["home_score"] > row["away_score"]:
            home_pts, away_pts = 3, 0
        elif row["home_score"] < row["away_score"]:
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1

        team_points_history.setdefault(home, []).append((row["date"], home_pts))
        team_points_history.setdefault(away, []).append((row["date"], away_pts))

    matches["home_form"] = home_form
    matches["away_form"] = away_form
    return matches


def compute_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous match, capped at 60 for first appearances."""
    matches = matches.sort_values("date").reset_index(drop=True)
    last_played: dict[str, pd.Timestamp] = {}

    home_rest, away_rest = [], []

    for _, row in matches.iterrows():
        home, away, date = row["home_team"], row["away_team"], row["date"]

        home_rest.append((date - last_played[home]).days if home in last_played else 60)
        away_rest.append((date - last_played[away]).days if away in last_played else 60)

        last_played[home] = date
        last_played[away] = date

    matches["home_rest_days"] = np.clip(home_rest, 0, 60)
    matches["away_rest_days"] = np.clip(away_rest, 0, 60)
    return matches


def compute_head_to_head(matches: pd.DataFrame) -> pd.DataFrame:
    """Historical home-team win rate in this exact fixture pairing, prior to current match."""
    matches = matches.sort_values("date").reset_index(drop=True)
    h2h_record: dict[tuple[str, str], list[int]] = {}

    h2h_home_win_rate = []

    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        key = tuple(sorted([home, away]))
        history = h2h_record.get(key, [])

        if history:
            h2h_home_win_rate.append(sum(history) / len(history))
        else:
            h2h_home_win_rate.append(0.5)

        outcome = 1 if row["home_score"] > row["away_score"] else 0
        h2h_record.setdefault(key, []).append(outcome)

    matches["h2h_home_win_rate"] = h2h_home_win_rate
    return matches


def build_feature_set(
    matches: pd.DataFrame,
    elo_system: EloRatingSystem | None = None,
) -> tuple[pd.DataFrame, EloRatingSystem]:
    """
    Full feature engineering pipeline. Takes raw historical matches and
    returns a model-ready feature dataframe plus the fitted ELO system
    (so it can be reused to score upcoming fixtures with current ratings).
    """
    matches = matches.copy()
    matches["neutral"] = matches.get("neutral", False)

    if elo_system is None:
        elo_system = EloRatingSystem()

    elo_history = elo_system.build_ratings_from_history(matches)
    matches = matches.reset_index(drop=True)
    matches["home_elo_pre"] = elo_history["home_rating_pre"]
    matches["away_elo_pre"] = elo_history["away_rating_pre"]
    matches["elo_diff"] = matches["home_elo_pre"] - matches["away_elo_pre"]

    matches = compute_recent_form(matches)
    matches = compute_rest_days(matches)
    matches = compute_head_to_head(matches)

    matches["form_diff"] = matches["home_form"] - matches["away_form"]
    matches["rest_diff"] = matches["home_rest_days"] - matches["away_rest_days"]
    matches["is_neutral"] = matches["neutral"].astype(int)
    matches["is_knockout"] = matches.get(
        "tournament", pd.Series(["" for _ in range(len(matches))])
    ).str.contains("Knockout|Final|Semi|Quarter", case=False, na=False).astype(int)

    matches["result"] = matches.apply(
        lambda r: _result_label(r["home_score"], r["away_score"]), axis=1
    )

    feature_cols = [
        "elo_diff", "home_elo_pre", "away_elo_pre",
        "form_diff", "home_form", "away_form",
        "rest_diff", "h2h_home_win_rate", "is_neutral", "is_knockout",
    ]

    output_cols = ["date", "home_team", "away_team", "home_score", "away_score"] + feature_cols + ["result"]
    feature_df = matches[output_cols].dropna(subset=feature_cols)

    out_path = PROCESSED_DATA_DIR / "match_features.csv"
    feature_df.to_csv(out_path, index=False)

    return feature_df, elo_system


if __name__ == "__main__":
    from src.data.fetch_data import fetch_historical_matches

    raw = fetch_historical_matches()
    features, elo = build_feature_set(raw)
    print(f"Built {len(features)} feature rows with {features.shape[1]} columns")
    print(features.tail())
