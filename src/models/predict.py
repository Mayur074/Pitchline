"""
Generates win/draw/loss probability predictions for upcoming World Cup 2026
fixtures using the trained model and current ELO ratings, then optionally
enriches each prediction with an AI-generated match preview via the Claude API.
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

from src.data.elo_ratings import EloRatingSystem
try:
    from src.api.match_preview import generate_match_preview
except ImportError:
    def generate_match_preview(**kwargs):
        return "[AI preview unavailable: anthropic package not installed]"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURE_COLS = [
    "elo_diff", "home_elo_pre", "away_elo_pre",
    "form_diff", "home_form", "away_form",
    "rest_diff", "h2h_home_win_rate", "is_neutral", "is_knockout",
]


def load_model():
    model = joblib.load(MODEL_DIR / "match_predictor.joblib")
    scaler = joblib.load(MODEL_DIR / "feature_scaler.joblib")
    return model, scaler


def build_fixture_features(
    fixtures: pd.DataFrame,
    elo_system: EloRatingSystem,
    team_form: dict[str, float],
    h2h_lookup: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """
    Builds the same feature set used in training, but for upcoming (not yet
    played) fixtures, using the latest known ELO ratings and form figures.
    """
    rows = []
    for _, fixture in fixtures.iterrows():
        home, away = fixture["home_team"], fixture["away_team"]

        home_elo = elo_system.get_rating(home)
        away_elo = elo_system.get_rating(away)

        home_form = team_form.get(home, 1.0)
        away_form = team_form.get(away, 1.0)

        key = tuple(sorted([home, away]))
        h2h_rate = h2h_lookup.get(key, 0.5)

        is_knockout = 1 if fixture.get("stage", "").lower() not in ("group stage", "group") else 0

        rows.append({
            "home_team": home,
            "away_team": away,
            "date": fixture["date"],
            "elo_diff": home_elo - away_elo,
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
            "form_diff": home_form - away_form,
            "home_form": home_form,
            "away_form": away_form,
            "rest_diff": 0,  # neutral assumption for tournament play (teams rest similarly)
            "h2h_home_win_rate": h2h_rate,
            "is_neutral": 1,  # World Cup matches are played at neutral/host venues
            "is_knockout": is_knockout,
        })

    return pd.DataFrame(rows)


def predict_fixtures(fixtures_path: str, generate_previews: bool = True) -> pd.DataFrame:
    """
    Main prediction pipeline: loads the model, builds features for upcoming
    fixtures, generates win/draw/loss probabilities, and optionally calls
    the Claude API to produce a natural-language preview per match.
    """
    model, scaler = load_model()

    fixtures = pd.read_csv(fixtures_path, parse_dates=["date"])

    features_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "match_features.csv"
    historical = pd.read_csv(features_path, parse_dates=["date"])

    elo_system = EloRatingSystem()
    raw_matches_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    elo_system.build_ratings_from_history(historical[raw_matches_cols])

    team_form = {}
    for team in pd.concat([historical["home_team"], historical["away_team"]]).unique():
        team_form[team] = 1.0  # placeholder: replaced with rolling form in full pipeline

    h2h_lookup = {}

    feature_df = build_fixture_features(fixtures, elo_system, team_form, h2h_lookup)

    X = feature_df[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)

    feature_df["prob_away_win"] = probs[:, 0]
    feature_df["prob_draw"] = probs[:, 1]
    feature_df["prob_home_win"] = probs[:, 2]
    feature_df["predicted_outcome"] = np.argmax(probs, axis=1)

    if generate_previews:
        previews = []
        for _, row in feature_df.iterrows():
            preview = generate_match_preview(
                home_team=row["home_team"],
                away_team=row["away_team"],
                prob_home_win=row["prob_home_win"],
                prob_draw=row["prob_draw"],
                prob_away_win=row["prob_away_win"],
                elo_diff=row["elo_diff"],
            )
            previews.append(preview)
        feature_df["ai_preview"] = previews

    output_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "fixture_predictions.csv"
    feature_df.to_csv(output_path, index=False)
    logger.info("Saved %d fixture predictions to %s", len(feature_df), output_path)

    return feature_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=str, required=True, help="Path to upcoming fixtures CSV")
    parser.add_argument("--no-preview", action="store_true", help="Skip AI preview generation")
    args = parser.parse_args()

    results = predict_fixtures(args.fixtures, generate_previews=not args.no_preview)
    print(results[["home_team", "away_team", "prob_home_win", "prob_draw", "prob_away_win"]])
