"""
Tracks model prediction accuracy against actual World Cup 2026 results as
they happen. This is the "honesty layer" of the project: rather than only
reporting historical backtest accuracy, this module scores the model
against ground truth in real time as the tournament progresses.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_LOG_PATH = PROCESSED_DIR / "live_results_log.csv"


def log_actual_result(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    match_date: str,
) -> None:
    """
    Appends a completed match result to the live tracking log. Call this
    after each real World Cup match is played to update the running
    accuracy dashboard.
    """
    predictions_path = PROCESSED_DIR / "fixture_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            "No predictions file found. Run predict.py before logging results."
        )

    predictions = pd.read_csv(predictions_path)
    match_row = predictions[
        (predictions["home_team"] == home_team) & (predictions["away_team"] == away_team)
    ]

    if match_row.empty:
        raise ValueError(f"No prediction found for {home_team} vs {away_team}")

    row = match_row.iloc[0]

    if home_score > away_score:
        actual_outcome = 2
    elif home_score == away_score:
        actual_outcome = 1
    else:
        actual_outcome = 0

    predicted_outcome = int(row["predicted_outcome"])
    correct = int(predicted_outcome == actual_outcome)

    log_entry = pd.DataFrame([{
        "logged_at": datetime.now().isoformat(),
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "predicted_outcome": predicted_outcome,
        "actual_outcome": actual_outcome,
        "correct": correct,
        "prob_home_win": row["prob_home_win"],
        "prob_draw": row["prob_draw"],
        "prob_away_win": row["prob_away_win"],
    }])

    if RESULTS_LOG_PATH.exists():
        existing = pd.read_csv(RESULTS_LOG_PATH)
        combined = pd.concat([existing, log_entry], ignore_index=True)
    else:
        combined = log_entry

    combined.to_csv(RESULTS_LOG_PATH, index=False)


def get_live_accuracy() -> dict:
    """
    Returns running accuracy statistics for the tournament so far.
    """
    if not RESULTS_LOG_PATH.exists():
        return {"matches_scored": 0, "accuracy": None, "log_loss_avg": None}

    log = pd.read_csv(RESULTS_LOG_PATH)

    if log.empty:
        return {"matches_scored": 0, "accuracy": None, "log_loss_avg": None}

    accuracy = log["correct"].mean()

    import numpy as np
    actual_probs = log.apply(
        lambda r: r[["prob_away_win", "prob_draw", "prob_home_win"]].iloc[r["actual_outcome"]],
        axis=1,
    )
    log_loss_avg = -np.mean(np.log(actual_probs.clip(lower=1e-9)))

    return {
        "matches_scored": len(log),
        "accuracy": round(accuracy, 3),
        "log_loss_avg": round(log_loss_avg, 3),
    }


if __name__ == "__main__":
    stats = get_live_accuracy()
    print(stats)
