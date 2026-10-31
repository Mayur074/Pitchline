"""
Upset / anomaly detector for completed World Cup matches.

Mirrors the same statistical pattern used in production anomaly detection
systems (fraud detection, sensor monitoring, predictive maintenance): define
an expected baseline from the model, measure how far the real outcome
deviated from it, and flag deviations that cross a significance threshold.

Here the "expected baseline" is the ELO-implied win probability before the
match, and the "anomaly" is a real result that the model considered unlikely
— i.e. a genuine upset. This is the same z-score / probability-threshold
pattern as flagging an anomalous transaction or an out-of-spec sensor read,
just applied to football results instead of fraud or telemetry.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data.elo_ratings import EloRatingSystem
from src.data.combine_real_history import load_full_real_history
from src.data.real_data import build_seed_elo_ratings

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# A result is flagged as an upset when the team that actually won had a
# pre-match win probability below this threshold. 25% is a reasonable bar:
# below it, the model considered the winner a clear underdog.
UPSET_PROBABILITY_THRESHOLD = 0.25

# A draw is flagged as an upset when the favorite's pre-match win
# probability was above this threshold — i.e. the model expected a clear
# win, but the underdog held them to a draw. This is a genuinely different
# kind of upset (an "underdog earns a point") from a clean underdog win,
# and treating draws as automatically "not an upset" was a real blind spot:
# a 0-0 draw between a top-5 team and a tournament debutant is exactly the
# kind of shock result this detector exists to catch.
DRAW_UPSET_FAVORITE_PROBABILITY_THRESHOLD = 0.75

# Goal-difference threshold for flagging a "statement win" — a big margin
# of victory that goes beyond just winning, regardless of who was favored.
STATEMENT_WIN_GOAL_DIFF = 3


def _elo_diff_to_win_prob(elo_diff: float) -> float:
    """Standard ELO formula: probability the higher-rated side wins."""
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def detect_upsets(real_results_year: int = 2026) -> pd.DataFrame:
    """
    Re-plays real World Cup history chronologically, computing each team's
    pre-match ELO-implied win probability, then flags matches where the
    actual winner was a significant underdog (an "upset") or won by an
    unusually large margin (a "statement win").

    This intentionally processes EVERY real historical match chronologically
    first to build accurate pre-match ratings, then reports anomalies only
    for the requested year — so a 2026 upset is judged against ratings that
    reflect real pre-2026 history, not a cold start.
    """
    full_history = load_full_real_history()
    seed_ratings = build_seed_elo_ratings()
    elo = EloRatingSystem(seed_ratings=seed_ratings)

    records = []
    for _, row in full_history.iterrows():
        home_rating = elo.get_rating(row["home_team"])
        away_rating = elo.get_rating(row["away_team"])
        elo_diff = home_rating - away_rating

        home_win_prob = _elo_diff_to_win_prob(elo_diff)
        away_win_prob = 1.0 - home_win_prob

        if row["home_score"] > row["away_score"]:
            winner, winner_prob = row["home_team"], home_win_prob
        elif row["away_score"] > row["home_score"]:
            winner, winner_prob = row["away_team"], away_win_prob
        else:
            winner, winner_prob = None, None  # a clean win/loss "winner" doesn't apply to draws

        goal_diff = abs(row["home_score"] - row["away_score"])

        # Upset detection has two real cases, not one:
        # 1. Clear win/loss where the actual winner was a significant underdog
        # 2. A draw where one side was a heavy pre-match favorite (e.g. a
        #    top-5 team held scoreless by a tournament debutant) — this is
        #    just as much a shock result as an outright underdog win, and
        #    excluding draws entirely would miss exactly this kind of result.
        is_clear_upset = winner_prob is not None and winner_prob < UPSET_PROBABILITY_THRESHOLD
        is_draw_upset = (
            winner is None
            and max(home_win_prob, away_win_prob) >= DRAW_UPSET_FAVORITE_PROBABILITY_THRESHOLD
        )
        favorite_held_to_draw = None
        if is_draw_upset:
            favorite_held_to_draw = row["home_team"] if home_win_prob >= away_win_prob else row["away_team"]

        records.append({
            "date": row["date"],
            "year": row["year"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "stage": row.get("stage", "Unknown"),
            "home_win_prob_pre_match": round(home_win_prob, 3),
            "away_win_prob_pre_match": round(away_win_prob, 3),
            "winner": winner,
            "winner_pre_match_prob": round(winner_prob, 3) if winner_prob is not None else None,
            "goal_diff": goal_diff,
            "is_upset": bool(is_clear_upset or is_draw_upset),
            "upset_type": (
                "underdog_win" if is_clear_upset else
                "favorite_held_to_draw" if is_draw_upset else
                None
            ),
            "favorite_held_to_draw": favorite_held_to_draw,
            "is_statement_win": bool(goal_diff >= STATEMENT_WIN_GOAL_DIFF),
        })

        # update ratings AFTER recording this match's pre-match probabilities
        elo.process_match(
            row["home_team"], row["away_team"],
            row["home_score"], row["away_score"],
            neutral_venue=bool(row.get("neutral", False)),
            date=row["date"],
        )

    all_matches = pd.DataFrame(records)
    target = all_matches[all_matches["year"] == real_results_year].reset_index(drop=True)

    out_path = PROCESSED_DIR / f"upset_detection_{real_results_year}.csv"
    target.to_csv(out_path, index=False)

    return target


def summarize_upsets(upsets_df: pd.DataFrame) -> dict:
    """Quick rollup stats for a dashboard summary card."""
    if upsets_df.empty or "upset_type" not in upsets_df.columns:
        return {
            "total_matches": len(upsets_df),
            "upsets_detected": 0,
            "upset_rate": 0.0,
            "underdog_wins": 0,
            "favorites_held_to_draw": 0,
            "statement_wins": 0,
            "biggest_upset": None,
            "biggest_draw_upset": None,
        }

    flagged = upsets_df[upsets_df["is_upset"]]
    statements = upsets_df[upsets_df["is_statement_win"]]
    underdog_wins = flagged[flagged["upset_type"] == "underdog_win"]
    draw_upsets = flagged[flagged["upset_type"] == "favorite_held_to_draw"]

    return {
        "total_matches": len(upsets_df),
        "upsets_detected": len(flagged),
        "upset_rate": round(len(flagged) / len(upsets_df), 3) if len(upsets_df) else 0.0,
        "underdog_wins": len(underdog_wins),
        "favorites_held_to_draw": len(draw_upsets),
        "statement_wins": len(statements),
        "biggest_upset": (
            underdog_wins.sort_values("winner_pre_match_prob").iloc[0].to_dict()
            if len(underdog_wins) else None
        ),
        "biggest_draw_upset": (
            draw_upsets.assign(
                fav_prob=draw_upsets[["home_win_prob_pre_match", "away_win_prob_pre_match"]].max(axis=1)
            ).sort_values("fav_prob", ascending=False).iloc[0].to_dict()
            if len(draw_upsets) else None
        ),
    }


if __name__ == "__main__":
    results = detect_upsets()
    summary = summarize_upsets(results)

    print(f"Analyzed {summary['total_matches']} real 2026 matches")
    print(f"Upsets detected: {summary['upsets_detected']} ({summary['upset_rate']:.1%} of matches)")
    print(f"  - Underdog wins: {summary['underdog_wins']}")
    print(f"  - Favorites held to a draw: {summary['favorites_held_to_draw']}")
    print(f"Statement wins (3+ goal margin): {summary['statement_wins']}\n")

    if summary["biggest_upset"]:
        u = summary["biggest_upset"]
        print(f"Biggest underdog win: {u['winner']} won despite only a "
              f"{u['winner_pre_match_prob']:.1%} pre-match win probability "
              f"({u['home_team']} {u['home_score']}-{u['away_score']} {u['away_team']})")

    if summary["biggest_draw_upset"]:
        d = summary["biggest_draw_upset"]
        print(f"Biggest favorite-held-to-draw: {d['favorite_held_to_draw']} was expected to win big but only drew "
              f"({d['home_team']} {d['home_score']}-{d['away_score']} {d['away_team']})")

    print("\nAll flagged upsets:")
    print(results[results["is_upset"]][
        ["date", "home_team", "away_team", "home_score", "away_score", "upset_type", "winner", "favorite_held_to_draw"]
    ].to_string(index=False))
