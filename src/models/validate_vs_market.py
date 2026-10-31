"""
Benchmarks our ELO-based model against real betting market probabilities
for the same real upcoming World Cup 2026 fixtures.

This is the most honest validation available before the matches are played:
we can't yet know who wins, but we CAN check whether our model agrees with
the market on WHICH team is favored and by roughly how much. Professional
sports books aggregate enormous amounts of information (injuries, lineups,
sharp money) that a simple ELO model doesn't have access to, so close
agreement with the market is a meaningful signal that the model is sane,
even before any of these matches kick off.
"""

import pandas as pd
from pathlib import Path

from src.data.build_real_features import build_real_fixture_predictions_input

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def elo_diff_to_implied_prob(elo_diff: float) -> float:
    """Standard ELO win probability formula, used for a same-scale comparison to market odds."""
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def compare_model_to_market() -> pd.DataFrame:
    """
    For every real upcoming fixture with a known market probability, compares
    our ELO-implied home win probability to the market's home win probability
    and reports the agreement.
    """
    df = build_real_fixture_predictions_input()
    has_market = df.dropna(subset=["market_home_win_pct"]).copy()

    has_market["elo_implied_home_win_pct"] = (
        has_market["elo_diff"].apply(elo_diff_to_implied_prob) * 100
    )

    has_market["abs_difference"] = (
        has_market["elo_implied_home_win_pct"] - has_market["market_home_win_pct"]
    ).abs()

    has_market["same_favorite"] = (
        (has_market["elo_diff"] > 0) == (has_market["market_home_win_pct"] > 50)
    )

    output_cols = [
        "home_team", "away_team", "elo_implied_home_win_pct",
        "market_home_win_pct", "abs_difference", "same_favorite",
    ]
    result = has_market[output_cols].round(1)

    out_path = PROCESSED_DIR / "model_vs_market_comparison.csv"
    result.to_csv(out_path, index=False)

    return result


if __name__ == "__main__":
    comparison = compare_model_to_market()
    print(comparison.to_string(index=False))

    agreement_rate = comparison["same_favorite"].mean()
    mean_abs_diff = comparison["abs_difference"].mean()

    print(f"\nFavorite agreement with real betting market: {agreement_rate:.0%}")
    print(f"Mean absolute difference in implied win probability: {mean_abs_diff:.1f} points")
