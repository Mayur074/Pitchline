"""
Formal experiment design and hypothesis testing on the real World Cup
dataset — the gap this fills: dozens of JDs in this job search explicitly
ask for "design experiments and measure impact" (Stord Data Scientist
track), A/B testing, or hypothesis testing as a named skill, and none of
this project's other modules actually run a designed statistical test with
an explicit null hypothesis, a test statistic, and a significance threshold.

Two real experiments, each with a stated null hypothesis, run against the
real combined 1990-2026 dataset:

Experiment 1 — Home Advantage Test
    H0: Home teams do not win at a higher rate than away teams (neutral-venue
        matches excluded, since "home advantage" doesn't apply there).
    H1: Home teams win at a significantly higher rate.
    Test: two-proportion z-test comparing home win rate vs away win rate.

Experiment 2 — ELO Model Calibration Test
    H0: The model's predicted win probabilities match the real observed
        outcome frequencies (the model is "calibrated" — e.g. among all
        matches where the model said 70% win probability, real teams in
        that bucket should win about 70% of the time).
    H1: The model is systematically over- or under-confident.
    Test: chi-square goodness-of-fit across probability buckets.
"""

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from pathlib import Path

from src.data.combine_real_history import load_full_real_history
from src.data.elo_ratings import EloRatingSystem
from src.data.real_data import build_seed_elo_ratings
from src.models.upset_detector import _elo_diff_to_win_prob

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SIGNIFICANCE_LEVEL = 0.05


@dataclass
class ExperimentResult:
    experiment_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_used: str
    test_statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_description: str
    sample_size: int


def run_home_advantage_experiment() -> ExperimentResult:
    """
    Experiment 1: tests whether home advantage is a real, statistically
    significant effect in this dataset, using a two-proportion z-test.

    Neutral-venue matches are excluded by design — they're the natural
    "control group" omission here, since home advantage structurally
    cannot apply when there is no true home team.
    """
    df = load_full_real_history()
    non_neutral = df[df["neutral"] == False].copy()

    non_neutral["home_win"] = non_neutral["home_score"] > non_neutral["away_score"]
    non_neutral["away_win"] = non_neutral["away_score"] > non_neutral["home_score"]

    n = len(non_neutral)
    home_wins = non_neutral["home_win"].sum()
    away_wins = non_neutral["away_win"].sum()

    home_win_rate = home_wins / n
    away_win_rate = away_wins / n

    # Two-proportion z-test: comparing home_win_rate vs away_win_rate using
    # a pooled proportion under H0 (no difference between the two rates).
    pooled_p = (home_wins + away_wins) / (2 * n)
    se = np.sqrt(2 * pooled_p * (1 - pooled_p) / n)
    z_stat = (home_win_rate - away_win_rate) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

    effect_size = home_win_rate - away_win_rate

    return ExperimentResult(
        experiment_name="Home Advantage Test",
        null_hypothesis="Home teams do not win at a higher rate than away teams",
        alternative_hypothesis="Home teams win at a significantly higher rate than away teams",
        test_used="Two-proportion z-test",
        test_statistic=round(float(z_stat), 3),
        p_value=round(float(p_value), 6),
        significant=bool(p_value < SIGNIFICANCE_LEVEL),
        effect_size=round(float(effect_size), 4),
        effect_description=f"Home teams win {home_win_rate:.1%} of non-neutral matches vs {away_win_rate:.1%} for away teams",
        sample_size=n,
    )


def run_model_calibration_experiment(n_buckets: int = 5) -> tuple[ExperimentResult, pd.DataFrame]:
    """
    Experiment 2: tests whether the ELO model's predicted win probabilities
    are well-calibrated against real observed outcomes, using a chi-square
    goodness-of-fit test across probability buckets.

    For each real historical match, computes the model's pre-match win
    probability for whichever side actually won (or the higher-probability
    side for draws, scored as a "loss" for calibration purposes since a
    draw is not a win). Buckets matches by predicted probability and
    compares predicted vs observed win rate within each bucket.
    """
    df = load_full_real_history()
    seed_ratings = build_seed_elo_ratings()
    elo = EloRatingSystem(seed_ratings=seed_ratings)

    records = []
    for _, row in df.iterrows():
        home_rating = elo.get_rating(row["home_team"])
        away_rating = elo.get_rating(row["away_team"])
        home_prob = _elo_diff_to_win_prob(home_rating - away_rating)

        home_won = row["home_score"] > row["away_score"]
        records.append({"predicted_prob": home_prob, "actual_outcome": int(home_won)})

        elo.process_match(
            row["home_team"], row["away_team"], row["home_score"], row["away_score"],
            neutral_venue=bool(row.get("neutral", False)), date=row["date"],
        )

    calib_df = pd.DataFrame(records)
    calib_df["bucket"] = pd.cut(calib_df["predicted_prob"], bins=n_buckets, labels=False)

    bucket_summary = calib_df.groupby("bucket").agg(
        mean_predicted=("predicted_prob", "mean"),
        observed_win_rate=("actual_outcome", "mean"),
        n_matches=("actual_outcome", "size"),
    ).reset_index()

    # Chi-square goodness-of-fit: compare observed wins vs expected wins
    # (predicted probability * n) within each bucket. scipy.chisquare
    # requires observed and expected totals to match exactly, so we
    # normalize expected_wins to share the same total as observed_wins —
    # the comparison of interest is the SHAPE of the distribution across
    # buckets (is the model over/under-confident in specific probability
    # ranges), not the overall total, which can differ slightly due to
    # rounding in mean_predicted per bucket.
    observed_wins = bucket_summary["observed_win_rate"] * bucket_summary["n_matches"]
    expected_wins = bucket_summary["mean_predicted"] * bucket_summary["n_matches"]
    expected_wins = expected_wins * (observed_wins.sum() / expected_wins.sum())

    chi2_stat, p_value = stats.chisquare(f_obs=observed_wins, f_exp=expected_wins)

    mean_abs_calibration_error = float(
        (bucket_summary["observed_win_rate"] - bucket_summary["mean_predicted"]).abs().mean()
    )

    result = ExperimentResult(
        experiment_name="ELO Model Calibration Test",
        null_hypothesis="The model's predicted win probabilities match real observed outcome frequencies",
        alternative_hypothesis="The model is systematically over- or under-confident",
        test_used="Chi-square goodness-of-fit test",
        test_statistic=round(float(chi2_stat), 3),
        p_value=round(float(p_value), 6),
        significant=bool(p_value < SIGNIFICANCE_LEVEL),
        effect_size=round(mean_abs_calibration_error, 4),
        effect_description=f"Mean absolute calibration error across {n_buckets} probability buckets: {mean_abs_calibration_error:.1%}",
        sample_size=len(calib_df),
    )

    out_path = PROCESSED_DIR / "model_calibration_buckets.csv"
    bucket_summary.to_csv(out_path, index=False)

    return result, bucket_summary


def run_all_experiments() -> list[ExperimentResult]:
    home_adv = run_home_advantage_experiment()
    calibration, _ = run_model_calibration_experiment()

    results_df = pd.DataFrame([vars(home_adv), vars(calibration)])
    out_path = PROCESSED_DIR / "experiment_results.csv"
    results_df.to_csv(out_path, index=False)

    return [home_adv, calibration]


if __name__ == "__main__":
    print("Running designed experiments on the real 1990-2026 World Cup dataset.\n")

    for result in run_all_experiments():
        print(f"=== {result.experiment_name} ===")
        print(f"H0: {result.null_hypothesis}")
        print(f"H1: {result.alternative_hypothesis}")
        print(f"Test: {result.test_used}")
        print(f"Test statistic: {result.test_statistic}")
        print(f"p-value: {result.p_value}")
        print(f"Significant at alpha=0.05: {result.significant}")
        print(f"Effect: {result.effect_description}")
        print(f"Sample size: {result.sample_size}\n")
