"""
Monte Carlo simulation of the remaining 2026 World Cup, using the real,
continuously-updated ELO ratings as the underlying win-probability model.

This is the same simulation pattern used in production demand forecasting
and scenario planning: rather than predicting a single point outcome, run
the season forward thousands of times with randomized outcomes drawn from
the model's real probability distribution at each step, then report the
distribution of final outcomes (e.g. "Argentina wins the tournament in 18%
of simulated futures"). This is structurally identical to Monte Carlo demand
forecasting or price elasticity scenario modeling — same method, different
domain.

Important honesty note: this simulates the REMAINING fixtures using the
real, currently-known group stage matchups, but since the actual knockout
bracket isn't determined until the group stage finishes, this module
simulates at the GROUP level (who finishes top 2 in each real group) rather
than fabricating a full knockout bracket with invented matchups. That keeps
every simulated match a real, scheduled fixture between real teams, never
an invented hypothetical pairing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.data.elo_ratings import EloRatingSystem
from src.data.real_data import build_seed_elo_ratings, load_real_upcoming_fixtures
from src.data.combine_real_history import load_full_real_history
from src.models.upset_detector import _elo_diff_to_win_prob

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _current_real_elo_ratings() -> EloRatingSystem:
    """Builds the ELO system from all real history through today, same as build_real_features.py."""
    seed_ratings = build_seed_elo_ratings()
    full_history = load_full_real_history()
    elo = EloRatingSystem(seed_ratings=seed_ratings)
    elo.build_ratings_from_history(full_history)
    return elo


def _simulate_one_match(home_team: str, away_team: str, elo: EloRatingSystem, rng: np.random.Generator) -> tuple[int, int, str]:
    """
    Draws a random outcome for a single real fixture, weighted by the
    model's real win/draw/loss probabilities derived from current ELO
    ratings. Returns (home_goals, away_goals, winner) - goals are a simple
    Poisson draw calibrated so the win/draw/loss split matches the ELO
    probabilities, not a separate goals model.
    """
    home_elo = elo.get_rating(home_team)
    away_elo = elo.get_rating(away_team)
    elo_diff = home_elo - away_elo

    home_win_prob = _elo_diff_to_win_prob(elo_diff)
    # crude but reasonable draw probability model: draws are more likely
    # when teams are evenly matched (elo_diff near 0)
    draw_prob = 0.28 * np.exp(-abs(elo_diff) / 400)
    away_win_prob = 1.0 - home_win_prob - draw_prob
    away_win_prob = max(away_win_prob, 0.0)

    probs = np.array([home_win_prob, draw_prob, away_win_prob])
    probs = probs / probs.sum()  # renormalize after clipping

    outcome = rng.choice(["home_win", "draw", "away_win"], p=probs)

    if outcome == "home_win":
        away_goals = rng.poisson(0.9)
        home_goals = away_goals + rng.poisson(1.2) + 1  # guarantees home_goals > away_goals
        return home_goals, away_goals, home_team
    elif outcome == "away_win":
        home_goals = rng.poisson(0.9)
        away_goals = home_goals + rng.poisson(1.2) + 1  # guarantees away_goals > home_goals
        return home_goals, away_goals, away_team
    else:
        goals = rng.poisson(1.2)
        return goals, goals, None


def simulate_group_outcomes(n_simulations: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Runs a Monte Carlo simulation of the real remaining group-stage fixtures
    n_simulations times, tracking how often each team finishes in 1st, 2nd,
    or fails to advance from their real group.

    Only fixtures that are REAL and currently scheduled (from
    wc2026_upcoming_fixtures_real.csv) are simulated — no invented matchups.
    """
    rng = np.random.default_rng(seed)
    fixtures = load_real_upcoming_fixtures()
    group_fixtures = fixtures[fixtures["stage"].str.startswith("Group", na=False)]

    base_elo = _current_real_elo_ratings()

    # group membership: infer from fixtures (every team appearing under a
    # given group label is in that group)
    group_of_team = {}
    for _, row in group_fixtures.iterrows():
        group_of_team[row["home_team"]] = row["stage"]
        group_of_team[row["away_team"]] = row["stage"]

    finish_counts = defaultdict(lambda: {"1st": 0, "2nd": 0, "eliminated": 0})

    for sim in range(n_simulations):
        # copy current points/goal-difference state fresh each simulation
        points = defaultdict(int)
        goal_diff = defaultdict(int)

        for _, fixture in group_fixtures.iterrows():
            home, away = fixture["home_team"], fixture["away_team"]
            h_goals, a_goals, winner = _simulate_one_match(home, away, base_elo, rng)

            goal_diff[home] += h_goals - a_goals
            goal_diff[away] += a_goals - h_goals

            if winner == home:
                points[home] += 3
            elif winner == away:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

        # rank within each group by (points, goal_diff) — simplified
        # tiebreak vs. real FIFA rules (head-to-head, goals scored, etc.)
        groups = defaultdict(list)
        for team, group in group_of_team.items():
            groups[group].append(team)

        for group, teams in groups.items():
            ranked = sorted(teams, key=lambda t: (points[t], goal_diff[t]), reverse=True)
            if len(ranked) >= 1:
                finish_counts[ranked[0]]["1st"] += 1
            if len(ranked) >= 2:
                finish_counts[ranked[1]]["2nd"] += 1
            for eliminated_team in ranked[2:]:
                finish_counts[eliminated_team]["eliminated"] += 1

    rows = []
    for team, counts in finish_counts.items():
        rows.append({
            "team": team,
            "group": group_of_team.get(team, "Unknown"),
            "pct_finish_1st": round(counts["1st"] / n_simulations, 3),
            "pct_finish_2nd": round(counts["2nd"] / n_simulations, 3),
            "pct_advance": round((counts["1st"] + counts["2nd"]) / n_simulations, 3),
            "pct_eliminated": round(counts["eliminated"] / n_simulations, 3),
        })

    result = pd.DataFrame(rows).sort_values(["group", "pct_advance"], ascending=[True, False]).reset_index(drop=True)

    out_path = PROCESSED_DIR / "group_stage_simulation.csv"
    result.to_csv(out_path, index=False)
    return result


if __name__ == "__main__":
    print("Running Monte Carlo simulation of remaining real group fixtures (10,000 runs)...\n")
    result = simulate_group_outcomes(n_simulations=10_000)
    print(result.to_string(index=False))
