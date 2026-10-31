"""
Generates a realistic synthetic sample dataset for international football
matches, FIFA rankings, and World Cup 2026 fixtures.

This allows the full pipeline (fetch -> features -> train -> predict) to run
end-to-end in CI and for anyone cloning the repo, without requiring a live
data source or API key. Swap this out for real historical data by placing
your own CSVs at the same paths (see README Quickstart).
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TEAMS = [
    "Brazil", "Argentina", "France", "Germany", "Spain", "England", "Portugal",
    "Netherlands", "Belgium", "Italy", "Croatia", "Uruguay", "Mexico", "USA",
    "Japan", "South Korea", "Morocco", "Senegal", "Colombia", "Switzerland",
    "Canada", "Australia", "Denmark", "Poland", "Ghana", "Tunisia", "Norway",
]

TEAM_STRENGTH = {team: np.random.normal(1500, 200) for team in TEAMS}
TEAM_STRENGTH.update({
    "Brazil": 1950, "Argentina": 1920, "France": 1900, "England": 1850,
    "Spain": 1840, "Germany": 1820, "Portugal": 1810, "Netherlands": 1790,
})


def generate_historical_matches(n_matches: int = 6000) -> pd.DataFrame:
    """Simulates historical match results weighted by synthetic team strength."""
    rows = []
    start_date = pd.Timestamp("2010-01-01")

    for i in range(n_matches):
        home, away = np.random.choice(TEAMS, size=2, replace=False)
        home_strength = TEAM_STRENGTH[home] + 50  # home advantage baked into simulation
        away_strength = TEAM_STRENGTH[away]

        strength_diff = (home_strength - away_strength) / 400.0
        win_prob = 1 / (1 + 10 ** (-strength_diff))

        outcome_roll = np.random.random()
        if outcome_roll < win_prob * 0.75:
            home_score = np.random.poisson(2.0) + 1
            away_score = max(0, home_score - np.random.randint(1, 3))
        elif outcome_roll > 1 - (1 - win_prob) * 0.75:
            away_score = np.random.poisson(2.0) + 1
            home_score = max(0, away_score - np.random.randint(1, 3))
        else:
            draw_score = np.random.poisson(1.3)
            home_score = away_score = draw_score

        match_date = start_date + pd.Timedelta(days=int(i * (5800 / n_matches)))

        rows.append({
            "date": match_date,
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": np.random.choice(
                ["Friendly", "World Cup Qualifier", "Continental Championship", "World Cup"],
                p=[0.5, 0.3, 0.15, 0.05],
            ),
            "neutral": bool(np.random.random() < 0.2),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def generate_fifa_rankings() -> pd.DataFrame:
    """Synthetic FIFA ranking snapshots, monthly, derived from team strength."""
    rows = []
    ranking_dates = pd.date_range("2024-01-01", "2026-06-01", freq="MS")

    for date in ranking_dates:
        strengths = {
            team: TEAM_STRENGTH[team] + np.random.normal(0, 15) for team in TEAMS
        }
        ranked = sorted(strengths.items(), key=lambda x: -x[1])
        for rank, (team, strength) in enumerate(ranked, start=1):
            rows.append({
                "team": team,
                "ranking_date": date,
                "rank": rank,
                "points": round(strength, 1),
            })

    return pd.DataFrame(rows)


def generate_wc2026_fixtures() -> pd.DataFrame:
    """A representative slice of World Cup 2026 group stage fixtures."""
    groups = {
        "A": ["Mexico", "South Korea", "Morocco", "Australia"],
        "B": ["Brazil", "Switzerland", "Senegal", "Poland"],
        "C": ["France", "Croatia", "Ghana", "Canada"],
        "D": ["Argentina", "Denmark", "Tunisia", "Norway"],
        "E": ["England", "Netherlands", "Colombia", "Japan"],
        "F": ["Spain", "Germany", "Uruguay", "USA"],
        "G": ["Portugal", "Belgium", "Italy", "Italy"],  # placeholder slot
    }

    rows = []
    base_date = pd.Timestamp("2026-06-21")
    match_id = 0

    for group_letter, teams in groups.items():
        teams = [t for t in teams if t in TEAMS]
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                rows.append({
                    "date": base_date + pd.Timedelta(days=match_id % 14),
                    "home_team": teams[i],
                    "away_team": teams[j],
                    "stage": "Group Stage",
                    "venue": f"Group {group_letter} Venue",
                })
                match_id += 1

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    matches = generate_historical_matches()
    matches.to_csv(RAW_DIR / "international_matches.csv", index=False)
    print(f"Wrote {len(matches)} historical matches")

    rankings = generate_fifa_rankings()
    rankings.to_csv(RAW_DIR / "fifa_rankings.csv", index=False)
    print(f"Wrote {len(rankings)} ranking records")

    fixtures = generate_wc2026_fixtures()
    fixtures.to_csv(RAW_DIR / "wc2026_fixtures.csv", index=False)
    print(f"Wrote {len(fixtures)} upcoming fixtures")
