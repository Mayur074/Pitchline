"""
Fetches historical international football match data and current FIFA rankings.

Data sources:
- Historical results: kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
  (extended with 2018-2026 results for World Cup 2026 relevance)
- FIFA Rankings: fetched via public ranking snapshots

For reproducibility, a cached sample dataset is included in data/raw/ so the
pipeline runs end-to-end without requiring live API access.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_historical_matches() -> pd.DataFrame:
    """
    Load historical international match results.

    Expected columns: date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral

    Returns:
        DataFrame of historical match results, sorted by date.
    """
    matches_path = RAW_DATA_DIR / "international_matches.csv"

    if not matches_path.exists():
        logger.warning(
            "No cached match data found at %s. Run scripts/download_sample_data.py "
            "first, or place a results CSV at this path.",
            matches_path,
        )
        raise FileNotFoundError(
            f"Expected historical match data at {matches_path}. "
            "See README.md Quickstart for setup instructions."
        )

    df = pd.read_csv(matches_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d historical matches spanning %s to %s",
                len(df), df["date"].min().date(), df["date"].max().date())
    return df


def fetch_fifa_rankings(as_of_date: str | None = None) -> pd.DataFrame:
    """
    Load FIFA World Ranking points snapshot.

    Args:
        as_of_date: optional ISO date string to filter the most recent ranking
            snapshot at or before this date. Defaults to the latest available.

    Returns:
        DataFrame with columns: team, ranking_date, rank, points
    """
    rankings_path = RAW_DATA_DIR / "fifa_rankings.csv"

    if not rankings_path.exists():
        raise FileNotFoundError(
            f"Expected FIFA rankings data at {rankings_path}. "
            "See README.md Quickstart for setup instructions."
        )

    df = pd.read_csv(rankings_path, parse_dates=["ranking_date"])

    if as_of_date:
        cutoff = pd.Timestamp(as_of_date)
        df = df[df["ranking_date"] <= cutoff]
        latest_per_team = df.sort_values("ranking_date").groupby("team").tail(1)
        return latest_per_team.reset_index(drop=True)

    latest_per_team = df.sort_values("ranking_date").groupby("team").tail(1)
    return latest_per_team.reset_index(drop=True)


def fetch_upcoming_fixtures() -> pd.DataFrame:
    """
    Load the upcoming World Cup 2026 fixture list.

    Returns:
        DataFrame with columns: date, home_team, away_team, stage, venue
    """
    fixtures_path = RAW_DATA_DIR / "wc2026_fixtures.csv"

    if not fixtures_path.exists():
        raise FileNotFoundError(
            f"Expected fixture list at {fixtures_path}. "
            "See README.md Quickstart for setup instructions."
        )

    df = pd.read_csv(fixtures_path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    matches = fetch_historical_matches()
    rankings = fetch_fifa_rankings()
    fixtures = fetch_upcoming_fixtures()
    print(f"Matches: {len(matches)} | Rankings: {len(rankings)} | Fixtures: {len(fixtures)}")
