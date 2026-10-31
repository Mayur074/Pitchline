"""
ELO rating system for international football teams.

Implements a standard ELO update rule adapted for football, with adjustments
for goal difference and home advantage, similar in spirit to the approach
used by FiveThirtyEight's Soccer Power Index and the World Football ELO
Ratings project.
"""

from collections import defaultdict
import pandas as pd
import numpy as np

BASE_RATING = 1500.0
K_FACTOR = 32.0
HOME_ADVANTAGE = 60.0


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard ELO expected score (probability A 'wins' the rating exchange)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _goal_diff_multiplier(goal_diff: int) -> float:
    """
    Scales the K-factor by margin of victory.
    A 1-goal win counts less than a 3+ goal win, capped to avoid blowouts
    dominating the rating update.
    """
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8.0


class EloRatingSystem:
    """
    Maintains and updates ELO ratings for all teams across a sequence of matches.
    """

    def __init__(
        self,
        base_rating: float = BASE_RATING,
        k_factor: float = K_FACTOR,
        seed_ratings: dict | None = None,
    ):
        """
        Args:
            base_rating: default rating for any team not in seed_ratings.
            k_factor: ELO update sensitivity.
            seed_ratings: optional dict of team -> starting rating, used to
                initialize ratings from a real source (e.g. official FIFA
                rankings) instead of a flat default for every team.
        """
        self.ratings: dict[str, float] = defaultdict(lambda: base_rating)
        if seed_ratings:
            for team, rating in seed_ratings.items():
                self.ratings[team] = rating
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.history: list[dict] = []

    def get_rating(self, team: str) -> float:
        return self.ratings[team]

    def process_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        neutral_venue: bool = False,
        date: pd.Timestamp | None = None,
    ) -> dict:
        """
        Updates ratings for both teams based on a single match result.

        Returns a dict snapshot of pre-match ratings, used for feature engineering
        (we want the rating BEFORE the match as a predictive feature, not after).
        """
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]

        home_adj = home_rating + (0 if neutral_venue else HOME_ADVANTAGE)

        expected_home = _expected_score(home_adj, away_rating)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        goal_diff = abs(home_score - away_score)
        multiplier = _goal_diff_multiplier(goal_diff)

        delta = self.k_factor * multiplier * (actual_home - expected_home)

        snapshot = {
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "home_rating_pre": home_rating,
            "away_rating_pre": away_rating,
            "expected_home_result": expected_home,
            "actual_home_result": actual_home,
            "rating_delta": delta,
        }
        self.history.append(snapshot)

        self.ratings[home_team] = home_rating + delta
        self.ratings[away_team] = away_rating - delta

        return snapshot

    def build_ratings_from_history(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a full historical match dataframe chronologically and returns
        a dataframe of pre-match ratings for every match, suitable for joining
        back as model features.

        Expects columns: date, home_team, away_team, home_score, away_score,
        and optionally `neutral`.
        """
        records = []
        for _, row in matches.iterrows():
            neutral = bool(row.get("neutral", False))
            snapshot = self.process_match(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_score=row["home_score"],
                away_score=row["away_score"],
                neutral_venue=neutral,
                date=row["date"],
            )
            records.append(snapshot)

        return pd.DataFrame(records)

    def current_ratings_table(self) -> pd.DataFrame:
        """Returns a sorted snapshot of current ratings for all teams seen so far."""
        df = pd.DataFrame(
            [{"team": team, "elo_rating": rating} for team, rating in self.ratings.items()]
        )
        return df.sort_values("elo_rating", ascending=False).reset_index(drop=True)
