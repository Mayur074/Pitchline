"""
Updates data/raw/wc2026_real_results.csv with newly completed World Cup 2026
matches, fetched automatically from a free, public, no-API-key data source.

Data source: openfootball/worldcup.json on GitHub
    https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json
    - Public domain (CC0), no API key, no rate limit, no auth required
    - Maintained by the long-running openfootball project (300+ stars)
    - Includes finished match scores AND scheduled future fixtures
    - Knockout-stage placeholders (e.g. "2A", "3C/D/F") appear until those
      slots are mathematically determined by group results

This script is safe to re-run as often as you like:
    - Only matches with a real final score are considered "completed"
    - Already-present matches (same date + home_team + away_team) are
      automatically skipped, so no duplicates are ever created
    - Matches with placeholder team names (round of 32 onward, before
      groups finish) are skipped until real team names appear in the feed

Usage:
    python scripts/update_2026_results.py            # fetch + apply real update
    python scripts/update_2026_results.py --dry-run   # preview only, no write
"""

import argparse
import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_PATH = RAW_DIR / "wc2026_real_results.csv"

SOURCE_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"

REQUIRED_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "neutral", "stage", "year",
]

# Team name aliases: the openfootball feed uses slightly different naming
# conventions than our existing dataset in places. Map their names to ours
# so ELO ratings track the same team consistently across both sources.
TEAM_NAME_ALIASES = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "Turkey": "Turkiye",
    "Iran": "IR Iran",
}


def _is_placeholder_team(name: str) -> bool:
    """
    Knockout-stage slots before the group stage finishes look like '2A' or
    '3C/D/F' rather than a real team name. Detect and skip these.
    """
    if not name:
        return True
    return any(ch.isdigit() for ch in name) and ("/" in name or len(name) <= 3)


def _normalize_team_name(name: str) -> str:
    return TEAM_NAME_ALIASES.get(name, name)


def fetch_live_2026_matches() -> list[dict]:
    """
    Fetches the current World Cup 2026 match feed and returns only the
    matches that have a real final score (i.e. have actually been played),
    in our internal schema.
    """
    response = requests.get(SOURCE_URL, timeout=15)
    response.raise_for_status()
    data = response.json()

    completed = []
    for match in data["matches"]:
        score = match.get("score", {})
        ft = score.get("ft")  # full-time score, e.g. [2, 0]

        if ft is None:
            continue  # not played yet

        team1 = match.get("team1", "")
        team2 = match.get("team2", "")
        if _is_placeholder_team(team1) or _is_placeholder_team(team2):
            continue  # knockout slot not yet resolved to a real team

        completed.append({
            "date": match["date"],
            "home_team": _normalize_team_name(team1),
            "away_team": _normalize_team_name(team2),
            "home_score": int(ft[0]),
            "away_score": int(ft[1]),
            "tournament": "World Cup",
            "neutral": match.get("round", "").lower() != "matchday 1" or True,
            "stage": match.get("group") or match.get("round", "Unknown Stage"),
            "year": 2026,
        })

    return completed


def load_existing_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Expected results file at {RESULTS_PATH}")
    return pd.read_csv(RESULTS_PATH, parse_dates=["date"])


def _validate_match(match: dict) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in match]
    if missing:
        raise ValueError(f"Match is missing required fields: {missing}. Got: {match}")
    if match["home_team"] == match["away_team"]:
        raise ValueError(f"home_team and away_team are identical: {match['home_team']}")
    if match["home_score"] < 0 or match["away_score"] < 0:
        raise ValueError(f"Scores cannot be negative: {match}")
    pd.Timestamp(match["date"])  # raises if invalid


def append_new_results(new_matches: list[dict], dry_run: bool = False) -> pd.DataFrame:
    """
    Appends newly completed real matches to the 2026 results file.

    Matches already present are skipped, matched on the pair of teams
    involved (regardless of home/away order) and a date within 1 day —
    different data sources sometimes record a match under the UTC date vs.
    the local kickoff date for late-night matches, so a strict same-day
    match would otherwise falsely treat real duplicates as new.
    """
    existing = load_existing_results()
    existing_lookup = [
        (row["date"], frozenset([row["home_team"], row["away_team"]]))
        for _, row in existing.iterrows()
    ]

    rows_to_add, skipped = [], []
    for match in new_matches:
        _validate_match(match)
        match_date = pd.Timestamp(match["date"])
        match_pair = frozenset([match["home_team"], match["away_team"]])

        is_duplicate = any(
            match_pair == existing_pair and abs((match_date - existing_date).days) <= 1
            for existing_date, existing_pair in existing_lookup
        )

        if is_duplicate:
            skipped.append(match)
        else:
            rows_to_add.append(match)

    print(f"Checked {len(new_matches)} completed matches from the live feed.")
    print(f"Already in dataset (skipped): {len(skipped)}")

    if not rows_to_add:
        print("No new matches to add. Dataset is already current.")
        return existing

    new_df = pd.DataFrame(rows_to_add)
    new_df["date"] = pd.to_datetime(new_df["date"])
    updated = pd.concat([existing, new_df], ignore_index=True).sort_values("date").reset_index(drop=True)

    print(f"\nAdding {len(rows_to_add)} new real match(es):")
    for m in rows_to_add:
        print(f"  {m['date']}: {m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']} ({m['stage']})")

    if dry_run:
        print("\n[DRY RUN] No changes written to disk.")
        return updated

    updated.to_csv(RESULTS_PATH, index=False)
    print(f"\nWrote {len(updated)} total matches to {RESULTS_PATH}")
    return updated


def run_update(dry_run: bool = False) -> pd.DataFrame:
    """Full pipeline: fetch live data, then append whatever is genuinely new."""
    print(f"Fetching live World Cup 2026 data from {SOURCE_URL} ...\n")
    live_matches = fetch_live_2026_matches()
    return append_new_results(live_matches, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing to disk")
    args = parser.parse_args()

    run_update(dry_run=args.dry_run)
