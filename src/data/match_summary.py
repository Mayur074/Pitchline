"""
Builds an at-a-glance match summary ("what happened in this match without
watching it") from real goal-event data: who scored, when, and the resulting
scoreline momentum across the 90+ minutes.

This is explicitly NOT a pitch heatmap (player positions, touch maps, etc.) —
we don't have that data, and faking it would violate the real-data principle
this whole project is built on. What we genuinely have, from the real
openfootball/worldcup.json feed, is goal scorer names and minutes (including
stoppage time, penalties, and own goals), which is enough to build something
real and useful: a momentum timeline showing exactly when control of the
match shifted.
"""

import re
import requests
import pandas as pd
from dataclasses import dataclass, field

SOURCE_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"

# Accept either spelling so dashboard dropdowns and user input don't have to
# match special characters exactly.
TEAM_NAME_VARIANTS = {
    "Curacao": "Curaçao",
    "Cote d'Ivoire": "Ivory Coast",
}


def _resolve_team_name(name: str) -> str:
    return TEAM_NAME_VARIANTS.get(name, name)


@dataclass
class GoalEvent:
    minute_display: str   # e.g. "90+7" as shown in the real data
    minute_sort: float    # numeric value for sorting/plotting, e.g. 90.7
    team: str
    scorer: str
    is_penalty: bool = False
    is_own_goal: bool = False


@dataclass
class MatchSummary:
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    date: str
    stage: str
    ground: str
    half_time_score: tuple[int, int] | None
    goal_events: list[GoalEvent] = field(default_factory=list)
    momentum_timeline: list[dict] = field(default_factory=list)


def _parse_minute(minute_str: str) -> float:
    """
    Converts a real match minute string into a sortable/plottable number.
    Handles stoppage time notation: "90+7" -> 90.7, "45+5" -> 45.5
    """
    match = re.match(r"(\d+)(?:\+(\d+))?", minute_str)
    if not match:
        return 0.0
    base = int(match.group(1))
    stoppage = int(match.group(2)) if match.group(2) else 0
    return base + stoppage / 10.0


def _build_momentum_timeline(goal_events: list[GoalEvent], home_team: str) -> list[dict]:
    """
    Tracks the running scoreline after each goal, which is what "momentum"
    actually means here — we don't have possession/territory data, but we
    do have the real, factual sequence of who was leading and by how much
    at every point in the match.
    """
    timeline = [{"minute": 0.0, "home_score": 0, "away_score": 0, "event": None}]
    home_running, away_running = 0, 0

    for event in sorted(goal_events, key=lambda e: e.minute_sort):
        if event.team == home_team:
            home_running += 1
        else:
            away_running += 1
        timeline.append({
            "minute": event.minute_sort,
            "home_score": home_running,
            "away_score": away_running,
            "event": event,
        })

    return timeline


def _extract_goals(raw_goals: list[dict], team_name: str) -> list[GoalEvent]:
    events = []
    for g in raw_goals:
        minute_str = g.get("minute", "0")
        events.append(GoalEvent(
            minute_display=minute_str,
            minute_sort=_parse_minute(minute_str),
            team=team_name,
            scorer=g.get("name", "Unknown"),
            is_penalty=g.get("penalty", False),
            is_own_goal=g.get("owngoal", False),
        ))
    return events


def fetch_match_summary(home_team: str, away_team: str, date: str | None = None) -> MatchSummary | None:
    """
    Fetches the real goal-event data for a specific completed match and
    builds a MatchSummary. Returns None if the match can't be found or
    hasn't been played yet (no final score available).

    Matching is team-pair based (order-independent) since some matches in
    the feed may have home/away recorded differently than expected.
    """
    response = requests.get(SOURCE_URL, timeout=15)
    response.raise_for_status()
    data = response.json()

    target_pair = frozenset([_resolve_team_name(home_team), _resolve_team_name(away_team)])

    for match in data["matches"]:
        team1, team2 = match.get("team1", ""), match.get("team2", "")
        if frozenset([team1, team2]) != target_pair:
            continue

        score = match.get("score", {})
        ft = score.get("ft")
        if ft is None:
            continue  # not played yet

        if date and match.get("date") != date:
            continue

        ht = score.get("ht")
        goals1 = _extract_goals(match.get("goals1", []), team1)
        goals2 = _extract_goals(match.get("goals2", []), team2)
        all_goals = goals1 + goals2

        summary = MatchSummary(
            home_team=team1,
            away_team=team2,
            home_score=ft[0],
            away_score=ft[1],
            date=match.get("date", ""),
            stage=match.get("group") or match.get("round", "Unknown"),
            ground=match.get("ground", ""),
            half_time_score=tuple(ht) if ht else None,
            goal_events=sorted(all_goals, key=lambda e: e.minute_sort),
        )
        summary.momentum_timeline = _build_momentum_timeline(all_goals, team1)
        return summary

    return None


def list_available_completed_matches() -> pd.DataFrame:
    """
    Returns a simple table of every completed real match available for
    summary lookup, for populating a dropdown selector in the dashboard.
    """
    response = requests.get(SOURCE_URL, timeout=15)
    response.raise_for_status()
    data = response.json()

    rows = []
    for match in data["matches"]:
        ft = match.get("score", {}).get("ft")
        if ft is None:
            continue
        rows.append({
            "date": match.get("date"),
            "home_team": match.get("team1"),
            "away_team": match.get("team2"),
            "home_score": ft[0],
            "away_score": ft[1],
            "stage": match.get("group") or match.get("round", "Unknown"),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    available = list_available_completed_matches()
    print(f"{len(available)} completed real matches available for summary.\n")

    sample = available.iloc[-1]
    summary = fetch_match_summary(sample["home_team"], sample["away_team"])
    print(f"{summary.home_team} {summary.home_score}-{summary.away_score} {summary.away_team}")
    print(f"Half-time: {summary.half_time_score}")
    for event in summary.goal_events:
        tag = " (pen)" if event.is_penalty else " (OG)" if event.is_own_goal else ""
        print(f"  {event.minute_display}' {event.team} - {event.scorer}{tag}")
