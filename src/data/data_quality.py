"""
Data quality monitoring for the World Cup dataset, in the same spirit as a
production dbt test suite or a data engineering observability dashboard:
define explicit quality checks, run them against the real data, and report
pass/fail status with enough detail to act on a failure.

This mirrors language like "implement data quality standards", "own
reliability of downstream pipeline outputs", and "data validation and
monitoring" — the checks here are the same categories you'd find in a real
dbt project: null checks, referential integrity, freshness, and
business-rule validation (e.g. scores can't be negative, no team plays
itself).
"""

import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.data.combine_real_history import load_full_real_history

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class QualityCheckResult:
    check_name: str
    category: str
    passed: bool
    detail: str
    affected_rows: int = 0


def check_no_nulls_in_required_fields(df: pd.DataFrame) -> QualityCheckResult:
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    null_counts = df[required].isnull().sum()
    total_nulls = int(null_counts.sum())
    return QualityCheckResult(
        check_name="no_nulls_in_required_fields",
        category="Completeness",
        passed=total_nulls == 0,
        detail=(
            "All required fields populated" if total_nulls == 0
            else f"Found {total_nulls} null values across: {null_counts[null_counts > 0].to_dict()}"
        ),
        affected_rows=total_nulls,
    )


def check_no_negative_scores(df: pd.DataFrame) -> QualityCheckResult:
    bad = df[(df["home_score"] < 0) | (df["away_score"] < 0)]
    return QualityCheckResult(
        check_name="no_negative_scores",
        category="Validity",
        passed=len(bad) == 0,
        detail="No negative scores found" if len(bad) == 0 else f"{len(bad)} matches with negative scores",
        affected_rows=len(bad),
    )


def check_no_team_plays_itself(df: pd.DataFrame) -> QualityCheckResult:
    bad = df[df["home_team"] == df["away_team"]]
    return QualityCheckResult(
        check_name="no_team_plays_itself",
        category="Validity",
        passed=len(bad) == 0,
        detail="No self-matches found" if len(bad) == 0 else f"{len(bad)} matches where home_team == away_team",
        affected_rows=len(bad),
    )


# Real countries that appear under more than one valid name across data
# sources. Without normalizing these before comparing, a duplicate like
# "Portugal vs Congo DR" and "Portugal vs DR Congo" looks like two different
# matches to a naive string comparison — this exact bug shipped in this
# project's own dataset until this check caught it.
TEAM_NAME_CANONICAL_FORM = {
    "DR Congo": "Congo DR",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "Turkey": "Turkiye",
    "Iran": "IR Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def _canonical_team_name(name: str) -> str:
    return TEAM_NAME_CANONICAL_FORM.get(name, name)


def check_no_duplicate_matches(df: pd.DataFrame) -> QualityCheckResult:
    """
    Duplicate = same date + same pair of teams, regardless of home/away
    order AND regardless of which valid spelling of a team's name was used
    (e.g. "Congo DR" vs "DR Congo" are the same real country).
    """
    pairs = df.apply(
        lambda r: (
            r["date"],
            frozenset([_canonical_team_name(r["home_team"]), _canonical_team_name(r["away_team"])]),
        ),
        axis=1,
    )
    duplicate_count = int(pairs.duplicated().sum())
    return QualityCheckResult(
        check_name="no_duplicate_matches",
        category="Uniqueness",
        passed=duplicate_count == 0,
        detail="No duplicate matches found" if duplicate_count == 0 else f"{duplicate_count} duplicate match(es) found",
        affected_rows=duplicate_count,
    )


def check_expected_match_count_per_tournament(df: pd.DataFrame) -> QualityCheckResult:
    """
    Validates real, known World Cup match counts: 52 for the 24-team era
    (1990, 1994), 64 for the 32-team era (1998-2022). This is a real,
    verifiable business rule, not an arbitrary threshold.
    """
    expected = {1990: 52, 1994: 52, 1998: 64, 2002: 64, 2006: 64, 2010: 64, 2014: 64, 2018: 64, 2022: 64}
    actual = df[df["year"].isin(expected.keys())].groupby("year").size().to_dict()

    mismatches = {
        year: (expected[year], actual.get(year, 0))
        for year in expected
        if actual.get(year, 0) != expected[year]
    }

    return QualityCheckResult(
        check_name="expected_match_count_per_tournament",
        category="Business Rule",
        passed=len(mismatches) == 0,
        detail=(
            "All completed tournaments have the correct real match count"
            if not mismatches
            else f"Mismatches (expected, actual): {mismatches}"
        ),
        affected_rows=len(mismatches),
    )


def check_dates_within_real_tournament_windows(df: pd.DataFrame) -> QualityCheckResult:
    """
    Each tournament year's matches should fall within that World Cup's real
    known date range (roughly June-July for most editions; 2022 was
    Nov-Dec due to Qatar's climate).
    """
    real_windows = {
        1990: ("1990-06-08", "1990-07-08"), 1994: ("1994-06-17", "1994-07-17"),
        1998: ("1998-06-10", "1998-07-12"), 2002: ("2002-05-31", "2002-06-30"),
        2006: ("2006-06-09", "2006-07-09"), 2010: ("2010-06-11", "2010-07-11"),
        2014: ("2014-06-12", "2014-07-13"), 2018: ("2018-06-14", "2018-07-15"),
        2022: ("2022-11-20", "2022-12-18"),
    }

    violations = 0
    details = []
    for year, (start, end) in real_windows.items():
        year_data = df[df["year"] == year]
        out_of_range = year_data[(year_data["date"] < start) | (year_data["date"] > end)]
        if len(out_of_range):
            violations += len(out_of_range)
            details.append(f"{year}: {len(out_of_range)} match(es) outside {start} to {end}")

    return QualityCheckResult(
        check_name="dates_within_real_tournament_windows",
        category="Validity",
        passed=violations == 0,
        detail="All match dates fall within their tournament's real window" if violations == 0 else "; ".join(details),
        affected_rows=violations,
    )


def check_data_freshness(df: pd.DataFrame, max_staleness_days: int = 7,
                         as_of: str | None = None) -> QualityCheckResult:
    """
    Flags if the most recent 2026 match in the dataset is older than
    max_staleness_days — a real freshness check, the same pattern used to
    monitor whether a production pipeline has stopped receiving new data.
    `as_of` pins the reference date (defaults to today) so tests are not clock-dependent.
    """
    matches_2026 = df[df["year"] == 2026]
    if matches_2026.empty:
        return QualityCheckResult(
            check_name="data_freshness",
            category="Freshness",
            passed=False,
            detail="No 2026 matches found in dataset at all",
        )

    most_recent = pd.Timestamp(matches_2026["date"].max())
    reference = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
    staleness_days = (reference - most_recent).days

    return QualityCheckResult(
        check_name="data_freshness",
        category="Freshness",
        passed=staleness_days <= max_staleness_days,
        detail=f"Most recent real match: {most_recent.date()} ({staleness_days} days ago)",
        affected_rows=0 if staleness_days <= max_staleness_days else 1,
    )


ALL_CHECKS = [
    check_no_nulls_in_required_fields,
    check_no_negative_scores,
    check_no_team_plays_itself,
    check_no_duplicate_matches,
    check_expected_match_count_per_tournament,
    check_dates_within_real_tournament_windows,
    check_data_freshness,
]


def run_all_quality_checks() -> pd.DataFrame:
    """Runs every registered quality check against the real combined dataset."""
    df = load_full_real_history()
    df["date"] = df["date"].astype(str)  # for safe string comparison in date-window checks

    results = [check(df) for check in ALL_CHECKS]

    out_df = pd.DataFrame([{
        "check_name": r.check_name,
        "category": r.category,
        "passed": r.passed,
        "detail": r.detail,
        "affected_rows": r.affected_rows,
    } for r in results])

    out_path = PROCESSED_DIR / "data_quality_report.csv"
    out_df.to_csv(out_path, index=False)
    return out_df


if __name__ == "__main__":
    report = run_all_quality_checks()
    passed = report["passed"].sum()
    total = len(report)

    print(f"Data Quality Report — {passed}/{total} checks passed\n")
    for _, row in report.iterrows():
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{status}] {row['category']:<14} {row['check_name']}")
        print(f"       {row['detail']}\n")
