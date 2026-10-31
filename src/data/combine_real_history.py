"""
Combines all real World Cup tournament results from 1990-2022 into a single
chronological dataset, then chains it into the real 2026 tournament data.

Every row in every source file (wc_history_1990.csv through wc_history_2022.csv,
plus wc2026_real_results.csv) is a real match result, individually verified
against Wikipedia, RSSSF, and other primary football statistics sources.

This gives the ELO system genuine match volume (552 historical matches plus
the real 2026 results so far) to calibrate against, rather than relying on
a single FIFA-ranking snapshot as the only signal.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_FILES = [
    "wc_history_1990.csv",
    "wc_history_1994.csv",
    "wc_history_1998.csv",
    "wc_history_2002.csv",
    "wc_history_2006.csv",
    "wc_history_2010.csv",
    "wc_history_2014.csv",
    "wc_history_2018.csv",
    "wc_history_2022.csv",
]


def load_full_real_history(include_2026: bool = True) -> pd.DataFrame:
    """
    Loads and concatenates every real World Cup tournament file (1990-2022),
    optionally appending the real 2026 results played so far, sorted
    chronologically.
    """
    frames = []
    for filename in HISTORICAL_FILES:
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing expected historical file: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        frames.append(df)

    if include_2026:
        path_2026 = RAW_DIR / "wc2026_real_results.csv"
        if path_2026.exists():
            df_2026 = pd.read_csv(path_2026, parse_dates=["date"])
            # align columns: 2026 file may not have a 'year' column
            if "year" not in df_2026.columns:
                df_2026["year"] = 2026
            frames.append(df_2026)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def save_combined_history() -> pd.DataFrame:
    """Writes the combined real dataset to data/processed/ for reuse."""
    combined = load_full_real_history()
    out_path = PROCESSED_DIR / "wc_real_history_1990_2026.csv"
    combined.to_csv(out_path, index=False)
    return combined


if __name__ == "__main__":
    combined = save_combined_history()
    print(f"Combined real World Cup history: {len(combined)} matches")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"\nMatches per tournament year:")
    print(combined["year"].value_counts().sort_index())
