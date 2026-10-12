"""
Step 1. Load the raw Kyiv air-raid alert file, check it, and build a daily time series.

Input:
    data/raw/airAlert_2026-10-12.json
        Original file from data.kyivcity.gov.ua (Kyiv city only). Never modified.

Outputs:
    data/processed/alerts_clean.csv
        One row per alert, including the alerts from the incomplete download day.
    data/processed/daily_counts.csv
        Number of alert STARTS per calendar day, only for complete days
        (up to and including LAST_COMPLETE_DAY). Days without alerts are 0.

About time:
    Times in the file are Kyiv local time (checked against a UTC source).
    We use them as they are: an alert belongs to the day on which it STARTED.
    We do not use durations, so the clock change (summer/winter time)
    does not affect the daily counts.

Run from the project folder:
    python src/prepare_data.py
"""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "airAlert_2026-10-12.json"
OUT_DIR = ROOT / "data" / "processed"

# The file was downloaded on 12 Oct 2026, so that day is incomplete.
DOWNLOAD_DATE = pd.Timestamp("2026-10-12")
LAST_COMPLETE_DAY = DOWNLOAD_DATE - pd.Timedelta(days=1)

KYIV_CODE = "UA80000000000093317"
EXPECTED_FIELDS = {
    "uid", "CATUTTC", "addressAdminUnitL1", "addressAdminUnitL2",
    "addressPostName", "type", "level", "cause", "dateTimeStart", "dateTimeEnd",
}
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def load_raw(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(condition: bool, message: str, failures: list[str]) -> None:
    """Print OK/FAIL for one check and remember failures."""
    print(f"  [{'OK' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)


def validate_and_clean(records: list[dict]) -> pd.DataFrame:
    print("\n=== Checks ===")
    failures: list[str] = []

    check(isinstance(records, list) and len(records) > 0,
          "file is a non-empty list of records", failures)
    check(all(set(r) == EXPECTED_FIELDS for r in records),
          "every record has exactly the expected fields", failures)

    df = pd.DataFrame(records)
    check(df["uid"].is_unique, "uid values are unique", failures)
    check((df["CATUTTC"] == KYIV_CODE).all(),
          "all records are Kyiv city (one territory code)", failures)
    check((df["type"] == "Повітряна тривога").all(),
          "all records are air-raid alerts", failures)

    # Text -> time. errors="raise" stops the script if any value has a wrong format.
    start = pd.to_datetime(df["dateTimeStart"], format=TIME_FORMAT, errors="raise")
    end = pd.to_datetime(df["dateTimeEnd"], format=TIME_FORMAT, errors="raise")

    check((end > start).all(), "every alert ends after it starts", failures)
    check(start.is_monotonic_increasing, "records are sorted by start time", failures)
    check(not start.duplicated().any(), "no two alerts share the same start time", failures)
    check((start.shift(-1) >= end).iloc[:-1].all(), "alerts do not overlap in time", failures)

    if failures:
        raise SystemExit(f"\nStopped: {len(failures)} check(s) failed: {failures}")

    return pd.DataFrame({
        "uid": df["uid"].astype(int),
        "start": start,
        "end": end,
        # In the file a missing value is the TEXT "null", not a real empty value.
        "level": df["level"].replace("null", pd.NA),
        "cause": df["cause"].replace("null", pd.NA),
    })


def print_observations(alerts: pd.DataFrame) -> None:
    """Things worth knowing that do not stop the script."""
    print("\n=== Observations ===")
    print(f"  alerts in file: {len(alerts)}")
    print(f"  first start: {alerts['start'].min()}")
    print(f"  last start:  {alerts['start'].max()}")
    crosses = (alerts["end"].dt.date > alerts["start"].dt.date).sum()
    print(f"  alerts crossing midnight (counted on their start day): {crosses}")
    has_level = alerts["level"].notna()
    print(f"  level/cause filled from: {alerts.loc[has_level, 'start'].min()} "
          f"({has_level.sum()} records)")


def build_daily_counts(alerts: pd.DataFrame) -> pd.DataFrame:
    # Calendar day on which each alert STARTED (time of day removed).
    start_day = alerts["start"].dt.normalize()
    start_day = start_day[start_day <= LAST_COMPLETE_DAY]

    counts = start_day.value_counts().sort_index()

    # Full list of days, so that days without alerts appear as 0.
    all_days = pd.date_range(start_day.min(), LAST_COMPLETE_DAY, freq="D")
    daily = counts.reindex(all_days, fill_value=0)
    daily.index.name = "date"
    return daily.rename("n_starts").reset_index()


def main() -> None:
    print(f"Reading {RAW_FILE.relative_to(ROOT)}")
    alerts = validate_and_clean(load_raw(RAW_FILE))
    print_observations(alerts)

    daily = build_daily_counts(alerts)
    n_incomplete = (alerts["start"].dt.normalize() > LAST_COMPLETE_DAY).sum()

    print("\n=== Daily series ===")
    print(f"  days: {daily['date'].min().date()} .. {daily['date'].max().date()} "
          f"({len(daily)} days, {(daily['n_starts'] == 0).sum()} days with 0 alerts)")
    print(f"  alerts after {LAST_COMPLETE_DAY.date()} (incomplete day, not in daily series): "
          f"{n_incomplete}")
    adds_up = daily["n_starts"].sum() + n_incomplete == len(alerts)
    print(f"  [{'OK' if adds_up else 'FAIL'}] daily counts + excluded alerts = all alerts")
    if not adds_up:
        raise SystemExit("Stopped: daily counts do not add up.")

    print("\n  last 14 complete days:")
    print(daily.tail(14).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(OUT_DIR / "alerts_clean.csv", index=False)
    daily.to_csv(OUT_DIR / "daily_counts.csv", index=False)
    print(f"\nSaved: {(OUT_DIR / 'alerts_clean.csv').relative_to(ROOT)}, "
          f"{(OUT_DIR / 'daily_counts.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()