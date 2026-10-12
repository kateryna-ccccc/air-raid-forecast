"""
Shared settings and helpers for all forecasting steps.

Everything that defines "what is past and what is future" lives here,
so every script follows the same rules.
"""

from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY_FILE = ROOT / "data" / "processed" / "daily_counts.csv"
RESULTS_DIR = ROOT / "results"

# Three periods, in time order. Nothing is shuffled.
# - train: only for learning (and as history for the first forecasts)
# - validation: here we compare methods and choose settings
# - test: used ONCE at the end, with settings already fixed
TRAIN_END = pd.Timestamp("2026-08-16")
VAL_START, VAL_END = pd.Timestamp("2026-08-17"), pd.Timestamp("2026-09-13")
TEST_START, TEST_END = pd.Timestamp("2026-09-14"), pd.Timestamp("2026-10-11")

# A forecast function gets the history (all days BEFORE the forecast day)
# and returns the expected number of alert starts for the next day.
ForecastFn = Callable[[pd.Series], float]


def load_daily() -> pd.Series:
    """Daily number of alert starts, indexed by date. Checks that no day is missing."""
    if not DAILY_FILE.exists():
        raise SystemExit(f"{DAILY_FILE} not found. Run: python src/prepare_data.py")
    df = pd.read_csv(DAILY_FILE, parse_dates=["date"])
    series = df.set_index("date")["n_starts"]

    expected_days = pd.date_range(series.index.min(), series.index.max(), freq="D")
    if not series.index.equals(expected_days):
        raise SystemExit("Daily series has missing or repeated days.")
    return series


def rolling_forecast(series: pd.Series, forecast_fn: ForecastFn,
                     start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    One-day-ahead forecasts for every day from start to end.

    For each day D the forecast is made at 00:00 of D, so forecast_fn sees
    only days strictly before D. This is the main protection against data leakage.
    """
    rows = []
    for day in pd.date_range(start, end, freq="D"):
        history = series[series.index < day]

        # Leakage guard: history must end exactly on the day before D.
        if history.index.max() != day - pd.Timedelta(days=1):
            raise SystemExit(f"Leakage guard failed for {day.date()}")

        predicted = float(forecast_fn(history.copy()))
        rows.append({"date": day, "actual": int(series[day]), "predicted": predicted})
    return pd.DataFrame(rows)


def mae(results: pd.DataFrame) -> float:
    """Mean absolute error: on average, by how many alerts the forecast is off."""
    return float((results["actual"] - results["predicted"]).abs().mean())


def bias(results: pd.DataFrame) -> float:
    """Mean of (predicted - actual). Negative = the forecast is too low on average."""
    return float((results["predicted"] - results["actual"]).mean())