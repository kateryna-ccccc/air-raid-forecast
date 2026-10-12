"""
Step 2. Simple baseline forecasts on the VALIDATION period.

A trained model is useful only if it beats these simple rules.
The test period is not used here.

Run from the project folder:
    python src/baselines.py
"""

import pandas as pd

from common import (RESULTS_DIR, VAL_END, VAL_START, load_daily, mae,
                    rolling_forecast)


def same_as_yesterday(history: pd.Series) -> float:
    return history.iloc[-1]


def mean_of_last(n_days: int):
    """Returns a forecast function: average of the last n_days."""
    def forecast(history: pd.Series) -> float:
        return history.iloc[-n_days:].mean()
    return forecast


BASELINES = {
    "yesterday": same_as_yesterday,
    "mean_7d": mean_of_last(7),
    "mean_28d": mean_of_last(28),
}


def main() -> None:
    series = load_daily()
    print(f"Validation period: {VAL_START.date()} .. {VAL_END.date()}")

    all_results = []
    for name, forecast_fn in BASELINES.items():
        results = rolling_forecast(series, forecast_fn, VAL_START, VAL_END)
        results["method"] = name
        all_results.append(results)

    table = pd.concat(all_results)

    # MAE for the whole validation period and for its two halves,
    # because the number of alerts jumped in late August.
    middle = VAL_START + pd.Timedelta(days=14)
    summary = pd.DataFrame({
        "MAE_all": table.groupby("method").apply(mae, include_groups=False),
        "MAE_first_2w": table[table["date"] < middle].groupby("method").apply(mae, include_groups=False),
        "MAE_last_2w": table[table["date"] >= middle].groupby("method").apply(mae, include_groups=False),
    }).loc[list(BASELINES)].round(2)

    print("\nMean absolute error (alerts per day), lower is better:")
    print(summary.to_string())

    actual = table[table["method"] == "yesterday"].set_index("date")["actual"]
    print(f"\nActual alerts per day: first 2 weeks avg {actual[actual.index < middle].mean():.2f}, "
          f"last 2 weeks avg {actual[actual.index >= middle].mean():.2f}")

    wide = table.pivot(index="date", columns="method", values="predicted")[list(BASELINES)]
    wide.insert(0, "actual", actual)
    print("\nLast 7 validation days (forecasts made at 00:00 of each day):")
    print(wide.tail(7).round(1).to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "validation_baselines.csv")
    print(f"\nSaved: results/validation_baselines.csv")


if __name__ == "__main__":
    main()