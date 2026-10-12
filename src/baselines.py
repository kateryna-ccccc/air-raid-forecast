"""
Step 2. Simple baseline forecasts on the VALIDATION period.

A trained model is useful only if it beats these simple rules.
The test period is not used here.

Run from the project folder:
    python src/baselines.py
"""

import pandas as pd

from common import (RESULTS_DIR, VAL_END, VAL_START, bias, load_daily, mae,
                    rolling_forecast)


def same_as_yesterday(history: pd.Series) -> float:
    return float(history.iloc[-1])


def mean_of_last(n_days: int):
    """Returns a forecast function: average of the last n_days."""
    def forecast(history: pd.Series) -> float:
        return float(history.iloc[-n_days:].mean())
    return forecast


BASELINES = {
    "yesterday": same_as_yesterday,
    "mean_7d": mean_of_last(7),
    "mean_28d": mean_of_last(28),
}


def main() -> None:
    series = load_daily()
    print(f"Validation period: {VAL_START.date()} .. {VAL_END.date()}")

    # MAE for the whole validation period and for its two halves.
    middle = VAL_START + pd.Timedelta(days=14)
    rows = []
    for name, forecast_fn in BASELINES.items():
        results = rolling_forecast(series, forecast_fn, VAL_START, VAL_END)
        first, last = results[results["date"] < middle], results[results["date"] >= middle]
        rows.append({"method": name,
                     "MAE_all": mae(results), "bias_all": bias(results),
                     "MAE_first_2w": mae(first), "MAE_last_2w": mae(last)})
        if name == "yesterday":
            actual = results.set_index("date")["actual"]

    summary = pd.DataFrame(rows).set_index("method").round(2)
    print("\nValidation (MAE: lower is better; bias < 0: too low on average):")
    print(summary.to_string())
    print(f"\nActual alerts per day: first 2 weeks avg {actual[actual.index < middle].mean():.2f}, "
          f"last 2 weeks avg {actual[actual.index >= middle].mean():.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "validation_baselines.csv")
    print("\nSaved: results/validation_baselines.csv")


if __name__ == "__main__":
    main()