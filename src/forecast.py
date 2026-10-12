"""
Step 5. Reproduced historical forecast for 12 October 2026, Kyiv city.

What the method would have said at 00:00 on 12 Oct 2026, for the whole day:
    - place: Kyiv city only (not Kyiv oblast);
    - target: number of air-raid alerts that START during the day (Kyiv local time);
    - data: complete days up to and including 11 Oct (daily_counts.csv,
      from the Kyiv city open data portal, data.kyivcity.gov.ua);
    - main method: poisson_log_all (fixed on validation, not changed);
    - comparison: yesterday.
"Reproduced" = we run it later, but with the same 00:00 cut-off.

The raw file also has some alerts from 12 Oct. Data for that day are incomplete
(the file was downloaded during the day). They are NOT used for the forecast and
NOT compared with it; the script only reports that they are incomplete.

Run from the project folder:
    python src/forecast.py
"""

import pandas as pd

from baselines import same_as_yesterday
from common import RESULTS_DIR, ROOT, load_daily
from final_test import (COMPARISON, MAIN_METHOD, checked_poisson_forecast,
                        training_problems)

FORECAST_DAY = pd.Timestamp("2026-10-12")
ALERTS_FILE = ROOT / "data" / "processed" / "alerts_clean.csv"


def print_test_context() -> None:
    """Average error of the main method on the test period, if available."""
    summary_file = RESULTS_DIR / "test_summary.csv"
    if not summary_file.exists():
        return
    test_mae = pd.read_csv(summary_file, index_col="method").loc[MAIN_METHOD, "MAE"]

    print("\nFor context (not an interval for this forecast):")
    print(f"  on the test period (14 Sep - 11 Oct) the main method was off by "
          f"{test_mae:.2f} alerts per day on average (MAE).")
    print("  This is not a guaranteed range for 12 Oct.")


def main() -> None:
    series = load_daily()

    # Cut-off 00:00 of FORECAST_DAY: only days before it are used.
    history = series[series.index < FORECAST_DAY]
    if history.index.max() != FORECAST_DAY - pd.Timedelta(days=1):
        raise SystemExit("History must end on the day before the forecast day.")

    diagnostics = []
    main_forecast = checked_poisson_forecast(diagnostics)(history)
    comparison_forecast = same_as_yesterday(history)
    problems = training_problems(pd.DataFrame(diagnostics))

    print("Kyiv city (not Kyiv oblast): air-raid alerts, data from data.kyivcity.gov.ua")
    print(f"Forecast day: {FORECAST_DAY.date()} (whole day, Kyiv local time)")
    print(f"Made as if at 00:00 on {FORECAST_DAY.date()}, "
          f"using data up to and including {history.index.max().date()}")
    print("Counted: alerts that START during the day.")
    print(f"\n  {MAIN_METHOD} (main):       {main_forecast:.1f} expected alert starts")
    print(f"  {COMPARISON} (comparison):  {comparison_forecast:.0f}")
    print(f"  last 7 days before the forecast: {', '.join(str(int(v)) for v in history.iloc[-7:])}")
    print("  (the main forecast is an expected average, so it can be a fraction)")

    if problems:
        print("\n  TRAINING PROBLEMS FOUND - this forecast may be wrong:")
        for p in problems:
            print("   -", p)
    else:
        print(f"\n  training check: no problems. RuntimeWarnings from the numerical library: "
              f"{diagnostics[0]['runtime_warnings']} (fit and forecast verified by hand)")

    print_test_context()

    alerts = pd.read_csv(ALERTS_FILE, parse_dates=["start"])
    n_in_file = int((alerts["start"].dt.normalize() == FORECAST_DAY).sum())
    print(f"\nActual alerts on {FORECAST_DAY.date()}: data incomplete "
          f"(file downloaded during the day; {n_in_file} alert(s) recorded in it). "
          f"Not a full-day actual, not compared with the forecast.")

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([{
        "place": "Kyiv city",
        "forecast_day": FORECAST_DAY.date(),
        "made_as_if_at": f"{FORECAST_DAY.date()} 00:00",
        "data_up_to_and_including": history.index.max().date(),
        "target": "alert starts per day",
        MAIN_METHOD: round(main_forecast, 2),
        COMPARISON: comparison_forecast,
        "actual_for_forecast_day": "incomplete data",
        "training_problems": "; ".join(problems) or "none",
    }]).to_csv(RESULTS_DIR / f"forecast_{FORECAST_DAY.date()}.csv", index=False)
    print(f"\nSaved: results/forecast_{FORECAST_DAY.date()}.csv")


if __name__ == "__main__":
    main()