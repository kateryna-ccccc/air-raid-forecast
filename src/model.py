"""
Step 3. Trained model (Poisson regression) on the VALIDATION period.

What the model does:
    For each past day t it looks at three hints known at 00:00 of day t:
        lag_1    - number of alert starts yesterday (t-1)
        mean_7   - average over the 7 days before t
        mean_28  - average over the 28 days before t
    and learns how to combine them into the expected number of starts on day t.
    Poisson regression is a standard model for counts: the forecast is never negative.

Two versions of the hints:
    raw  - hints as they are. On validation the forecast reached 60.7 alerts
           (4 Sep, actual 7). Reason: mean_7 and mean_28 are almost the same
           (correlation 0.98), so with few examples the model picks large opposite
           weights, and the exponent inside the model turns a small step outside
           the training range into a huge forecast.
    log  - hints as log(1 + x). Smaller numbers and a power law instead of an
           exponent, so such jumps are much weaker (but not impossible).
           Added AFTER looking at validation results (validation used twice;
           noted in README).

How it is trained without looking into the future:
    The forecast for day D is made at 00:00 of D. rolling_forecast (common.py)
    gives the model only the days before D. build_training() turns that history
    into training examples (hints -> actual count) and the hints for D.
    The model is fitted and predicts D. This repeats every day.

What we compare (validation only; the test period is blocked in common.py):
    hint version (raw / log) x number of recent training examples
    (28, 56, 112 days or the whole history). Main metric: MAE; bias shown too.

Run from the project folder:
    python src/model.py
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from baselines import BASELINES
from common import (RESULTS_DIR, VAL_END, VAL_START, bias, load_daily, mae,
                    rolling_forecast)

FEATURES = ["lag_1", "mean_7", "mean_28"]
MIN_HISTORY = 28                 # a day needs 28 earlier days to compute mean_28
WINDOWS = [28, 56, 112, None]    # None = whole history
TRANSFORMS = ["raw", "log"]
ALPHA = 1e-4                     # almost no regularisation; fixed in advance, not tuned


def make_features(counts: pd.Series, transform: str) -> pd.DataFrame:
    """
    Hints for every day in `counts` PLUS one extra row for the next day (day D).
    Every row uses ONLY earlier days: shift(1) moves the series one day forward,
    so the row for day t sees t-1 and earlier.
    """
    next_day = counts.index[-1] + pd.Timedelta(days=1)
    extended = pd.concat([counts, pd.Series([np.nan], index=[next_day])])
    past = extended.shift(1)
    features = pd.DataFrame({
        "lag_1": past,
        "mean_7": past.rolling(7).mean(),
        "mean_28": past.rolling(28).mean(),
    })
    return np.log1p(features) if transform == "log" else features


def build_training(history: pd.Series, transform: str,
                   window: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From the history (days before D) build:
      train    - past days with their hints and actual count ("target")
      next_day - one row with the hints for day D
    """
    features = make_features(history, transform)
    train = features.iloc[:-1].assign(target=history.values).iloc[MIN_HISTORY:]
    if window is not None:
        train = train.iloc[-window:]
    return train, features.iloc[[-1]]


def poisson_forecast(window: Optional[int], transform: str):
    """Returns a forecast function that trains on the last `window` examples."""
    def forecast(history: pd.Series) -> float:
        train, next_day = build_training(history, transform, window)
        model = PoissonRegressor(alpha=ALPHA, max_iter=1000)
        model.fit(train[FEATURES], train["target"])
        return float(model.predict(next_day[FEATURES])[0])
    return forecast


def method_name(window: Optional[int], transform: str) -> str:
    return f"poisson_{transform}_{window}d" if window else f"poisson_{transform}_all"


def choose_best(summary: pd.DataFrame, candidates: list) -> str:
    """Name of the candidate with the lowest MAE."""
    return summary.loc[candidates, "MAE"].idxmin()


def paired_comparison(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[float, float, int]:
    """
    Day-by-day difference of absolute errors (a minus b).
    Returns: mean difference, its rough standard error, days where a was better.
    The standard error is rough: it treats days as independent.
    """
    diff = (a["actual"] - a["predicted"]).abs() - (b["actual"] - b["predicted"]).abs()
    return float(diff.mean()), float(diff.std() / np.sqrt(len(diff))), int((diff < 0).sum())


def main() -> None:
    series = load_daily()
    print(f"Validation period: {VAL_START.date()} .. {VAL_END.date()} (test period not used)")

    methods = dict(BASELINES)
    for transform in TRANSFORMS:
        for window in WINDOWS:
            methods[method_name(window, transform)] = poisson_forecast(window, transform)

    rows, results_by_method = [], {}
    for name, forecast_fn in methods.items():
        results = rolling_forecast(series, forecast_fn, VAL_START, VAL_END)
        results_by_method[name] = results
        rows.append({"method": name, "MAE": mae(results), "bias": bias(results),
                     "max_forecast": results["predicted"].max()})

    summary = pd.DataFrame(rows).set_index("method").round(2)
    print("\nValidation results (MAE: lower is better; bias < 0: too low on average):")
    print(summary.to_string())

    model_names = [m for m in methods if m not in BASELINES]
    best_model = choose_best(summary, model_names)
    best_baseline = choose_best(summary, list(BASELINES))
    diff, std_err, better_days = paired_comparison(results_by_method[best_model],
                                                   results_by_method[best_baseline])
    print(f"\nBest model by MAE:    {best_model} ({summary.loc[best_model, 'MAE']})")
    print(f"Best baseline by MAE: {best_baseline} ({summary.loc[best_baseline, 'MAE']})")
    print(f"{best_model} minus {best_baseline}, absolute error per day: "
          f"{diff:+.2f} (rough +/- {std_err:.2f}); model better on {better_days} "
          f"of {len(results_by_method[best_model])} days")
    print("Note: the best of 8 model variants was picked on the same 28 days, "
          "so its validation MAE is optimistic.")

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "validation_models.csv")
    pd.concat([r.assign(method=n) for n, r in results_by_method.items()]).to_csv(
        RESULTS_DIR / "validation_forecasts.csv", index=False)
    print("\nSaved: results/validation_models.csv, results/validation_forecasts.csv")


if __name__ == "__main__":
    main()