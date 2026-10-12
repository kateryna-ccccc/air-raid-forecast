"""
Step 4. FINAL TEST: run ONCE on the test period with settings fixed on validation.

Fixed before the test (chosen on validation by lowest MAE, see model.py):
    main method:   poisson_log_all  (log hints, whole history, re-trained every day)
    comparison:    yesterday        (same as yesterday)
Nothing is chosen or tuned here.

What it checks:
    - one-day-ahead forecasts for every test day, each made at 00:00 using
      only earlier days (rolling_forecast in common.py);
    - MAE and bias of both methods, and a day-by-day comparison;
    - training of the main model on EVERY test day: RuntimeWarnings,
      ConvergenceWarning, iteration limit, finite weights, finite forecast,
      and the forecast recomputed by hand. Problems are printed, not hidden.

Run from the project folder (once):
    python src/final_test.py
If results already exist the script stops; to run again on purpose:
    python src/final_test.py --again
"""

import math
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from baselines import same_as_yesterday
from common import (RESULTS_DIR, TEST_END, TEST_START, bias, load_daily, mae,
                    rolling_forecast)
from model import ALPHA, FEATURES, build_training, paired_comparison

MAIN_METHOD = "poisson_log_all"
MAIN_TRANSFORM, MAIN_WINDOW = "log", None
COMPARISON = "yesterday"
MAX_ITER = 1000

SUMMARY_FILE = RESULTS_DIR / "test_summary.csv"


def checked_poisson_forecast(diagnostics: list):
    """poisson_log_all forecast that also records a training check for every day."""
    def forecast(history: pd.Series) -> float:
        train, next_day = build_training(history, MAIN_TRANSFORM, MAIN_WINDOW)
        x_next = next_day[FEATURES].to_numpy(dtype=float)

        model = PoissonRegressor(alpha=ALPHA, max_iter=MAX_ITER)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train[FEATURES].to_numpy(dtype=float), train["target"].to_numpy(dtype=float))
            predicted = float(model.predict(x_next)[0])

        try:
            by_hand = math.exp(model.intercept_ + sum(float(x) * float(w)
                                                      for x, w in zip(x_next[0], model.coef_)))
        except OverflowError:
            by_hand = math.inf

        diagnostics.append({
            "date": history.index[-1] + pd.Timedelta(days=1),
            "runtime_warnings": sum(issubclass(w.category, RuntimeWarning) for w in caught),
            "not_converged": any(issubclass(w.category, ConvergenceWarning) for w in caught),
            "n_iter": int(model.n_iter_),
            "finite_weights": bool(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_)),
            "forecast": predicted,
            "diff_vs_by_hand": abs(predicted - by_hand),
        })
        return predicted
    return forecast


def training_problems(diag: pd.DataFrame) -> list:
    """List of training problems; NaN/inf always count as a problem."""
    problems = []
    if diag["not_converged"].any():
        problems.append(f"{int(diag['not_converged'].sum())} days: training did not converge")
    if (diag["n_iter"] >= MAX_ITER).any():
        problems.append("some days hit the iteration limit")
    if not diag["finite_weights"].all():
        problems.append("some weights are not finite numbers")
    if not np.isfinite(diag["forecast"]).all():
        problems.append("some forecasts are not finite numbers")
    if not all(math.isfinite(d) and d <= 1e-6 * max(1.0, f)
               for d, f in zip(diag["diff_vs_by_hand"], diag["forecast"])):
        problems.append("model forecast differs from the by-hand calculation")
    return problems


def evaluate(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, allow_test: bool):
    diagnostics = []
    main_results = rolling_forecast(series, checked_poisson_forecast(diagnostics),
                                    start, end, allow_test=allow_test)
    comparison_results = rolling_forecast(series, same_as_yesterday, start, end,
                                          allow_test=allow_test)

    summary = pd.DataFrame([
        {"method": MAIN_METHOD, "role": "main (fixed on validation)",
         "MAE": mae(main_results), "bias": bias(main_results)},
        {"method": COMPARISON, "role": "comparison",
         "MAE": mae(comparison_results), "bias": bias(comparison_results)},
    ]).set_index("method")

    forecasts = pd.DataFrame({
        "date": main_results["date"],
        "actual": main_results["actual"],
        MAIN_METHOD: main_results["predicted"],
        COMPARISON: comparison_results["predicted"],
    })
    return summary, forecasts, pd.DataFrame(diagnostics), main_results, comparison_results


def save_plot(forecasts: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(forecasts["date"], forecasts["actual"], color="#c9ced6", label="actual")
    ax.plot(forecasts["date"], forecasts[MAIN_METHOD], color="#1f5fbf", marker="o",
            markersize=3, label=f"{MAIN_METHOD} (main)")
    ax.plot(forecasts["date"], forecasts[COMPARISON], color="#d9822b", linestyle="--",
            label=f"{COMPARISON} (comparison)")
    ax.set_title("Kyiv: alert starts per day, test period (forecast made at 00:00 each day)")
    ax.set_ylabel("alert starts per day")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "test_forecast_plot.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if SUMMARY_FILE.exists() and "--again" not in sys.argv:
        raise SystemExit(f"{SUMMARY_FILE.name} already exists: the test was already run. "
                         "Use --again only if you really need to re-run it.")

    series = load_daily()
    print(f"FINAL TEST period: {TEST_START.date()} .. {TEST_END.date()} "
          f"(settings fixed on validation, nothing tuned here)")

    summary, forecasts, diag, main_res, comp_res = evaluate(series, TEST_START, TEST_END,
                                                            allow_test=True)

    print("\n=== Training check of the main model (every test day) ===")
    print(f"  days with RuntimeWarnings: {int((diag['runtime_warnings'] > 0).sum())} of {len(diag)}")
    print(f"  max iterations: {diag['n_iter'].max()} (limit {MAX_ITER})")
    problems = training_problems(diag)
    if problems:
        print("  TRAINING PROBLEMS FOUND - results below may be wrong:")
        for p in problems:
            print("   -", p)
    else:
        print("  no training problems: every fit finished, all numbers finite, "
              "forecasts match the by-hand calculation")

    print("\n=== Test results (MAE: lower is better; bias < 0: too low on average) ===")
    print(summary.round(2).to_string())
    diff, std_err, better = paired_comparison(main_res, comp_res)
    print(f"\n{MAIN_METHOD} minus {COMPARISON}, absolute error per day: {diff:+.2f} "
          f"(rough +/- {std_err:.2f}); {MAIN_METHOD} better on {better} of {len(main_res)} days")
    print(f"Average actual alerts per day in test: {forecasts['actual'].mean():.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.round(4).to_csv(SUMMARY_FILE)
    forecasts.to_csv(RESULTS_DIR / "test_forecasts.csv", index=False)
    diag.to_csv(RESULTS_DIR / "test_training_check.csv", index=False)
    save_plot(forecasts)
    print("\nSaved: results/test_summary.csv, results/test_forecasts.csv, "
          "results/test_training_check.csv, results/test_forecast_plot.png")


if __name__ == "__main__":
    main()