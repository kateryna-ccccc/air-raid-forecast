"""
Check whether the RuntimeWarnings seen on macOS changed any model result.
VALIDATION period only; the test period is not used.

Checks, for each of the 8 model variants x 28 validation days (224 fits):
  1. warnings: which variants and dates produce RuntimeWarnings;
  2. training finished properly: no ConvergenceWarning, iterations below the limit,
     all weights are ordinary finite numbers, the forecast is a finite number;
  3. forecast recomputed BY HAND in plain Python (no matrix library):
        exp(intercept + w1*x1 + w2*x2 + w3*x3)  -> must equal the model's forecast;
  4. "X @ weights" (the line in the warnings) recomputed in plain Python -> must match;
  5. all validation forecasts (11 methods x 28 days) must match the reference file
     tests/reference_validation_forecasts.csv, produced on a computer with no warnings.

Run from the project folder:
    python tests/diagnose_warnings.py
"""

import math
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import BASELINES  # noqa: E402
from common import VAL_END, VAL_START, load_daily, rolling_forecast  # noqa: E402
from model import (ALPHA, FEATURES, TRANSFORMS, WINDOWS,  # noqa: E402
                   build_training, method_name, poisson_forecast)

MAX_ITER = 1000          # same as model.py
REFERENCE = ROOT / "tests" / "reference_validation_forecasts.csv"
TOLERANCE = 1e-6         # allowed difference, in alerts per day


def plain_dot(row, weights) -> float:
    """w1*x1 + w2*x2 + w3*x3 with plain Python numbers."""
    return sum(float(x) * float(w) for x, w in zip(row, weights))


def is_bad(value: float, limit: float) -> bool:
    """True if the value is above the limit OR not a finite number (NaN/inf count as bad)."""
    return not math.isfinite(value) or value > limit


def check_fits(series: pd.Series) -> pd.DataFrame:
    rows = []
    for transform in TRANSFORMS:
        for window in WINDOWS:
            name = method_name(window, transform)
            for day in pd.date_range(VAL_START, VAL_END, freq="D"):
                history = series[series.index < day]
                train, next_day = build_training(history, transform, window)
                X = train[FEATURES].to_numpy(dtype=float)
                y = train["target"].to_numpy(dtype=float)
                x_next = next_day[FEATURES].to_numpy(dtype=float)

                model = PoissonRegressor(alpha=ALPHA, max_iter=MAX_ITER)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(X, y)
                    forecast = float(model.predict(x_next)[0])

                runtime = [str(w.message).split(" encountered")[0]
                           for w in caught if issubclass(w.category, RuntimeWarning)]
                weights_ok = bool(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_))
                try:
                    by_hand = math.exp(model.intercept_ + plain_dot(x_next[0], model.coef_))
                except OverflowError:
                    by_hand = math.inf
                library = X @ model.coef_ + model.intercept_
                plain = np.array([plain_dot(r, model.coef_) + model.intercept_ for r in X])

                rows.append({
                    "method": name, "date": day.date(),
                    "runtime_warnings": len(runtime),
                    "warning_types": ", ".join(sorted(set(runtime))),
                    "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
                    "n_iter": int(model.n_iter_),
                    "finite_weights": weights_ok,
                    "forecast": forecast,
                    "forecast_vs_by_hand": abs(forecast - by_hand),
                    "matmul_vs_plain": float(np.max(np.abs(library - plain))),
                })
    return pd.DataFrame(rows)


def compare_with_reference(series: pd.Series) -> float:
    """Largest difference between this computer's validation forecasts and the reference."""
    reference = pd.read_csv(REFERENCE, parse_dates=["date"])
    methods = dict(BASELINES)
    for transform in TRANSFORMS:
        for window in WINDOWS:
            methods[method_name(window, transform)] = poisson_forecast(window, transform)
    worst = 0.0
    for name, fn in methods.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # counted separately in check_fits
            mine = rolling_forecast(series, fn, VAL_START, VAL_END)
        ref = reference[reference["method"] == name].sort_values("date")
        if len(ref) != len(mine) or not (ref["date"].values == mine["date"].values).all():
            return math.inf
        diff = np.abs(mine["predicted"].to_numpy() - ref["predicted"].to_numpy())
        worst = max(worst, float(np.max(diff)) if np.isfinite(diff).all() else math.inf)
    return worst


def verdict(df: pd.DataFrame, reference_diff: float) -> list:
    """List of problems found (empty list = no problems)."""
    problems = []
    if df["convergence_warning"].any():
        problems.append(f"{int(df['convergence_warning'].sum())} fits did not converge (ConvergenceWarning)")
    if (df["n_iter"] >= MAX_ITER).any():
        problems.append("some fits hit the iteration limit")
    if not df["finite_weights"].all():
        problems.append("some weights are not finite numbers")
    if not np.isfinite(df["forecast"]).all():
        problems.append("some forecasts are not finite numbers")
    if any(is_bad(v, TOLERANCE) for v in df["forecast_vs_by_hand"]):
        problems.append("model forecast differs from the by-hand calculation")
    if any(is_bad(v, 1e-8) for v in df["matmul_vs_plain"]):
        problems.append("matrix multiplication differs from the plain calculation")
    if is_bad(reference_diff, TOLERANCE):
        problems.append(f"forecasts differ from the reference file (max diff {reference_diff})")
    return problems


def main() -> None:
    print(f"Python {platform.python_version()}, {platform.platform()}")
    print(f"numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}")

    series = load_daily()
    df = check_fits(series)
    reference_diff = compare_with_reference(series)

    summary = df.groupby("method", sort=False).agg(
        days_with_warnings=("runtime_warnings", lambda s: int((s > 0).sum())),
        not_converged=("convergence_warning", "sum"),
        max_iterations=("n_iter", "max"),
        weights_finite=("finite_weights", "all"),
        max_diff_by_hand=("forecast_vs_by_hand", "max"),
        max_diff_matmul=("matmul_vs_plain", "max"),
    )
    print(f"\n{len(df)} fits (8 variants x 28 validation days), iteration limit {MAX_ITER}\n")
    with pd.option_context("display.float_format", "{:.1e}".format, "display.width", 200):
        print(summary.to_string())

    flagged = df[df["runtime_warnings"] > 0]
    print(f"\nFits with RuntimeWarnings: {len(flagged)} of {len(df)}")
    if len(flagged):
        print("Warning types:", sorted(set(", ".join(flagged["warning_types"]).split(", "))))
        print("Dates with warnings, per variant:")
        for name, group in flagged.groupby("method", sort=False):
            print(f"  {name}: {', '.join(str(d) for d in group['date'])}")

    print(f"\nLargest difference from reference forecasts: {reference_diff:.1e} alerts")

    problems = verdict(df, reference_diff)
    if problems:
        print("\nVERDICT: PROBLEM FOUND -> send this output")
        for p in problems:
            print("  -", p)
    else:
        print(f"\nVERDICT: no problem found. {len(flagged)} fits had warnings, but every fit finished "
              f"normally and every forecast matches the by-hand calculation and the reference.")

    out = ROOT / "results" / "warnings_diagnosis.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved details: results/{out.name}")


if __name__ == "__main__":
    main()