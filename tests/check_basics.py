"""
Checks on small made-up examples where the right answer can be computed by hand.
Uses no real data and never touches the test period.

Run from the project folder:
    python tests/check_basics.py
"""

import math
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import common  # noqa: E402
from baselines import mean_of_last, same_as_yesterday  # noqa: E402
from common import bias, mae, rolling_forecast  # noqa: E402
from model import (build_training, choose_best, make_features,  # noqa: E402
                   paired_comparison, poisson_forecast)

passed = 0


def ok(name: str, condition: bool) -> None:
    global passed
    if not condition:
        raise SystemExit(f"FAIL: {name}")
    passed += 1
    print(f"PASS: {name}")


def days(values, start="2026-01-01") -> pd.Series:
    """Daily series with the given values."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


# 1. Error metrics. actual 2, 4; forecast 3, 1 -> errors +1, -3.
#    MAE = (1 + 3) / 2 = 2.  bias = (+1 - 3) / 2 = -1.
r = pd.DataFrame({"actual": [2, 4], "predicted": [3.0, 1.0]})
ok("MAE of (2,4) vs (3,1) is 2", mae(r) == 2)
ok("bias of (2,4) vs (3,1) is -1", bias(r) == -1)

# 2. Baselines on the series 1, 2, ..., 49 (value = day number).
#    yesterday = 49; mean of last 7 = mean(43..49) = 46; mean of last 28 = mean(22..49) = 35.5
h = days(range(1, 50))
ok("yesterday = 49", same_as_yesterday(h) == 49)
ok("mean of last 7 = 46", mean_of_last(7)(h) == 46)
ok("mean of last 28 = 35.5", mean_of_last(28)(h) == 35.5)

# 3. Hints for the model on the same series.
#    Row for day 30 (value 30) must use days 1..29 only:
#    lag_1 = 29, mean_7 = mean(23..29) = 26, mean_28 = mean(2..29) = 15.5
f = make_features(h, "raw")
row = f.iloc[29]
ok("day 30 hints = (29, 26, 15.5)",
   (row["lag_1"], row["mean_7"], row["mean_28"]) == (29, 26, 15.5))
#    Extra last row = hints for day 50 (the forecast day): (49, 46, 35.5)
last = f.iloc[-1]
ok("forecast-day hints = (49, 46, 35.5)",
   (last["lag_1"], last["mean_7"], last["mean_28"]) == (49, 46, 35.5))
ok("log hints = log(1 + raw)",
   math.isclose(make_features(h, "log").iloc[-1]["lag_1"], math.log(50)))

# 4. Training examples: with 49 days and MIN_HISTORY = 28 the first usable
#    example is day 29 (its mean_28 uses days 1..28), so there are 21 examples
#    (days 29..49); the target of each example is that day's own value.
train, next_day = build_training(h, "raw", window=None)
ok("21 training examples, days 29..49", len(train) == 21 and train["target"].iloc[0] == 29)
ok("no missing hints in training", not train.isna().any().any())
ok("last training example is the day before the forecast day",
   train.index[-1] == h.index[-1] and next_day.index[0] == h.index[-1] + pd.Timedelta(days=1))
train10, _ = build_training(h, "raw", window=10)
ok("window=10 keeps the 10 most recent examples (days 40..49)",
   len(train10) == 10 and train10["target"].tolist() == list(range(40, 50)))

# 5. Model on a constant series: every day 5 alerts -> forecast must be about 5.
const = days([5] * 60)
for transform in ["raw", "log"]:
    pred = poisson_forecast(None, transform)(const)
    ok(f"constant 5 -> forecast about 5 ({transform}: {pred:.3f})", abs(pred - 5) < 0.05)

# 6. Leakage: forecasts for days 40..45 must not change if all LATER days change.
base = days([i % 7 for i in range(80)])
spiked = base.copy()
spiked[spiked.index > base.index[44]] = 1000          # days 46.. become 1000
start, end = base.index[39], base.index[44]
a = rolling_forecast(base, poisson_forecast(28, "log"), start, end)
b = rolling_forecast(spiked, poisson_forecast(28, "log"), start, end)
ok("forecasts do not change when future days change", a["predicted"].equals(b["predicted"]))

seen = []
rolling_forecast(base, lambda hist: seen.append(hist.index.max()) or 0.0, start, end)
ok("each forecast sees history up to the day before only",
   seen == list(pd.date_range(start - pd.Timedelta(days=1), end - pd.Timedelta(days=1))))

# 7. Test period is blocked unless explicitly allowed.
try:
    rolling_forecast(days([1] * 400, start="2025-10-01"), same_as_yesterday,
                     common.TEST_START, common.TEST_START)
    blocked = False
except SystemExit:
    blocked = True
ok("rolling_forecast refuses the test period by default", blocked)

# 8. load_daily stops if a day is missing.
with tempfile.TemporaryDirectory() as tmp:
    bad = Path(tmp) / "daily.csv"
    pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-04"],
                  "n_starts": [1, 2, 3]}).to_csv(bad, index=False)
    original = common.DAILY_FILE
    common.DAILY_FILE = bad
    try:
        common.load_daily()
        stopped = False
    except SystemExit:
        stopped = True
    finally:
        common.DAILY_FILE = original
ok("load_daily stops when a day is missing", stopped)

# 9. Choosing the best and comparing day by day.
summary = pd.DataFrame({"MAE": [3.0, 2.5, 4.0]}, index=["a", "b", "c"])
ok("best of a, b, c by MAE is b", choose_best(summary, ["a", "b", "c"]) == "b")
ok("best among a, c only is a", choose_best(summary, ["a", "c"]) == "a")
#    errors of x: 1, 1, 4; errors of y: 2, 2, 2 -> differences -1, -1, +2 -> mean 0; x better on 2 days
x = pd.DataFrame({"actual": [5, 5, 5], "predicted": [4.0, 6.0, 1.0]})
y = pd.DataFrame({"actual": [5, 5, 5], "predicted": [3.0, 7.0, 3.0]})
diff, std_err, better = paired_comparison(x, y)
ok("paired comparison: mean 0, better on 2 of 3 days", diff == 0 and better == 2)

print(f"\nAll {passed} checks passed.")