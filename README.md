# Dataset Overview: mmm_example.csv

* **Data Source:** [mmm_example.csv](https://githubusercontent.com)
* **Shape:** 179 rows, 8 columns

## Column Definitions
* **Date columns:** `date_week` and `dayofyear`
* **Sales column:** `y`
* **Channel media spend columns:** `x1` and `x2`
* **Event columns:** `event_1` and `event_2`
* **Linear trend column:** `t`

## Columns and Data Types
```text
date_week        str
y            float64
x1           float64
x2           float64
event_1      float64
event_2      float64
dayofyear      int64
t              int64
dtype: object
```

## First 5 Rows

| Index | date_week | y | x1 | x2 | event_1 | event_2 | dayofyear | t |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 2018-04-02 | 3984.662237 | 0.318580 | 0.0 | 0.0 | 0.0 | 92 | 0 |
| **1** | 2018-04-09 | 3762.871794 | 0.112388 | 0.0 | 0.0 | 0.0 | 99 | 1 |
| **2** | 2018-04-16 | 4466.967388 | 0.292400 | 0.0 | 0.0 | 0.0 | 106 | 2 |
| **3** | 2018-04-23 | 3864.219373 | 0.071399 | 0.0 | 0.0 | 0.0 | 113 | 3 |
| **4** | 2018-04-30 | 4441.625278 | 0.386745 | 0.0 | 0.0 | 0.0 | 120 | 4 |

## Missing Values

| Column | Missing Count |
| :--- | :--- |
| `date_week` | 0 |
| `y` | 0 |
| `x1` | 0 |
| `x2` | 0 |
| `event_1` | 0 |
| `event_2` | 0 |
| `dayofyear` | 0 |
| `t` | 0 |

## Numeric Summary

| Metric | y | x1 | x2 | event_1 | event_2 | dayofyear | t |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **count** | 179.000000 | 179.000000 | 179.000000 | 179.000000 | 179.000000 | 179.000000 | 179.000000 |
| **mean** | 5284.947576 | 0.309082 | 0.161850 | 0.005587 | 0.005587 | 182.597765 | 89.000000 |
| **std** | 1087.276350 | 0.237638 | 0.348671 | 0.074744 | 0.074744 | 100.370019 | 51.816986 |
| **min** | 3192.879593 | 0.000929 | 0.000000 | 0.000000 | 0.000000 | 4.000000 | 0.000000 |
| **25%** | 4484.221415 | 0.154219 | 0.000000 | 0.000000 | 0.000000 | 103.000000 | 44.500000 |
| **50%** | 5163.385726 | 0.274762 | 0.000000 | 0.000000 | 0.000000 | 181.000000 | 89.000000 |
| **75%** | 5941.275944 | 0.397176 | 0.000000 | 0.000000 | 0.000000 | 262.500000 | 133.500000 |
| **max** | 8312.407544 | 0.996658 | 0.994374 | 1.000000 | 1.000000 | 365.000000 | 178.000000 |

## Baseline model (Phase 4)

The baseline evaluation uses an 80/20 chronological split. Rows are sorted by
`date_week`; the first 143 weeks train an ordinary least squares regression and
the final 36 weeks are held out. Predictors are `x1`, `x2`, `event_1`,
`event_2`, and the dataset's linear trend `t`. The model is compared with a
previous-period benchmark that predicts each week from the prior week's actual
sales. Both models are scored with mean absolute error (MAE) on the same
holdout weeks.

Run after preparing the data:

```powershell
python src/mmm_project/baseline.py
```

The script prints the split dates and MAE values and writes
`data/processed/baseline_holdout_predictions.csv` plus an interactive chart at
`data/processed/baseline_holdout.html`. The result is a predictive
baseline, not a causal estimate: spend may be correlated with omitted factors,
and the model assumes contemporaneous channel spend and event indicators are
known at prediction time.

## MMM model (Phase 5)

The MMM uses PyMC-Marketing with `x1` and `x2` as channels, geometric adstock
with an eight-week maximum lag, logistic saturation, two yearly Fourier modes,
and `event_1`, `event_2`, and a scaled `t` trend as controls. The date range is
about 3.4 years, so the seasonal terms have multiple annual cycles to learn
from, although the dataset remains short for stable channel attribution.
Controls are not automatically scaled by PyMC-Marketing; `t` is transformed
to a 0–1 range before fitting. The package applies adstock and saturation to
media and supports posterior prediction on unseen data, including carry-in from
the final training observations.

Run the model from the project root after Phase 3 preparation:

```powershell
python src/mmm_project/model.py
```

The default run uses two MCMC chains with 1,000 warmup and 1,000 posterior
draws per chain and a fixed random seed. It fits only the first 143 weeks, then
compares MMM and Phase 4 linear-regression MAE on the same final 36 weeks. It
writes local, git-ignored artifacts to `data/processed/`: holdout predictions
with 90% posterior predictive intervals, a comparison chart, model diagnostics,
channel contribution summaries, and a metrics JSON file. The model is refitted
when the script runs; its inference data is not serialized.
Review `r_hat` (values near 1 are preferred), effective sample sizes, and any
sampler divergences before interpreting channel estimates. The channel
contributions are posterior counterfactual summaries from the fitted model,
not experimental causal effects. Correlated spend, omitted drivers, and short
coverage can make channel-level estimates unstable.

## Budget simulator (Phase 6)

`simulate_budget_scenario()` in `src/mmm_project/scenarios.py` accepts a fitted
MMM, historical channel data, a future-period context, a total budget, and
allocations for `x1` and `x2`. The budget and allocations cover the entire
scenario horizon; the function spreads each channel amount evenly across the
context weeks. It compares the proposed split with an equal-total baseline that
uses the historical training-period spend shares. Dates and event/trend
controls stay fixed across both mixes. Spend amounts use the same units as the
dataset columns; this example dataset does not document a currency unit.

Example for a 36-week horizon:

```python
import sys
import pandas as pd

sys.path.insert(0, "src")

from mmm_project.model import fit_and_evaluate_mmm, prepare_model_data
from mmm_project.scenarios import simulate_budget_scenario

raw = pd.read_csv("data/processed/mmm_cleaned.csv")
prepared = prepare_model_data(raw)
split = int(len(prepared) * 0.8)
fitted = fit_and_evaluate_mmm(raw)

scenario = simulate_budget_scenario(
    fitted.model,
    historical_data=prepared.iloc[:split],
    context=prepared.iloc[split:],
    total_budget=18.0,
    allocations={"x1": 12.0, "x2": 6.0},
)
print(scenario.summary)
print(scenario.warnings)
```

The result includes weekly posterior predictions and 90% intervals, horizon
total summaries for each mix, a proposed-minus-baseline estimate, and warnings
when proposed weekly spend is more than 10% of the observed channel range
beyond its historical minimum or maximum. Allocations must be finite,
nonnegative, include exactly `x1` and `x2`, and sum to `total_budget`. The
simulator produces model-based projections with uncertainty; it does not
guarantee that changing spend will cause the displayed sales difference.
