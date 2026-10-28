"""Chronological baseline model and persistence benchmark for weekly sales data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


@dataclass
class BaselineResult:
    """Predictions and metrics from one chronological train/holdout evaluation."""

    predictions: pd.DataFrame
    metrics: dict[str, float]
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    model: LinearRegression


def evaluate_baseline(
    data: pd.DataFrame,
    *,
    date_column: str,
    target_column: str,
    feature_columns: list[str],
    holdout_fraction: float = 0.2,
) -> BaselineResult:
    """Fit a linear regression on earlier rows and compare it with last-period sales.

    The persistence benchmark predicts each holdout row from the prior observed
    period's actual target. Current-period features must be available when a
    forecast is made; this function does not create lagged or future features.
    """
    required = [date_column, target_column, *feature_columns]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")

    frame = data[required].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    if frame.isna().any().any():
        raise ValueError("Input data contains missing values.")
    if frame[date_column].duplicated().any():
        raise ValueError(f"Duplicate values in {date_column}.")
    frame = frame.sort_values(date_column).reset_index(drop=True)
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in [target_column, *feature_columns]):
        raise ValueError("Target and feature columns must be numeric.")
    if not all(np.isfinite(frame[c]).all() for c in [target_column, *feature_columns]):
        raise ValueError("Target and feature columns must contain only finite values.")

    split = int(len(frame) * (1 - holdout_fraction))
    if split < 2 or split >= len(frame):
        raise ValueError("Not enough rows for the requested chronological split.")

    train, test = frame.iloc[:split], frame.iloc[split:]
    model = LinearRegression()
    model.fit(train[feature_columns], train[target_column])
    prediction = model.predict(test[feature_columns])
    # The first holdout prediction uses the last training observation; subsequent
    # predictions use the immediately preceding observed period.
    persistence = frame[target_column].iloc[split - 1 : -1].to_numpy()
    actual = test[target_column].to_numpy()

    predictions = pd.DataFrame(
        {
            date_column: test[date_column].to_numpy(),
            "actual": actual,
            "baseline_prediction": prediction,
            "previous_period_prediction": persistence,
        }
    )
    metrics = {
        "baseline_mae": float(mean_absolute_error(actual, prediction)),
        "previous_period_mae": float(mean_absolute_error(actual, persistence)),
    }
    return BaselineResult(
        predictions=predictions,
        metrics=metrics,
        train_end=train[date_column].iloc[-1],
        test_start=test[date_column].iloc[0],
        model=model,
    )


def main() -> None:
    """Evaluate the project's example dataset and print holdout results."""
    from pathlib import Path

    path = Path("data/processed/mmm_cleaned.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    data = pd.read_csv(path)
    result = evaluate_baseline(
        data,
        date_column="date_week",
        target_column="y",
        feature_columns=["x1", "x2", "event_1", "event_2", "t"],
    )
    print(f"Training through: {result.train_end.date()}")
    print(f"Holdout starts:   {result.test_start.date()}")
    print(f"Holdout periods:  {len(result.predictions)}")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.2f}")
    output = Path("data/processed/baseline_holdout_predictions.csv")
    result.predictions.to_csv(output, index=False)
    print(f"Saved predictions: {output}")
    figure = px.line(
        result.predictions,
        x="date_week",
        y=["actual", "baseline_prediction", "previous_period_prediction"],
        labels={"value": "Sales (y)", "date_week": "Week", "variable": "Series"},
        title="Holdout sales: actual vs baseline predictions",
    )
    chart_path = Path("data/processed/baseline_holdout.html")
    figure.write_html(chart_path)
    print(f"Saved holdout chart: {chart_path}")


if __name__ == "__main__":
    main()
