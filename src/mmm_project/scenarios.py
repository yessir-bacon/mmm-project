"""Budget allocation scenarios using a fitted PyMC-Marketing MMM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


DATE_COLUMN = "date_week"
CHANNEL_COLUMNS = ("x1", "x2")
CONTROL_COLUMNS = ("event_1", "event_2", "trend_scaled")


@dataclass
class ScenarioResult:
    """Posterior forecasts for a proposed channel mix and its historical mix."""

    predictions: pd.DataFrame
    summary: pd.DataFrame
    proposed_allocations: dict[str, float]
    baseline_allocations: dict[str, float]
    total_budget: float
    horizon_weeks: int
    warnings: list[str]


def validate_allocation(
    total_budget: float,
    allocations: Mapping[str, float],
    *,
    channel_columns: tuple[str, ...] = CHANNEL_COLUMNS,
) -> dict[str, float]:
    """Validate channel budgets and return them as finite float values."""
    try:
        budget = float(total_budget)
    except (TypeError, ValueError) as exc:
        raise ValueError("total_budget must be a finite positive number.") from exc
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("total_budget must be a finite positive number.")
    if not isinstance(allocations, Mapping):
        raise ValueError("allocations must map every channel to a budget amount.")

    expected = set(channel_columns)
    received = set(allocations)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise ValueError(f"Allocation channels do not match. Missing: {missing}; extra: {extra}.")

    normalized: dict[str, float] = {}
    for channel in channel_columns:
        try:
            amount = float(allocations[channel])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Allocation for {channel} must be a finite number.") from exc
        if not np.isfinite(amount) or amount < 0:
            raise ValueError(f"Allocation for {channel} must be finite and nonnegative.")
        normalized[channel] = amount

    allocated_total = sum(normalized.values())
    if not np.isclose(allocated_total, budget, rtol=1e-9, atol=1e-8):
        raise ValueError(
            f"Channel allocations sum to {allocated_total:g}, but total_budget is {budget:g}."
        )
    return normalized


def _prepare_historical_data(data: pd.DataFrame) -> pd.DataFrame:
    required = [DATE_COLUMN, *CHANNEL_COLUMNS]
    if "t" in data.columns:
        required.append("t")
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Historical data is missing required columns: {missing}")
    frame = data[required].copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    for channel in CHANNEL_COLUMNS:
        frame[channel] = pd.to_numeric(frame[channel], errors="raise")
        if not np.isfinite(frame[channel]).all():
            raise ValueError(f"Historical spend for {channel} must be finite.")
        if frame[channel].lt(0).any():
            raise ValueError(f"Historical spend for {channel} must be nonnegative.")
    if "t" in frame.columns:
        frame["t"] = pd.to_numeric(frame["t"], errors="raise")
        if not np.isfinite(frame["t"]).all():
            raise ValueError("Historical trend 't' must contain only finite values.")
    if frame[DATE_COLUMN].isna().any() or frame[DATE_COLUMN].duplicated().any():
        raise ValueError("Historical dates must be present and unique.")
    return frame.sort_values(DATE_COLUMN).reset_index(drop=True)


def _prepare_context(
    context: pd.DataFrame,
    historical: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare known dates and controls, then replace media with scenario spend."""
    needed = [DATE_COLUMN, "event_1", "event_2"]
    missing = sorted(set(needed) - set(context.columns))
    if "trend_scaled" not in context.columns and "t" not in context.columns:
        missing.append("trend_scaled (or t)")
    if missing:
        raise ValueError(f"Scenario context is missing required columns: {sorted(set(missing))}")

    frame = context.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    if frame[DATE_COLUMN].isna().any() or frame[DATE_COLUMN].duplicated().any():
        raise ValueError("Scenario context dates must be present and unique.")
    if "trend_scaled" not in frame:
        if "t" not in historical.columns:
            raise ValueError(
                "Historical data must include 't' when context does not include 'trend_scaled'."
            )
        trend = pd.to_numeric(frame["t"], errors="raise")
        historical_trend = pd.to_numeric(historical["t"], errors="raise")
        combined_trend = np.concatenate([historical_trend.to_numpy(), trend.to_numpy()])
        trend_min = float(combined_trend.min())
        trend_range = float(combined_trend.max() - trend_min)
        if not np.isfinite(trend_range) or trend_range <= 0:
            raise ValueError("Historical trend 't' must vary to prepare scenario controls.")
        frame["trend_scaled"] = (trend - trend_min) / trend_range

    for column in ("event_1", "event_2", "trend_scaled"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"Scenario control {column} must contain finite values.")
    frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    return frame[[DATE_COLUMN, "event_1", "event_2", "trend_scaled"]]


def _posterior_draws(model: object, features: pd.DataFrame, *, random_seed: int) -> np.ndarray:
    """Sample target predictions in original sales units, shaped (draw, week)."""
    posterior_predictive = model.sample_posterior_predictive(
        features,
        extend_idata=False,
        combined=True,
        include_last_observations=True,
        random_seed=random_seed,
    )
    target_name = getattr(model, "output_var", "y")
    if target_name not in posterior_predictive.data_vars:
        candidates = list(posterior_predictive.data_vars)
        if len(candidates) != 1:
            raise RuntimeError(
                "Could not identify target predictions in PyMC-Marketing output: "
                f"{candidates}"
            )
        target_name = candidates[0]
    target = posterior_predictive[target_name]
    values = np.asarray(target.values)
    date_axes = [index for index, name in enumerate(target.dims) if name in ("date", DATE_COLUMN)]
    if date_axes:
        date_axis = date_axes[0]
    else:
        candidates = [i for i, size in enumerate(values.shape) if size == len(features)]
        if not candidates:
            raise RuntimeError(
                f"Prediction output has shape {values.shape}; expected {len(features)} weeks."
            )
        date_axis = candidates[-1]
    values = np.moveaxis(values, date_axis, -1).reshape(-1, len(features))
    scale = float(np.asarray(model.get_scales_as_xarray()["target_scale"]).squeeze())
    return values * scale


def _summarize_draws(draws: np.ndarray) -> dict[str, float]:
    totals = draws.sum(axis=1)
    return {
        "predicted_total_median": float(np.median(totals)),
        "predicted_total_lower_90": float(np.quantile(totals, 0.05)),
        "predicted_total_upper_90": float(np.quantile(totals, 0.95)),
        "predicted_weekly_mean": float(draws.mean()),
    }


def simulate_budget_scenario(
    model: object,
    historical_data: pd.DataFrame,
    context: pd.DataFrame,
    *,
    total_budget: float,
    allocations: Mapping[str, float],
    range_tolerance: float = 0.1,
) -> ScenarioResult:
    """Compare a proposed horizon budget with a historical-share baseline.

    ``total_budget`` and ``allocations`` use total spend over the full scenario
    horizon, measured in the same units as the input channel columns. Each
    channel's amount is spread evenly over the context weeks. Dates and known
    controls in ``context`` stay fixed; `context` should follow the fitted
    model's training period so PyMC-Marketing can carry prior media forward.
    """
    if not np.isfinite(range_tolerance) or range_tolerance < 0:
        raise ValueError("range_tolerance must be finite and nonnegative.")
    proposed = validate_allocation(total_budget, allocations)
    historical = _prepare_historical_data(historical_data)
    scenario_context = _prepare_context(context, historical_data)
    horizon = len(scenario_context)
    if horizon == 0:
        raise ValueError("Scenario context must contain at least one week.")

    historical_totals = historical[list(CHANNEL_COLUMNS)].sum()
    historical_total = float(historical_totals.sum())
    if not np.isfinite(historical_total) or historical_total <= 0:
        raise ValueError("Historical channel spend must have a positive total.")
    baseline = {
        channel: float(total_budget * historical_totals[channel] / historical_total)
        for channel in CHANNEL_COLUMNS
    }
    # Correct the final floating-point remainder so the baseline also totals
    # exactly the requested budget.
    baseline[CHANNEL_COLUMNS[-1]] += float(total_budget - sum(baseline.values()))

    warnings: list[str] = []
    for channel in CHANNEL_COLUMNS:
        weekly_amount = proposed[channel] / horizon
        observed_min = float(historical[channel].min())
        observed_max = float(historical[channel].max())
        tolerance = (observed_max - observed_min) * range_tolerance
        if weekly_amount < observed_min - tolerance or weekly_amount > observed_max + tolerance:
            warnings.append(
                f"{channel} scenario spend per week ({weekly_amount:.4g}) is well outside "
                f"the historical range ({observed_min:.4g} to {observed_max:.4g})."
            )

    def build_features(channel_budgets: Mapping[str, float]) -> pd.DataFrame:
        features = scenario_context.copy()
        for channel in CHANNEL_COLUMNS:
            features[channel] = channel_budgets[channel] / horizon
        return features[[DATE_COLUMN, *CHANNEL_COLUMNS, *CONTROL_COLUMNS]]

    proposed_draws = _posterior_draws(
        model,
        build_features(proposed),
        random_seed=42,
    )
    baseline_draws = _posterior_draws(
        model,
        build_features(baseline),
        random_seed=42,
    )
    if proposed_draws.shape[1] != horizon or baseline_draws.shape[1] != horizon:
        raise RuntimeError("Model predictions do not align with scenario weeks.")

    difference_totals = proposed_draws.sum(axis=1) - baseline_draws.sum(axis=1)
    predictions = pd.DataFrame(
        {
            DATE_COLUMN: scenario_context[DATE_COLUMN],
            "proposed_prediction": proposed_draws.mean(axis=0),
            "proposed_lower_90": np.quantile(proposed_draws, 0.05, axis=0),
            "proposed_upper_90": np.quantile(proposed_draws, 0.95, axis=0),
            "baseline_prediction": baseline_draws.mean(axis=0),
            "baseline_lower_90": np.quantile(baseline_draws, 0.05, axis=0),
            "baseline_upper_90": np.quantile(baseline_draws, 0.95, axis=0),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "scenario": "proposed",
                "total_budget": float(total_budget),
                **_summarize_draws(proposed_draws),
            },
            {
                "scenario": "historical_share_baseline",
                "total_budget": float(total_budget),
                **_summarize_draws(baseline_draws),
            },
            {
                "scenario": "proposed_minus_baseline",
                "total_budget": float(total_budget),
                "predicted_total_median": float(np.median(difference_totals)),
                "predicted_total_lower_90": float(np.quantile(difference_totals, 0.05)),
                "predicted_total_upper_90": float(np.quantile(difference_totals, 0.95)),
                "predicted_weekly_mean": float(difference_totals.mean() / horizon),
            },
        ]
    )
    return ScenarioResult(
        predictions=predictions,
        summary=summary,
        proposed_allocations=proposed,
        baseline_allocations=baseline,
        total_budget=float(total_budget),
        horizon_weeks=horizon,
        warnings=warnings,
    )
