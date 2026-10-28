"""Fit and evaluate a PyMC-Marketing MMM on the project's weekly example data."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error

if __package__:
    from .baseline import evaluate_baseline
else:  # Support `python src/mmm_project/model.py` from the project root.
    from baseline import evaluate_baseline


DATE_COLUMN = "date_week"
TARGET_COLUMN = "y"
CHANNEL_COLUMNS = ["x1", "x2"]
CONTROL_COLUMNS = ["event_1", "event_2", "trend_scaled"]
HOLDOUT_FRACTION = 0.2
RANDOM_SEED = 42


@dataclass
class MMMResult:
    """Fitted MMM, holdout output, evaluation metrics, and diagnostics."""

    model: object
    predictions: pd.DataFrame
    metrics: dict[str, float]
    diagnostics: pd.DataFrame
    contributions: pd.DataFrame
    train_end: pd.Timestamp
    test_start: pd.Timestamp


def prepare_model_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort, and create a scaled trend control for the MMM."""
    required = [DATE_COLUMN, TARGET_COLUMN, *CHANNEL_COLUMNS, "event_1", "event_2", "t"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = data[required].copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    if frame.isna().any().any():
        raise ValueError("Model input contains missing values.")
    if frame[DATE_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate values in {DATE_COLUMN}.")
    frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    numeric_columns = [TARGET_COLUMN, *CHANNEL_COLUMNS, "event_1", "event_2", "t"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must contain only finite values.")
    if (frame[CHANNEL_COLUMNS] < 0).any().any():
        raise ValueError("Channel spend values must be nonnegative.")
    if frame[TARGET_COLUMN].le(0).any():
        raise ValueError("Sales target values must be positive.")

    trend_range = frame["t"].max() - frame["t"].min()
    if trend_range <= 0:
        raise ValueError("Trend column 't' must vary over time.")
    # Controls are not scaled automatically by PyMC-Marketing. Keeping this
    # deterministic time trend in [0, 1] gives its prior a sensible scale.
    frame["trend_scaled"] = (frame["t"] - frame["t"].min()) / trend_range
    return frame


def _prediction_draws(model: object, X: pd.DataFrame) -> np.ndarray:
    """Return posterior predictive draws with shape (samples, dates)."""
    result = model.sample_posterior_predictive(
        X,
        extend_idata=False,
        combined=True,
        include_last_observations=True,
        random_seed=RANDOM_SEED,
    )
    target_name = getattr(model, "output_var", TARGET_COLUMN)
    if target_name not in result.data_vars:
        candidates = [name for name in result.data_vars if "date" not in name.lower()]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Could not identify target predictions in PyMC-Marketing output: "
                f"{list(result.data_vars)}"
            )
        target_name = candidates[0]

    values = np.asarray(result[target_name].values)
    date_axis_candidates = [
        axis for axis, size in enumerate(values.shape) if size == len(X)
    ]
    if not date_axis_candidates:
        raise RuntimeError(
            f"Prediction output has shape {values.shape}; expected one axis of length {len(X)}."
        )
    # The time coordinate is conventionally the final dimension. If another
    # dimension happens to have the same length, prefer the dimension name.
    dims = list(result[target_name].dims)
    date_axes = [i for i, dim in enumerate(dims) if dim in (DATE_COLUMN, "date")]
    date_axis = date_axes[0] if date_axes else date_axis_candidates[-1]
    values = np.moveaxis(values, date_axis, -1)
    return values.reshape(-1, len(X))


def _posterior_diagnostics(model: object) -> pd.DataFrame:
    """Summarize convergence statistics for available posterior parameters."""
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError("ArviZ is required to calculate MCMC diagnostics.") from exc

    posterior = model.idata.posterior
    preferred = [
        "adstock_alpha",
        "saturation_beta",
        "saturation_lam",
        "gamma_control",
        "intercept",
        "sigma",
    ]
    available = [name for name in preferred if name in posterior.data_vars]
    if not available:
        available = list(posterior.data_vars)
    return az.summary(posterior, var_names=available, round_to=4).reset_index(names="parameter")


def _mean_channel_contributions(model: object) -> pd.DataFrame:
    """Get posterior mean counterfactual contribution by channel when supported."""
    contribution_method = getattr(model, "compute_mean_contributions_over_time", None)
    if contribution_method is None:
        raise RuntimeError(
            "This PyMC-Marketing version does not expose "
            "compute_mean_contributions_over_time(); upgrade PyMC-Marketing."
        )
    contributions = contribution_method()
    if not isinstance(contributions, pd.DataFrame):
        contributions = contributions.to_dataframe().reset_index()
    return contributions


def fit_and_evaluate_mmm(
    data: pd.DataFrame,
    *,
    holdout_fraction: float = HOLDOUT_FRACTION,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    progressbar: bool = True,
) -> MMMResult:
    """Fit an adstock/saturation MMM on the earlier period and evaluate the tail.

    The final 20% is held out chronologically. The model receives only training
    targets; posterior prediction uses the holdout features with the library's
    adstock carry-in support so the first test weeks retain training media history.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if draws < 1 or tune < 0 or chains < 1:
        raise ValueError("draws and chains must be positive; tune cannot be negative.")

    try:
        # Keep PyTensor's generated C cache inside the project's ignored data
        # directory instead of writing under the user's profile.
        cache_path = (Path("data/processed") / "pytensor_cache").resolve()
        existing_flags = os.environ.get("PYTENSOR_FLAGS", "")
        cache_flags = []
        if "base_compiledir=" not in existing_flags:
            cache_flags.append(f"base_compiledir={cache_path.as_posix()}")
        if "compiledir=" not in existing_flags:
            cache_flags.append(f"compiledir={(cache_path / 'compiledir').as_posix()}")
        if cache_flags:
            os.environ["PYTENSOR_FLAGS"] = ",".join(
                flag for flag in (existing_flags, *cache_flags) if flag
            )
        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, MMM
    except ImportError as exc:
        raise ImportError(
            "PyMC-Marketing could not be imported in this Python environment. "
            "Use the project's Conda environment and confirm pymc-marketing imports."
        ) from exc

    frame = prepare_model_data(data)
    split = int(len(frame) * (1 - holdout_fraction))
    if split < 2 or split >= len(frame):
        raise ValueError("Not enough rows for the requested chronological split.")

    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()
    X_train = train[[DATE_COLUMN, *CHANNEL_COLUMNS, *CONTROL_COLUMNS]]
    y_train = train[TARGET_COLUMN]
    X_test = test[[DATE_COLUMN, *CHANNEL_COLUMNS, *CONTROL_COLUMNS]]

    mmm = MMM(
        date_column=DATE_COLUMN,
        channel_columns=CHANNEL_COLUMNS,
        target_column=TARGET_COLUMN,
        control_columns=CONTROL_COLUMNS,
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        yearly_seasonality=2,
    )
    mmm.fit(
        X_train,
        y_train,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=chains,
        target_accept=0.9,
        random_seed=RANDOM_SEED,
        progressbar=progressbar,
    )

    draws_array = _prediction_draws(mmm, X_test)
    # PyMC-Marketing's posterior predictive group is on its scaled target
    # response; convert to sales units before calculating metrics or intervals.
    target_scale = float(np.asarray(mmm.get_scales_as_xarray()["target_scale"]).squeeze())
    draws_array = draws_array * target_scale
    predictions = pd.DataFrame(
        {
            DATE_COLUMN: test[DATE_COLUMN].to_numpy(),
            "actual": test[TARGET_COLUMN].to_numpy(),
            "mmm_prediction": draws_array.mean(axis=0),
            "mmm_lower_90": np.quantile(draws_array, 0.05, axis=0),
            "mmm_upper_90": np.quantile(draws_array, 0.95, axis=0),
        }
    )

    baseline = evaluate_baseline(
        frame,
        date_column=DATE_COLUMN,
        target_column=TARGET_COLUMN,
        feature_columns=[*CHANNEL_COLUMNS, "event_1", "event_2", "t"],
        holdout_fraction=holdout_fraction,
    )
    predictions = predictions.merge(
        baseline.predictions[[DATE_COLUMN, "baseline_prediction"]],
        on=DATE_COLUMN,
        how="left",
        validate="one_to_one",
    )
    actual = predictions["actual"]
    metrics = {
        "mmm_mae": float(mean_absolute_error(actual, predictions["mmm_prediction"])),
        "baseline_mae": float(mean_absolute_error(actual, predictions["baseline_prediction"])),
        "holdout_periods": float(len(predictions)),
        "sampler_divergences": float(
            np.asarray(mmm.idata.sample_stats["diverging"].values).sum()
        ),
    }
    return MMMResult(
        model=mmm,
        predictions=predictions,
        metrics=metrics,
        diagnostics=_posterior_diagnostics(mmm),
        contributions=_mean_channel_contributions(mmm),
        train_end=train[DATE_COLUMN].iloc[-1],
        test_start=test[DATE_COLUMN].iloc[0],
    )


def save_results(result: MMMResult, output_dir: Path) -> None:
    """Write local, git-ignored evaluation artifacts for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output_dir / "mmm_holdout_predictions.csv", index=False)
    result.diagnostics.to_csv(output_dir / "mmm_diagnostics.csv", index=False)
    result.contributions.to_csv(output_dir / "mmm_channel_contributions.csv", index=False)
    (output_dir / "mmm_metrics.json").write_text(
        json.dumps(
            {
                **result.metrics,
                "train_end": result.train_end.date().isoformat(),
                "test_start": result.test_start.date().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure = px.line(
        result.predictions,
        x=DATE_COLUMN,
        y=["actual", "mmm_prediction", "baseline_prediction"],
        labels={"value": "Sales (y)", DATE_COLUMN: "Week", "variable": "Series"},
        title="MMM and baseline predictions on chronological holdout",
    )
    figure.add_scatter(
        x=result.predictions[DATE_COLUMN],
        y=result.predictions["mmm_upper_90"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        name="90% interval upper bound",
    )
    figure.add_scatter(
        x=result.predictions[DATE_COLUMN],
        y=result.predictions["mmm_lower_90"],
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(99, 110, 250, 0.15)",
        showlegend=True,
        name="MMM 90% predictive interval",
    )
    figure.write_html(output_dir / "mmm_holdout.html")


def main() -> None:
    """Fit the project model and save evaluation outputs."""
    input_path = Path("data/processed/mmm_cleaned.csv")
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Prepare the data before fitting the MMM."
        )

    data = pd.read_csv(input_path)
    result = fit_and_evaluate_mmm(data)
    save_results(result, Path("data/processed"))
    print(f"Training through: {result.train_end.date()}")
    print(f"Holdout starts:   {result.test_start.date()}")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.2f}")
    if "r_hat" in result.diagnostics:
        max_rhat = result.diagnostics["r_hat"].max()
        print(f"Maximum R-hat: {max_rhat:.3f}")
    print(f"Sampler divergences: {int(result.metrics['sampler_divergences'])}")
    print("Saved MMM predictions, diagnostics, contributions, and chart.")
    print("Interpret results as uncertain historical associations, not causal effects.")


if __name__ == "__main__":
    main()
