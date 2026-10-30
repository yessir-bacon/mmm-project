import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mmm_project.scenarios import simulate_budget_scenario, validate_allocation


class FakeMMM:
    """Small deterministic posterior-predictive stand-in for the fitted MMM."""

    output_var = "y"

    def get_scales_as_xarray(self):
        return {"target_scale": xr.DataArray(1.0)}

    def sample_posterior_predictive(self, X, **kwargs):
        center = 100 + 1000 * X["x1"].to_numpy() + 100 * X["x2"].to_numpy()
        offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])[:, None]
        return xr.Dataset({"y": (("sample", "date"), center[None, :] + offsets)})


def _data():
    historical = pd.DataFrame(
        {
            "date_week": pd.date_range("2021-01-04", periods=4, freq="W-MON"),
            "x1": [0.4, 0.6, 0.5, 0.5],
            "x2": [0.1, 0.3, 0.2, 0.2],
            "t": [0, 1, 2, 3],
        }
    )
    context = pd.DataFrame(
        {
            "date_week": pd.date_range("2021-02-01", periods=3, freq="W-MON"),
            "event_1": [0, 1, 0],
            "event_2": [0, 0, 0],
            "t": [4, 5, 6],
        }
    )
    return historical, context


def test_validate_allocation_accepts_exact_channel_budget():
    assert validate_allocation(100, {"x1": 65, "x2": 35}) == {"x1": 65.0, "x2": 35.0}


@pytest.mark.parametrize(
    ("budget", "allocation", "message"),
    [
        (100, {"x1": 50}, "channels do not match"),
        (100, {"x1": 50, "x2": 40}, "sum to 90"),
        (100, {"x1": -1, "x2": 101}, "nonnegative"),
        (100, {"x1": np.inf, "x2": 0}, "finite"),
        (0, {"x1": 0, "x2": 0}, "positive"),
    ],
)
def test_validate_allocation_rejects_invalid_inputs(budget, allocation, message):
    with pytest.raises(ValueError, match=message):
        validate_allocation(budget, allocation)


def test_simulate_budget_scenario_compares_equal_budget_and_flags_extrapolation():
    historical, context = _data()
    result = simulate_budget_scenario(
        FakeMMM(),
        historical,
        context,
        total_budget=3,
        allocations={"x1": 2.7, "x2": 0.3},
    )

    assert result.horizon_weeks == 3
    assert sum(result.proposed_allocations.values()) == pytest.approx(3)
    assert sum(result.baseline_allocations.values()) == pytest.approx(3)
    assert len(result.predictions) == 3
    assert {"proposed_lower_90", "proposed_upper_90"}.issubset(result.predictions.columns)
    proposed = result.summary.loc[result.summary["scenario"] == "proposed", "predicted_total_median"].item()
    baseline = result.summary.loc[
        result.summary["scenario"] == "historical_share_baseline", "predicted_total_median"
    ].item()
    assert proposed > baseline
    assert any("x1" in warning for warning in result.warnings)
