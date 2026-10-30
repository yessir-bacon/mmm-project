import sys
from pathlib import Path

import pandas as pd

# Make the src/mmm_project package importable from this script.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mmm_project.model import fit_and_evaluate_mmm, prepare_model_data
from mmm_project.scenarios import simulate_budget_scenario


def main():
    data = pd.read_csv("data/processed/mmm_cleaned.csv")
    prepared = prepare_model_data(data)
    split = int(len(prepared) * 0.8)

    print("Fitting the MMM; this may take a few minutes...")
    fitted = fit_and_evaluate_mmm(data)

    scenario = simulate_budget_scenario(
        model=fitted.model,
        historical_data=prepared.iloc[:split],
        context=prepared.iloc[split:],
        total_budget=18.0,
        allocations={"x1": 12.0, "x2": 6.0},
    )

    print("\nScenario summary:")
    print(scenario.summary.to_string(index=False))

    print("\nWarnings:")
    print("\n".join(scenario.warnings) if scenario.warnings else "None")

    output_dir = Path("data/processed")
    scenario.summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    scenario.predictions.to_csv(output_dir / "scenario_predictions.csv", index=False)
    print(f"\nSaved scenario results to {output_dir}.")


if __name__ == "__main__":
    main()