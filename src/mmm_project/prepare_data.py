from pathlib import Path
import pandas as pd

path = Path("data/raw/mmm_example.csv")
df = pd.read_csv(path, parse_dates=["date_week"])

# Check for missing data
if df.isnull().any().any():
    raise ValueError("Missing data found in the dataset.")

# Check for duplicate date_week values
if df['date_week'].duplicated().any():
    raise ValueError("Duplicate date_week values found in the dataset.")

# Sort the DataFrame by date_week
df = df.sort_values(by="date_week").reset_index(drop=True)

# Save the cleaned DataFrame to a new CSV file
out_path = Path("data/processed")
out_path.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path / "mmm_cleaned.csv", index=False)

print("Saved:", out_path / "mmm_cleaned.csv")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Date range:", df['date_week'].min().date(), "to", df['date_week'].max().date())