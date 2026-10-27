from pathlib import Path
import pandas as pd

path = Path("data/raw/mmm_example.csv")
df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns and types:")
print(df.dtypes)
print('\nFirst 5 rows:')
print(df.head().to_string())
print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).to_string())
print("\nNumeric summary:")
print(df.describe().to_string())