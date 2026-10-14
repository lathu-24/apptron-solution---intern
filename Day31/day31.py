from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Keep house_prices.csv in the same folder as this script.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "house_prices.csv"

# 1. Load the dataset
df = pd.read_csv(DATA_FILE)

# 2. Inspect the dataset
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSummary statistics:")
print(df.describe())

# 3. Identify features and target
# House ID is an identifier, so it is excluded from model features.
target_column = "Price"
feature_columns = [
    column for column in df.columns
    if column not in ["House ID", target_column]
]

# Remove rows with missing values in selected features or target.
model_df = df[feature_columns + [target_column]].dropna()

X = model_df[feature_columns]
y = model_df[target_column]

print("\nFeatures (X):", feature_columns)
print("Target column (y):", target_column)

# 4. Compare 80/20 and 70/30 train-test splits.
for train_size, test_size, split_name in [
    (0.80, 0.20, "80% Training / 20% Testing"),
    (0.70, 0.30, "70% Training / 30% Testing"),
]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=42
    )

    print(f"\n--- {split_name} ---")
    print("Total Records:", len(model_df))
    print("Training Records:", len(X_train))
    print("Testing Records:", len(X_test))
    print("Number of Features:", X.shape[1])
    print("Target Column:", target_column)

print("\nComparison:")
print("80/20 split provides more records for training.")
print("70/30 split reserves more records for testing.")
print("Both splits use random_state=42 so the results are reproducible.")
