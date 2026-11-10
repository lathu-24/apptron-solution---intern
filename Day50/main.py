
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customer_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("\nFIRST 10 RECORDS\n", df.head(10).to_string(index=False))
print("\nSHAPE:", df.shape)
print("\nDATA TYPES\n", df.dtypes)
print("\nMISSING VALUES\n", df.isnull().sum())
print("\nDUPLICATE ROWS:", df.duplicated().sum())

before = len(df)
df = df.drop_duplicates().copy()
features = ["Age", "Annual Income", "Spending Score"]
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())
df = df[df["Age"].between(1, 100) & df["Annual Income"].ge(0) &
        df["Spending Score"].between(1, 100)].copy()
df.to_csv(OUT / "clean_customer_data.csv", index=False)

stats = df[features].agg(["mean", "median", "min", "max"]).T
stats.columns = ["Mean", "Median", "Minimum", "Maximum"]
stats.to_csv(OUT / "summary_statistics.csv")
print("\nSUMMARY STATISTICS\n", stats.round(2))

for col in features:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=15, edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(OUT / f"{col.lower().replace(' ', '_')}_histogram.png", dpi=150)
    plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["Annual Income"], df["Spending Score"], alpha=.75)
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.savefig(OUT / "income_vs_spending_scatter.png", dpi=150)
plt.close()

corr = df["Annual Income"].corr(df["Spending Score"])
low_high = ((df["Annual Income"] < df["Annual Income"].quantile(.25)) &
            (df["Spending Score"] >= df["Spending Score"].quantile(.75))).sum()
high_low = ((df["Annual Income"] >= df["Annual Income"].quantile(.75)) &
            (df["Spending Score"] < df["Spending Score"].quantile(.25))).sum()
report = f"""CUSTOMER DATA EXPLORATION REPORT

Rows after cleaning: {len(df)}
Columns: {len(df.columns)}
Duplicate rows removed: {before - len(df)}
Missing values after cleaning: {df.isnull().sum().sum()}

SUMMARY STATISTICS
{stats.round(2).to_string()}

OBSERVATIONS
1. Annual Income and Spending Score correlation: {corr:.2f}.
2. Customers with lower-quartile income and upper-quartile spending score: {low_high}.
3. Customers with upper-quartile income and lower-quartile spending score: {high_low}.
4. Customer ages range from {df['Age'].min()} to {df['Age'].max()} years.
5. Check the scatter plot for separated groups and unusual points; these may indicate distinct customer segments.

CLUSTERING FEATURES
Use Age, Annual Income, and Spending Score as clustering features.
Exclude Customer ID: it is an arbitrary identifier, not a behavioural measurement.
Including it can create meaningless distance relationships between customers.
Standardize the selected features before distance-based clustering such as K-Means.
"""
(OUT / "observation_report.txt").write_text(report, encoding="utf-8")
print("\nCompleted. See the outputs folder.")
