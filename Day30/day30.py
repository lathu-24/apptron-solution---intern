from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "house_prices.csv"
PLOT_FILE = BASE_DIR / "area_vs_price.png"

# Load and inspect the dataset
df = pd.read_csv(DATA_FILE)
print("Shape:", df.shape)
print("\nFirst five records:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# Plot Area vs Price and draw an approximate line manually
x = df["Area (sq.ft)"]
y = df["Price"]
line_x = np.array([x.min(), x.max()])
line_y = 50000 + 150 * line_x

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.75, label="House records")
plt.plot(line_x, line_y, "--", linewidth=2,
         label="Approximate line: Price = 50,000 + 150 × Area")
plt.title("House Area vs Price")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()

# Assess the relationship
correlation = x.corr(y)
relationship = "Positive" if correlation > 0 else "Negative" if correlation < 0 else "No linear"
linear_suitable = correlation > 0.5

# Find possible outliers as unusually large residuals from the approximate line
expected_price = 50000 + 150 * x
residuals = y - expected_price
q1, q3 = residuals.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(residuals < lower) | (residuals > upper)].copy()
outliers["Difference from manual line"] = residuals.loc[outliers.index].round(0).astype(int)
outlier_ids = ", ".join(outliers["House ID"].astype(str)) if not outliers.empty else "None detected"

report = f"""Day 30 - House Price Relationship Analyzer

Feature: Area (sq.ft)
Target: Price
Relationship: {relationship} relationship (correlation = {correlation:.3f}).
Possible Outliers: {outlier_ids}
Is Linear Regression Suitable? {"Yes, as a simple baseline." if linear_suitable else "Possibly, but the relationship is not strongly linear."}
Reason: The plot and correlation indicate whether price generally changes with area. The relationship is not perfect because age, bedrooms, bathrooms, location, and other factors can affect price. Potential outliers were flagged using the IQR rule on price.
Manual Line: Price = 50,000 + 150 × Area
Manual Line Fit: This is an approximate visual line; it will not pass through every point and may fit unusual prices poorly.
"""
REPORT_FILE.write_text(report, encoding="utf-8")
print(f"\nCorrelation (Area vs Price): {correlation:.3f}")
print("\nPossible outliers:")
print(outliers[["House ID", "Area (sq.ft)", "Price", "Difference from manual line"]] if not outliers.empty else "None detected")
print(f"\nSaved plot to: {PLOT_FILE}")
print(f"Saved report to: {REPORT_FILE}")
print("\n" + report)
