import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# 1. Create Customer Dataset
# --------------------------------

np.random.seed(42)

data = {
    "Age": np.random.randint(18, 66, 1000),

    "Income": np.random.randint(25000, 150001, 1000),

    "Purchase Amount": np.random.randint(500, 25001, 1000),

    "Website Visits": np.random.randint(1, 31, 1000),

    "Time Spent": np.random.randint(5, 181, 1000),

    "Customer Rating": np.round(
        np.random.uniform(1, 5, 1000), 1
    )
}

df = pd.DataFrame(data)

print("CUSTOMER BEHAVIOR DATA")
print("=" * 60)
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# --------------------------------
# 2. Scatter Plot
# Income vs Purchase Amount
# --------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Income"],
    df["Purchase Amount"],
    alpha=0.6,
    label="Customers"
)

plt.title("Income vs Purchase Amount")
plt.xlabel("Income")
plt.ylabel("Purchase Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------
# 3. Histogram
# Customer Age
# --------------------------------

plt.figure(figsize=(9, 6))

plt.hist(
    df["Age"],
    bins=10,
    edgecolor="black",
    label="Customer Age"
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------
# 4. Histogram
# Purchase Amount
# --------------------------------

plt.figure(figsize=(9, 6))

plt.hist(
    df["Purchase Amount"],
    bins=15,
    edgecolor="black",
    label="Purchase Amount"
)

plt.title("Purchase Amount Distribution")
plt.xlabel("Purchase Amount")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------
# 5. Scatter Plot
# Website Visits vs Purchase Amount
# --------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Website Visits"],
    df["Purchase Amount"],
    alpha=0.6,
    label="Customers"
)

plt.title("Website Visits vs Purchase Amount")
plt.xlabel("Website Visits")
plt.ylabel("Purchase Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------
# 6. Highest-Spending Group
# --------------------------------

age_groups = pd.cut(
    df["Age"],
    bins=[17, 25, 35, 45, 55, 65],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65"
    ]
)

df["Age Group"] = age_groups

spending_by_age_group = (
    df.groupby("Age Group", observed=False)["Purchase Amount"]
    .mean()
    .sort_values(ascending=False)
)

highest_spending_group = spending_by_age_group.idxmax()

print("\nCUSTOMER BEHAVIOR ANALYSIS")
print("=" * 60)

print(
    f"Highest-Spending Age Group: "
    f"{highest_spending_group}"
)

print(
    f"Average Purchase Amount: "
    f"{spending_by_age_group.max():.2f}"
)


# --------------------------------
# 7. Detect Purchase Amount Outliers
# Using IQR Method
# --------------------------------

Q1 = df["Purchase Amount"].quantile(0.25)
Q3 = df["Purchase Amount"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["Purchase Amount"] < lower_limit) |
    (df["Purchase Amount"] > upper_limit)
]

print("\nOUTLIER ANALYSIS")
print("=" * 60)

print(f"Q1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")

print(f"Lower Limit: {lower_limit:.2f}")
print(f"Upper Limit: {upper_limit:.2f}")

print(f"Number of Outliers: {len(outliers)}")


# --------------------------------
# 8. Correlation Analysis
# --------------------------------

numerical_features = [
    "Age",
    "Income",
    "Purchase Amount",
    "Website Visits",
    "Time Spent",
    "Customer Rating"
]

correlation_matrix = df[numerical_features].corr()

print("\nCORRELATION MATRIX")
print("=" * 60)
print(correlation_matrix.round(2))


# --------------------------------
# 9. Find Strongest Relationships
# --------------------------------

correlation_pairs = (
    correlation_matrix
    .where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )
    .stack()
    .sort_values(
        key=lambda x: abs(x),
        ascending=False
    )
)

print("\nSTRONGEST CORRELATION PAIRS")
print("=" * 60)
print(correlation_pairs.head(5))


# --------------------------------
# 10. Save Dataset
# --------------------------------

df.to_csv(
    "customer_behavior.csv",
    index=False
)

outliers.to_csv(
    "customer_outliers.csv",
    index=False
)

correlation_matrix.to_csv(
    "customer_correlations.csv"
)

spending_by_age_group.to_csv(
    "spending_by_age_group.csv"
)


# --------------------------------
# 11. Final Summary
# --------------------------------

print("\nFILES GENERATED")
print("=" * 60)

print("1. customer_behavior.csv")
print("2. customer_outliers.csv")
print("3. customer_correlations.csv")
print("4. spending_by_age_group.csv")

print("\nCustomer behavior analysis completed!")