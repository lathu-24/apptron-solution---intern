import pandas as pd
import numpy as np

# ==========================================
# Day 22 - AI/ML Customer Churn Dataset Preparation System
# ==========================================

np.random.seed(42)

# ==========================================
# PART 1 - CREATE RAW DATASET
# ==========================================

print("=" * 70)
print("CREATING CUSTOMER CHURN DATASET")
print("=" * 70)

n_customers = 5500

genders = ["Male", "Female"]

cities = [
    "Colombo",
    "Jaffna",
    "Kandy",
    "Galle",
    "Negombo",
    "Batticaloa",
    "Kurunegala"
]

contract_types = [
    "Month-to-Month",
    "One Year",
    "Two Year"
]

payment_methods = [
    "Credit Card",
    "Bank Transfer",
    "Cash",
    "Online Payment"
]

data = {
    "Customer ID": [
        f"C{i:05d}" for i in range(1, n_customers + 1)
    ],

    "Age": np.random.randint(
        18, 71, n_customers
    ),

    "Gender": np.random.choice(
        genders,
        n_customers
    ),

    "City": np.random.choice(
        cities,
        n_customers
    ),

    "Tenure": np.random.randint(
        1, 73, n_customers
    ),

    "Monthly Charges": np.round(
        np.random.uniform(
            1000, 50000, n_customers
        ),
        2
    ),

    "Total Charges": np.round(
        np.random.uniform(
            5000, 3000000, n_customers
        ),
        2
    ),

    "Contract Type": np.random.choice(
        contract_types,
        n_customers
    ),

    "Payment Method": np.random.choice(
        payment_methods,
        n_customers
    ),

    "Support Calls": np.random.randint(
        0, 16, n_customers
    ),

    "Internet Usage": np.round(
        np.random.uniform(
            1, 300, n_customers
        ),
        2
    ),

    "Churn": np.random.choice(
        ["Yes", "No"],
        n_customers,
        p=[0.25, 0.75]
    )
}

df = pd.DataFrame(data)


# ==========================================
# INTRODUCE DATA QUALITY PROBLEMS
# ==========================================

# Missing values

missing_age = np.random.choice(
    df.index,
    50,
    replace=False
)

df.loc[missing_age, "Age"] = np.nan


missing_monthly = np.random.choice(
    df.index,
    50,
    replace=False
)

df.loc[missing_monthly, "Monthly Charges"] = np.nan


missing_city = np.random.choice(
    df.index,
    30,
    replace=False
)

df.loc[missing_city, "City"] = np.nan


# Invalid ages

invalid_age_indices = np.random.choice(
    df.index,
    25,
    replace=False
)

df.loc[
    invalid_age_indices,
    "Age"
] = np.random.choice(
    [-5, 0, 150, 200],
    25
)


# Invalid charges

invalid_charge_indices = np.random.choice(
    df.index,
    20,
    replace=False
)

df.loc[
    invalid_charge_indices,
    "Monthly Charges"
] = np.random.choice(
    [-1000, -500, 0],
    20
)


# Inconsistent categorical values

city_problem_indices = np.random.choice(
    df.index,
    40,
    replace=False
)

df.loc[
    city_problem_indices,
    "City"
] = np.random.choice(
    ["colombo", "COLOMBO", " Colombo ", "Jaffna "],
    40
)


# Create duplicate records

duplicates = df.sample(
    30,
    random_state=42
)

df = pd.concat(
    [df, duplicates],
    ignore_index=True
)


# ==========================================
# SAVE RAW DATA
# ==========================================

df.to_csv(
    "raw_customer_data.csv",
    index=False
)

print("\nRaw dataset created.")
print("Raw records:", len(df))


# ==========================================
# PART 1 - LOAD
# ==========================================

df = pd.read_csv(
    "raw_customer_data.csv"
)

print("\n" + "=" * 70)
print("PART 1 - LOAD DATA")
print("=" * 70)

print(df.head())


# ==========================================
# PART 2 - INSPECT
# ==========================================

print("\n" + "=" * 70)
print("PART 2 - DATA INSPECTION")
print("=" * 70)

# Shape

print("\nShape:")
print(df.shape)


# Columns

print("\nColumns:")
print(df.columns.tolist())


# Data types

print("\nData Types:")
print(df.dtypes)


# Missing values

print("\nMissing Values:")
print(df.isnull().sum())


# Duplicate count

duplicate_count = df.duplicated(
    subset=["Customer ID"]
).sum()

print("\nDuplicate Count:")
print(duplicate_count)


# Statistical summary

print("\nStatistical Summary:")
print(df.describe())


# ==========================================
# SAVE BEFORE-CLEANING INFORMATION
# ==========================================

total_customers_before = len(df)

missing_values_before = (
    df.isnull().sum().sum()
)


# ==========================================
# PART 3 - CLEAN
# ==========================================

print("\n" + "=" * 70)
print("PART 3 - DATA CLEANING")
print("=" * 70)


# ------------------------------------------
# 1. Remove duplicate records
# ------------------------------------------

duplicates_removed = df.duplicated(
    subset=["Customer ID"]
).sum()

df = df.drop_duplicates(
    subset=["Customer ID"],
    keep="first"
)

print(
    "Duplicates removed:",
    duplicates_removed
)


# ------------------------------------------
# 2. Fix invalid ages
# ------------------------------------------

invalid_ages = (
    (df["Age"] < 18) |
    (df["Age"] > 100)
)

invalid_age_count = invalid_ages.sum()

median_age = df.loc[
    ~invalid_ages,
    "Age"
].median()

df.loc[
    invalid_ages,
    "Age"
] = median_age

print(
    "Invalid ages fixed:",
    invalid_age_count
)


# Fill missing Age

missing_age_count = df["Age"].isnull().sum()

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)


# ------------------------------------------
# 3. Fix invalid Monthly Charges
# ------------------------------------------

invalid_charges = (
    df["Monthly Charges"] <= 0
)

invalid_charge_count = invalid_charges.sum()

median_monthly_charge = df.loc[
    ~invalid_charges,
    "Monthly Charges"
].median()

df.loc[
    invalid_charges,
    "Monthly Charges"
] = median_monthly_charge


# Fill missing Monthly Charges

df["Monthly Charges"] = df[
    "Monthly Charges"
].fillna(
    df["Monthly Charges"].median()
)

print(
    "Invalid charges fixed:",
    invalid_charge_count
)


# ------------------------------------------
# 4. Clean Total Charges
# ------------------------------------------

invalid_total_charges = (
    df["Total Charges"] <= 0
)

total_charge_count = invalid_total_charges.sum()

df.loc[
    invalid_total_charges,
    "Total Charges"
] = df["Total Charges"].median()

print(
    "Invalid total charges fixed:",
    total_charge_count
)


# ------------------------------------------
# 5. Clean categorical values
# ------------------------------------------

df["City"] = (
    df["City"]
    .astype("string")
    .str.strip()
    .str.title()
)

df["Gender"] = (
    df["Gender"]
    .astype("string")
    .str.strip()
    .str.title()
)

df["Contract Type"] = (
    df["Contract Type"]
    .astype("string")
    .str.strip()
)

df["Payment Method"] = (
    df["Payment Method"]
    .astype("string")
    .str.strip()
)

# Fill missing categorical values

df["City"] = df["City"].fillna(
    df["City"].mode()[0]
)

df["Gender"] = df["Gender"].fillna(
    df["Gender"].mode()[0]
)

df["Contract Type"] = df[
    "Contract Type"
].fillna(
    df["Contract Type"].mode()[0]
)

df["Payment Method"] = df[
    "Payment Method"
].fillna(
    df["Payment Method"].mode()[0]
)


# ------------------------------------------
# 6. Clean Churn values
# ------------------------------------------

df["Churn"] = (
    df["Churn"]
    .astype(str)
    .str.strip()
    .str.title()
)


# ==========================================
# SAVE CLEAN DATASET
# ==========================================

df.to_csv(
    "clean_customer_data.csv",
    index=False
)

print("\nClean dataset saved.")


# ==========================================
# CHECK CLEAN DATA
# ==========================================

missing_values_after = (
    df.isnull().sum().sum()
)

print("\nRemaining Missing Values:")
print(missing_values_after)


# ==========================================
# PART 4 - FILTER
# ==========================================

print("\n" + "=" * 70)
print("PART 4 - CUSTOMER FILTERING")
print("=" * 70)


# High-value customers

high_value_customers = df[
    df["Total Charges"] > 1000000
]

print(
    "\nHigh-value customers:",
    len(high_value_customers)
)


# Customers with many support calls

many_support_calls = df[
    df["Support Calls"] >= 10
]

print(
    "Customers with many support calls:",
    len(many_support_calls)
)


# High monthly charges

high_monthly_charges = df[
    df["Monthly Charges"] > 40000
]

print(
    "Customers with high monthly charges:",
    len(high_monthly_charges)
)


# Churned customers

churned_customers = df[
    df["Churn"] == "Yes"
]

print(
    "Churned customers:",
    len(churned_customers)
)


# Customers who stayed

stayed_customers = df[
    df["Churn"] == "No"
]

print(
    "Customers who stayed:",
    len(stayed_customers)
)


# ==========================================
# PART 5 - GROUP ANALYSIS
# ==========================================

print("\n" + "=" * 70)
print("PART 5 - GROUP ANALYSIS")
print("=" * 70)


# Churn by Contract Type

churn_by_contract = pd.crosstab(
    df["Contract Type"],
    df["Churn"]
)

print("\nChurn by Contract Type:")
print(churn_by_contract)


# Churn by City

churn_by_city = pd.crosstab(
    df["City"],
    df["Churn"]
)

print("\nChurn by City:")
print(churn_by_city)


# Average charges by customer type

average_charges_by_contract = (
    df.groupby("Contract Type")
    .agg(
        Average_Monthly_Charges=(
            "Monthly Charges",
            "mean"
        ),
        Average_Total_Charges=(
            "Total Charges",
            "mean"
        )
    )
    .round(2)
)

print(
    "\nAverage Charges by Customer Type:"
)

print(average_charges_by_contract)


# Average support calls by churn status

average_support_by_churn = (
    df.groupby("Churn")["Support Calls"]
    .mean()
    .round(2)
)

print(
    "\nAverage Support Calls by Churn Status:"
)

print(average_support_by_churn)


# ==========================================
# PART 6 - PREPARE ML DATASET
# ==========================================

print("\n" + "=" * 70)
print("PART 6 - ML DATASET PREPARATION")
print("=" * 70)


# Remove unnecessary Customer ID

ml_df = df.drop(
    columns=["Customer ID"]
)


# Convert categorical columns using One-Hot Encoding

categorical_columns = [
    "Gender",
    "City",
    "Contract Type",
    "Payment Method"
]

ml_df = pd.get_dummies(
    ml_df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# Separate features and target

X = ml_df.drop(
    columns=["Churn"]
)

y = ml_df["Churn"].map({
    "No": 0,
    "Yes": 1
})


print("\nFeature Dataset Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# Combine X and y for final ML-ready CSV

ml_ready_df = X.copy()

ml_ready_df["Churn"] = y


# Save ML-ready dataset

ml_ready_df.to_csv(
    "ml_ready_customer_data.csv",
    index=False
)

print(
    "\nML-ready dataset saved."
)


# ==========================================
# PART 7 - FINAL REPORT
# ==========================================

total_customers = len(df)

clean_customers = len(df)

churned_count = len(
    df[df["Churn"] == "Yes"]
)

churn_rate = (
    churned_count /
    total_customers *
    100
)

average_monthly_charges = (
    df["Monthly Charges"].mean()
)


# ==========================================
# Find Highest Churn Segment
# ==========================================

churn_segment = (
    df.groupby("Contract Type")["Churn"]
    .apply(
        lambda x:
        (x == "Yes").mean() * 100
    )
    .sort_values(
        ascending=False
    )
)

highest_churn_segment = (
    churn_segment.index[0]
)

highest_churn_rate = (
    churn_segment.iloc[0]
)


# ==========================================
# DISPLAY FINAL REPORT
# ==========================================

print("\n" + "=" * 70)
print("FINAL CUSTOMER CHURN REPORT")
print("=" * 70)

print(
    "Total Customers:",
    total_customers
)

print(
    "Clean Customers:",
    clean_customers
)

print(
    "Removed Duplicates:",
    duplicates_removed
)

print(
    "Missing Values Before:",
    missing_values_before
)

print(
    "Missing Values After:",
    missing_values_after
)

print(
    "Churned Customers:",
    churned_count
)

print(
    "Churn Rate:",
    round(churn_rate, 2),
    "%"
)

print(
    "Average Monthly Charges:",
    round(
        average_monthly_charges,
        2
    )
)

print(
    "Highest Churn Segment:",
    highest_churn_segment
)

print(
    "Highest Churn Rate:",
    round(
        highest_churn_rate,
        2
    ),
    "%"
)


# ==========================================
# SAVE ANALYSIS REPORTS
# ==========================================

churn_by_contract.to_csv(
    "churn_by_contract.csv"
)

churn_by_city.to_csv(
    "churn_by_city.csv"
)

average_charges_by_contract.to_csv(
    "average_charges_by_contract.csv"
)

average_support_by_churn.to_csv(
    "average_support_by_churn.csv"
)

print("\n" + "=" * 70)
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")
print("1. raw_customer_data.csv")
print("2. clean_customer_data.csv")
print("3. ml_ready_customer_data.csv")
print("4. churn_by_contract.csv")
print("5. churn_by_city.csv")
print("6. average_charges_by_contract.csv")
print("7. average_support_by_churn.csv")