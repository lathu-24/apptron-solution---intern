import pandas as pd
import numpy as np

# ==========================================
# Day 18 - E-Commerce Data Cleaning Pipeline
# ==========================================

np.random.seed(42)

# ==========================================
# 1. Create 1,000 Raw Orders
# ==========================================

products = [
    "Laptop", "Phone", "Headphones", "Keyboard",
    "Mouse", "Monitor", "Tablet", "Smart Watch"
]

categories = [
    "Electronics", "Electronics", "Electronics",
    "Accessories", "Accessories", "Electronics",
    "Electronics", "Electronics"
]

cities = [
    "Colombo", "Jaffna", "Kandy", "Galle",
    "Negombo", "Colombo", "colombo",
    "Jaffna ", "KANDY", "galle"
]

dates = pd.date_range(
    start="2025-01-01",
    end="2025-12-31",
    periods=1000
)

raw_data = {
    "Order ID": [f"O{i:04d}" for i in range(1, 1001)],
    "Customer ID": [
        f"C{np.random.randint(1, 301):03d}"
        for _ in range(1000)
    ],
    "Product": np.random.choice(products, 1000),
    "Category": np.random.choice(categories, 1000),
    "Quantity": np.random.randint(1, 11, 1000),
    "Price": np.round(np.random.uniform(500, 250000, 1000), 2),
    "City": np.random.choice(cities, 1000),
    "Order Date": dates
}

df = pd.DataFrame(raw_data)

# ==========================================
# 2. Introduce Data Problems
# ==========================================

# Missing prices
missing_price_indices = np.random.choice(
    df.index,
    size=30,
    replace=False
)

df.loc[missing_price_indices, "Price"] = np.nan


# Invalid quantities
invalid_quantity_indices = np.random.choice(
    df.index,
    size=20,
    replace=False
)

df.loc[invalid_quantity_indices, "Quantity"] = np.random.choice(
    [0, -1, -5],
    size=20
)


# Incorrect date formats
date_indices = np.random.choice(
    df.index,
    size=25,
    replace=False
)

for index in date_indices:
    original_date = df.loc[index, "Order Date"]

    if pd.notna(original_date):
        df.loc[index, "Order Date"] = original_date.strftime("%d/%m/%Y")


# Create duplicate orders
duplicate_rows = df.sample(
    15,
    random_state=42
)

df = pd.concat(
    [df, duplicate_rows],
    ignore_index=True
)


# ==========================================
# 3. Save Raw Dataset
# ==========================================

df.to_csv("raw_orders.csv", index=False)

print("=" * 60)
print("RAW E-COMMERCE DATASET")
print("=" * 60)

print("Total records:", len(df))
print(df.head())


# ==========================================
# 4. Cleaning Log
# ==========================================

cleaning_log = []


# ==========================================
# 5. Check Missing Prices
# ==========================================

missing_prices = df["Price"].isnull().sum()

print("\nMissing Prices:", missing_prices)

if missing_prices > 0:

    median_price = df["Price"].median()

    df["Price"] = df["Price"].fillna(median_price)

    cleaning_log.append(
        f"Filled {missing_prices} missing Price values "
        f"using median price ({median_price:.2f})."
    )


# ==========================================
# 6. Remove Duplicate Orders
# ==========================================

duplicate_count = df.duplicated(
    subset=["Order ID"]
).sum()

print("Duplicate Orders:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates(
        subset=["Order ID"],
        keep="first"
    )

    cleaning_log.append(
        f"Removed {duplicate_count} duplicate orders "
        f"based on Order ID."
    )


# ==========================================
# 7. Fix Invalid Quantities
# ==========================================

invalid_quantity_count = (
    df["Quantity"] <= 0
).sum()

print("Invalid Quantities:", invalid_quantity_count)

if invalid_quantity_count > 0:

    median_quantity = df.loc[
        df["Quantity"] > 0,
        "Quantity"
    ].median()

    df.loc[
        df["Quantity"] <= 0,
        "Quantity"
    ] = median_quantity

    cleaning_log.append(
        f"Replaced {invalid_quantity_count} invalid "
        f"quantities using median quantity "
        f"({median_quantity:.0f})."
    )


# ==========================================
# 8. Standardize City Names
# ==========================================

city_before = df["City"].copy()

df["City"] = (
    df["City"]
    .astype(str)
    .str.strip()
    .str.title()
)

city_changes = (
    city_before != df["City"]
).sum()

print("City Names Standardized:", city_changes)

cleaning_log.append(
    f"Standardized {city_changes} inconsistent city names "
    f"using strip() and title()."
)


# ==========================================
# 9. Fix Order Date Formats
# ==========================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True,
    errors="coerce"
)

invalid_dates = df["Order Date"].isnull().sum()

if invalid_dates > 0:

    df["Order Date"] = df["Order Date"].fillna(
        df["Order Date"].median()
    )

    cleaning_log.append(
        f"Fixed {invalid_dates} invalid Order Date values "
        f"using the median date."
    )
else:

    cleaning_log.append(
        "Converted all Order Date values to a standard "
        "datetime format."
    )


# ==========================================
# 10. Standardize Date Format
# ==========================================

df["Order Date"] = df["Order Date"].dt.strftime(
    "%Y-%m-%d"
)

cleaning_log.append(
    "Converted Order Date to YYYY-MM-DD format."
)


# ==========================================
# 11. Check Remaining Problems
# ==========================================

remaining_missing = df.isnull().sum().sum()

remaining_duplicates = df.duplicated(
    subset=["Order ID"]
).sum()

remaining_invalid_quantity = (
    df["Quantity"] <= 0
).sum()


# ==========================================
# 12. Save Clean Dataset
# ==========================================

df.to_csv(
    "clean_orders.csv",
    index=False
)


# ==========================================
# 13. Generate Cleaning Log
# ==========================================

log_df = pd.DataFrame({
    "Step": range(1, len(cleaning_log) + 1),
    "Transformation": cleaning_log
})

log_df.to_csv(
    "cleaning_log.csv",
    index=False
)


# ==========================================
# 14. Final Summary
# ==========================================

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

print("Records before cleaning:", len(raw_data["Order ID"]))
print("Records after cleaning:", len(df))

print("\nRemaining missing values:", remaining_missing)
print("Remaining duplicate orders:", remaining_duplicates)
print("Remaining invalid quantities:", remaining_invalid_quantity)

print("\n" + "=" * 60)
print("CLEANING LOG")
print("=" * 60)

for i, log in enumerate(cleaning_log, start=1):
    print(f"{i}. {log}")


print("\n" + "=" * 60)
print("PIPELINE COMPLETED")
print("=" * 60)

print("Generated files:")
print("1. raw_orders.csv")
print("2. clean_orders.csv")
print("3. cleaning_log.csv")