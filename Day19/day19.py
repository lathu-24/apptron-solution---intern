import pandas as pd
import numpy as np

# ==========================================
# Day 19 - Customer Segmentation System
# ==========================================

np.random.seed(42)

# ==========================================
# 1. Create 1,000 Customer Records
# ==========================================

cities = [
    "Colombo",
    "Jaffna",
    "Kandy",
    "Galle",
    "Negombo",
    "Batticaloa",
    "Kurunegala"
]

product_categories = [
    "Electronics",
    "Clothing",
    "Food",
    "Beauty",
    "Sports",
    "Home"
]

data = {
    "Customer ID": [f"C{i:04d}" for i in range(1, 1001)],
    "Age": np.random.randint(18, 71, 1000),
    "Gender": np.random.choice(
        ["Male", "Female"],
        1000
    ),
    "City": np.random.choice(
        cities,
        1000
    ),
    "Income": np.random.randint(
        30000,
        200001,
        1000
    ),
    "Purchase Amount": np.random.randint(
        0,
        100001,
        1000
    ),
    "Product Category": np.random.choice(
        product_categories,
        1000
    )
}

df = pd.DataFrame(data)

# ==========================================
# 2. Create Customers Who Never Purchased
# ==========================================

never_purchased_indices = np.random.choice(
    df.index,
    size=50,
    replace=False
)

df.loc[
    never_purchased_indices,
    "Purchase Amount"
] = 0

# ==========================================
# 3. Display Dataset
# ==========================================

print("=" * 60)
print("CUSTOMER DATASET")
print("=" * 60)

print(df.head(10))

print("\nTotal Customers:", len(df))

# ==========================================
# 4. Customers Above Age 40
# ==========================================

age_above_40 = df[df["Age"] > 40]

print("\n" + "=" * 60)
print("CUSTOMERS ABOVE AGE 40")
print("=" * 60)

print(age_above_40)

print("Count:", len(age_above_40))

# ==========================================
# 5. Customers With Income > 100,000
# ==========================================

high_income = df[df["Income"] > 100000]

print("\n" + "=" * 60)
print("CUSTOMERS WITH INCOME > 100,000")
print("=" * 60)

print(high_income)

print("Count:", len(high_income))

# ==========================================
# 6. Customers Spending > 50,000
# ==========================================

high_spending = df[
    df["Purchase Amount"] > 50000
]

print("\n" + "=" * 60)
print("CUSTOMERS SPENDING > 50,000")
print("=" * 60)

print(high_spending)

print("Count:", len(high_spending))

# ==========================================
# 7. Colombo Customers
# ==========================================

colombo_customers = df[
    df["City"] == "Colombo"
]

print("\n" + "=" * 60)
print("COLOMBO CUSTOMERS")
print("=" * 60)

print(colombo_customers)

print("Count:", len(colombo_customers))

# ==========================================
# 8. High Income + High Spending
# ==========================================

high_income_high_spending = df[
    (df["Income"] > 100000) &
    (df["Purchase Amount"] > 50000)
]

print("\n" + "=" * 60)
print("HIGH-INCOME + HIGH-SPENDING CUSTOMERS")
print("=" * 60)

print(high_income_high_spending)

print("Count:", len(high_income_high_spending))

# ==========================================
# 9. Customers Who Never Purchased
# ==========================================

never_purchased = df[
    df["Purchase Amount"] == 0
]

print("\n" + "=" * 60)
print("CUSTOMERS WHO NEVER PURCHASED")
print("=" * 60)

print(never_purchased)

print("Count:", len(never_purchased))

# ==========================================
# 10. Customer Segmentation
# ==========================================

def classify_customer(spending):

    if spending > 50000:
        return "Premium"

    elif spending > 20000:
        return "Regular"

    else:
        return "Low Value"


df["Customer Category"] = (
    df["Purchase Amount"]
    .apply(classify_customer)
)

# ==========================================
# 11. Display Customer Categories
# ==========================================

print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION")
print("=" * 60)

print(
    df[
        [
            "Customer ID",
            "Purchase Amount",
            "Customer Category"
        ]
    ].head(20)
)

# ==========================================
# 12. Category Summary
# ==========================================

category_summary = (
    df["Customer Category"]
    .value_counts()
)

print("\n" + "=" * 60)
print("CUSTOMER CATEGORY SUMMARY")
print("=" * 60)

print(category_summary)

# ==========================================
# 13. Save Results
# ==========================================

df.to_csv(
    "customer_segmentation.csv",
    index=False
)

age_above_40.to_csv(
    "customers_above_40.csv",
    index=False
)

high_income_high_spending.to_csv(
    "high_income_high_spending.csv",
    index=False
)

print("\n" + "=" * 60)
print("FILES GENERATED")
print("=" * 60)

print("1. customer_segmentation.csv")
print("2. customers_above_40.csv")
print("3. high_income_high_spending.csv")

print("\nCustomer segmentation completed successfully!")
