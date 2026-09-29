import pandas as pd
import numpy as np

# ==========================================
# Day 20 - Sales Performance Analyzer
# ==========================================

np.random.seed(42)

# ==========================================
# 1. Create 2,000 Sales Records
# ==========================================

salespeople = [
    "John", "Sarah", "Michael", "David",
    "Emma", "Daniel", "Sophia", "James"
]

regions = [
    "North", "South", "East", "West"
]

products = [
    "Laptop", "Phone", "Tablet", "Monitor",
    "Keyboard", "Mouse", "Headphones", "Printer"
]

categories = [
    "Electronics", "Electronics", "Electronics",
    "Electronics", "Accessories", "Accessories",
    "Accessories", "Office"
]

data = {
    "Order ID": [
        f"O{i:05d}" for i in range(1, 2001)
    ],

    "Salesperson": np.random.choice(
        salespeople,
        2000
    ),

    "Region": np.random.choice(
        regions,
        2000
    ),

    "Product": np.random.choice(
        products,
        2000
    ),

    "Category": np.random.choice(
        categories,
        2000
    ),

    "Quantity": np.random.randint(
        1,
        11,
        2000
    ),

    "Revenue": np.round(
        np.random.uniform(
            1000,
            100000,
            2000
        ),
        2
    ),

    "Profit": np.round(
        np.random.uniform(
            100,
            30000,
            2000
        ),
        2
    )
}

df = pd.DataFrame(data)

# ==========================================
# 2. Display Dataset
# ==========================================

print("=" * 60)
print("SALES DATASET")
print("=" * 60)

print(df.head(10))

print("\nTotal Sales Records:", len(df))

# ==========================================
# 3. Revenue by Region
# ==========================================

revenue_by_region = (
    df.groupby("Region")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("REVENUE BY REGION")
print("=" * 60)

print(revenue_by_region)

# ==========================================
# 4. Revenue by Category
# ==========================================

revenue_by_category = (
    df.groupby("Category")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("REVENUE BY CATEGORY")
print("=" * 60)

print(revenue_by_category)

# ==========================================
# 5. Profit by Salesperson
# ==========================================

profit_by_salesperson = (
    df.groupby("Salesperson")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("PROFIT BY SALESPERSON")
print("=" * 60)

print(profit_by_salesperson)

# ==========================================
# 6. Top Salesperson
# ==========================================

top_salesperson = profit_by_salesperson.idxmax()
top_salesperson_profit = profit_by_salesperson.max()

print("\n" + "=" * 60)
print("TOP SALESPERSON")
print("=" * 60)

print("Salesperson:", top_salesperson)
print("Total Profit:", round(top_salesperson_profit, 2))

# ==========================================
# 7. Best-Performing Region
# ==========================================

region_profit = (
    df.groupby("Region")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

best_region = region_profit.idxmax()
best_region_profit = region_profit.max()

print("\n" + "=" * 60)
print("BEST-PERFORMING REGION")
print("=" * 60)

print("Region:", best_region)
print("Total Profit:", round(best_region_profit, 2))

# ==========================================
# 8. Best-Selling Category
# ==========================================

category_quantity = (
    df.groupby("Category")["Quantity"]
    .sum()
    .sort_values(ascending=False)
)

best_category = category_quantity.idxmax()
best_category_quantity = category_quantity.max()

print("\n" + "=" * 60)
print("BEST-SELLING CATEGORY")
print("=" * 60)

print("Category:", best_category)
print("Total Quantity Sold:", best_category_quantity)

# ==========================================
# 9. Rank Regions by Total Profit
# ==========================================

region_profit_ranking = (
    df.groupby("Region")["Profit"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

region_profit_ranking.insert(
    0,
    "Rank",
    range(1, len(region_profit_ranking) + 1)
)

print("\n" + "=" * 60)
print("REGION PROFIT RANKING")
print("=" * 60)

print(region_profit_ranking)

# ==========================================
# 10. Save Analysis Results
# ==========================================

df.to_csv(
    "sales_data.csv",
    index=False
)

revenue_by_region.to_csv(
    "revenue_by_region.csv"
)

revenue_by_category.to_csv(
    "revenue_by_category.csv"
)

profit_by_salesperson.to_csv(
    "profit_by_salesperson.csv"
)

region_profit_ranking.to_csv(
    "region_profit_ranking.csv",
    index=False
)

# ==========================================
# 11. Final Summary
# ==========================================

print("\n" + "=" * 60)
print("SALES PERFORMANCE SUMMARY")
print("=" * 60)

print("Total Records:", len(df))
print("Top Salesperson:", top_salesperson)
print("Best-Performing Region:", best_region)
print("Best-Selling Category:", best_category)

print("\nAnalysis files generated successfully!")