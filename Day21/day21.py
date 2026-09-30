import pandas as pd

# ==========================================
# Day 21 - Business Intelligence Report Generator
# ==========================================

# 1. Load Day 8 sales dataset
df = pd.read_csv("sales_data.csv")

print("=" * 60)
print("BUSINESS INTELLIGENCE REPORT GENERATOR")
print("=" * 60)

print("\nDataset:")
print(df.head())

print("\nTotal Records:", len(df))


# ==========================================
# 2. Regional Report
# ==========================================

regional_report = (
    df.groupby("Region")
    .agg(
        Total_Revenue=("Revenue", "sum"),
        Average_Revenue=("Revenue", "mean"),
        Total_Profit=("Profit", "sum")
    )
    .reset_index()
)

regional_report["Total_Revenue"] = regional_report[
    "Total_Revenue"
].round(2)

regional_report["Average_Revenue"] = regional_report[
    "Average_Revenue"
].round(2)

regional_report["Total_Profit"] = regional_report[
    "Total_Profit"
].round(2)

regional_report = regional_report.sort_values(
    "Total_Revenue",
    ascending=False
)

print("\n" + "=" * 60)
print("REGIONAL REPORT")
print("=" * 60)

print(regional_report)


# ==========================================
# 3. Product Report
# ==========================================

product_report = (
    df.groupby("Product")
    .agg(
        Total_Quantity=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
)

product_report["Revenue"] = product_report[
    "Revenue"
].round(2)

product_report["Profit"] = product_report[
    "Profit"
].round(2)

product_report = product_report.sort_values(
    "Revenue",
    ascending=False
)

print("\n" + "=" * 60)
print("PRODUCT REPORT")
print("=" * 60)

print(product_report)


# ==========================================
# 4. Salesperson Report
# ==========================================

salesperson_report = (
    df.groupby("Salesperson")
    .agg(
        Number_of_Orders=("Order ID", "count"),
        Revenue=("Revenue", "sum")
    )
    .reset_index()
)

# Calculate Average Order Value
salesperson_report["Average_Order_Value"] = (
    salesperson_report["Revenue"] /
    salesperson_report["Number_of_Orders"]
)

salesperson_report["Revenue"] = salesperson_report[
    "Revenue"
].round(2)

salesperson_report["Average_Order_Value"] = (
    salesperson_report["Average_Order_Value"]
    .round(2)
)

salesperson_report = salesperson_report.sort_values(
    "Revenue",
    ascending=False
)

print("\n" + "=" * 60)
print("SALESPERSON REPORT")
print("=" * 60)

print(salesperson_report)


# ==========================================
# 5. Top Region
# ==========================================

top_region = regional_report.iloc[0]["Region"]


# ==========================================
# 6. Top Product
# ==========================================

top_product = product_report.iloc[0]["Product"]


# ==========================================
# 7. Top Salesperson
# ==========================================

top_salesperson = salesperson_report.iloc[0]["Salesperson"]


# ==========================================
# 8. Overall Business Metrics
# ==========================================

total_revenue = df["Revenue"].sum()

total_profit = df["Profit"].sum()

average_order_value = df["Revenue"].mean()


# ==========================================
# 9. Final BI Summary
# ==========================================

print("\n" + "=" * 60)
print("FINAL BUSINESS INTELLIGENCE REPORT")
print("=" * 60)

print("Top Region:", top_region)
print("Top Product:", top_product)
print("Top Salesperson:", top_salesperson)
print("Total Revenue:", round(total_revenue, 2))
print("Total Profit:", round(total_profit, 2))
print("Average Order Value:", round(average_order_value, 2))


# ==========================================
# 10. Create Summary DataFrame
# ==========================================

summary_report = pd.DataFrame({
    "Metric": [
        "Top Region",
        "Top Product",
        "Top Salesperson",
        "Total Revenue",
        "Total Profit",
        "Average Order Value"
    ],
    "Value": [
        top_region,
        top_product,
        top_salesperson,
        round(total_revenue, 2),
        round(total_profit, 2),
        round(average_order_value, 2)
    ]
})

print("\n")
print(summary_report)


# ==========================================
# 11. Export Reports as CSV
# ==========================================

regional_report.to_csv(
    "regional_report.csv",
    index=False
)

product_report.to_csv(
    "product_report.csv",
    index=False
)

salesperson_report.to_csv(
    "salesperson_report.csv",
    index=False
)

summary_report.to_csv(
    "business_summary_report.csv",
    index=False
)


# ==========================================
# 12. Completion Message
# ==========================================

print("\n" + "=" * 60)
print("REPORTS GENERATED SUCCESSFULLY")
print("=" * 60)

print("1. regional_report.csv")
print("2. product_report.csv")
print("3. salesperson_report.csv")
print("4. business_summary_report.csv")