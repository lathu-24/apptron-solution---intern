import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------
# 1. Create Sales Dataset
# --------------------------------

data = {
    "Month": [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ],

    "Revenue": [
        125000, 138000, 152000, 145000,
        168000, 175000, 160000, 182000,
        155000, 195000, 210000, 225000
    ],

    "Orders": [
        420, 455, 490, 470,
        525, 550, 510, 575,
        495, 620, 680, 720
    ],

    "Profit": [
        28000, 31000, 35000, 33000,
        39000, 42000, 37000, 45000,
        36000, 48000, 53000, 58000
    ],

    "Product Category": [
        "Electronics", "Clothing", "Electronics", "Home",
        "Clothing", "Electronics", "Home", "Electronics",
        "Clothing", "Electronics", "Clothing", "Electronics"
    ]
}

df = pd.DataFrame(data)

print("SALES PERFORMANCE DATA")
print("=" * 50)
print(df.to_string(index=False))


# --------------------------------
# 2. Line Chart - Monthly Revenue
# --------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    df["Month"],
    df["Revenue"],
    marker="o",
    label="Revenue"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------
# 3. Bar Chart - Revenue by Category
# --------------------------------

revenue_by_category = (
    df.groupby("Product Category")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))

plt.bar(
    revenue_by_category.index,
    revenue_by_category.values,
    label="Revenue"
)

plt.title("Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------
# 4. Pie Chart - Orders by Category
# --------------------------------

orders_by_category = (
    df.groupby("Product Category")["Orders"]
    .sum()
)

plt.figure(figsize=(7, 7))

plt.pie(
    orders_by_category.values,
    labels=orders_by_category.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Orders by Product Category")
plt.legend(title="Categories")
plt.show()


# --------------------------------
# 5. Highest and Lowest Revenue
# --------------------------------

highest_revenue = df.loc[df["Revenue"].idxmax()]
lowest_revenue = df.loc[df["Revenue"].idxmin()]

print("\nSALES ANALYSIS")
print("=" * 50)

print(
    f"Highest Revenue Month: "
    f"{highest_revenue['Month']} - "
    f"{highest_revenue['Revenue']}"
)

print(
    f"Lowest Revenue Month: "
    f"{lowest_revenue['Month']} - "
    f"{lowest_revenue['Revenue']}"
)


# --------------------------------
# 6. Most Profitable Category
# --------------------------------

profit_by_category = (
    df.groupby("Product Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

most_profitable_category = profit_by_category.idxmax()

print(
    f"Most Profitable Category: "
    f"{most_profitable_category} - "
    f"{profit_by_category.max()}"
)


# --------------------------------
# 7. Category with Highest Orders
# --------------------------------

highest_order_category = orders_by_category.idxmax()

print(
    f"Category with Highest Order Count: "
    f"{highest_order_category} - "
    f"{orders_by_category.max()} orders"
)


# --------------------------------
# 8. Bonus - Revenue vs Profit
# --------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    df["Month"],
    df["Revenue"],
    marker="o",
    label="Revenue"
)

plt.plot(
    df["Month"],
    df["Profit"],
    marker="s",
    label="Profit"
)

plt.title("Monthly Revenue vs Profit")
plt.xlabel("Month")
plt.ylabel("Amount")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------
# 9. Save Dataset and Reports
# --------------------------------

df.to_csv("sales_performance.csv", index=False)

revenue_by_category.to_csv(
    "revenue_by_category.csv"
)

orders_by_category.to_csv(
    "orders_by_category.csv"
)

profit_by_category.to_csv(
    "profit_by_category.csv"
)

print("\nFiles generated:")
print("1. sales_performance.csv")
print("2. revenue_by_category.csv")
print("3. orders_by_category.csv")
print("4. profit_by_category.csv")

print("\nSales performance analysis completed!")