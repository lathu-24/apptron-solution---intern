import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------
# 1. Create Company Dataset
# --------------------------------

data = {
    "Month": [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ],

    "Revenue": [
        120000, 135000, 142000, 138000,
        155000, 168000, 162000, 178000,
        171000, 195000, 215000, 235000
    ],

    "Profit": [
        28000, 32000, 35000, 33000,
        39000, 43000, 41000, 47000,
        44000, 52000, 59000, 68000
    ],

    "Orders": [
        420, 450, 480, 465,
        510, 550, 535, 580,
        565, 620, 680, 750
    ],

    "Customers": [
        850, 900, 950, 980,
        1050, 1120, 1180, 1250,
        1300, 1380, 1480, 1620
    ],

    "Expenses": [
        92000, 103000, 107000, 105000,
        116000, 125000, 121000, 131000,
        127000, 143000, 156000, 167000
    ]
}

df = pd.DataFrame(data)

print("BUSINESS KPI DATA")
print("=" * 70)
print(df.to_string(index=False))


# --------------------------------
# 2. Find Best-Performing Month
# --------------------------------

# Calculate profit margin
df["Profit Margin"] = (
    df["Profit"] / df["Revenue"] * 100
)

best_month = df.loc[
    df["Profit"].idxmax()
]

print("\nBEST-PERFORMING MONTH")
print("=" * 70)

print(f"Month: {best_month['Month']}")
print(f"Revenue: {best_month['Revenue']}")
print(f"Profit: {best_month['Profit']}")
print(f"Orders: {best_month['Orders']}")
print(f"Customers: {best_month['Customers']}")
print(f"Expenses: {best_month['Expenses']}")


# --------------------------------
# 3. Create Dashboard
# --------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10)
)

fig.suptitle(
    "Business KPI Dashboard",
    fontsize=18,
    fontweight="bold"
)


# --------------------------------
# 4. Revenue Trend
# --------------------------------

axes[0, 0].plot(
    df["Month"],
    df["Revenue"],
    marker="o",
    label="Revenue"
)

axes[0, 0].set_title("Monthly Revenue Trend")
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Revenue")
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True)


# --------------------------------
# 5. Profit Trend
# --------------------------------

axes[0, 1].plot(
    df["Month"],
    df["Profit"],
    marker="o",
    label="Profit"
)

axes[0, 1].set_title("Monthly Profit Trend")
axes[0, 1].set_xlabel("Month")
axes[0, 1].set_ylabel("Profit")
axes[0, 1].tick_params(axis="x", rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True)


# --------------------------------
# 6. Monthly Orders
# --------------------------------

axes[1, 0].bar(
    df["Month"],
    df["Orders"],
    label="Orders"
)

axes[1, 0].set_title("Monthly Orders")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Number of Orders")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].legend()


# --------------------------------
# 7. Customer Growth
# --------------------------------

axes[1, 1].plot(
    df["Month"],
    df["Customers"],
    marker="o",
    label="Customers"
)

axes[1, 1].set_title("Customer Growth")
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylabel("Customers")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True)


# --------------------------------
# 8. Highlight Best Month
# --------------------------------

best_index = df["Profit"].idxmax()

axes[0, 1].scatter(
    df.loc[best_index, "Month"],
    df.loc[best_index, "Profit"],
    s=150,
    label="Best Month"
)

axes[0, 1].legend()


# --------------------------------
# 9. Display Dashboard
# --------------------------------

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.show()


# --------------------------------
# 10. Save Dataset
# --------------------------------

df.to_csv(
    "business_kpi_data.csv",
    index=False
)

print("\nFile generated:")
print("business_kpi_data.csv")

print("\nBusiness KPI Dashboard completed!")