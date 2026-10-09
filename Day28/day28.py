import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CREATE E-COMMERCE DATASET
# ==========================================

np.random.seed(42)

n = 5000

products = [
    "Laptop", "Phone", "Headphones", "Keyboard",
    "T-Shirt", "Jeans", "Shoes", "Jacket",
    "Chair", "Table", "Lamp", "Bookshelf"
]

categories = {
    "Laptop": "Electronics",
    "Phone": "Electronics",
    "Headphones": "Electronics",
    "Keyboard": "Electronics",
    "T-Shirt": "Clothing",
    "Jeans": "Clothing",
    "Shoes": "Clothing",
    "Jacket": "Clothing",
    "Chair": "Home",
    "Table": "Home",
    "Lamp": "Home",
    "Bookshelf": "Home"
}

product_prices = {
    "Laptop": 150000,
    "Phone": 95000,
    "Headphones": 12000,
    "Keyboard": 8500,
    "T-Shirt": 2500,
    "Jeans": 5500,
    "Shoes": 9000,
    "Jacket": 12000,
    "Chair": 18000,
    "Table": 35000,
    "Lamp": 6500,
    "Bookshelf": 25000
}

product = np.random.choice(products, n)

price = np.array([
    product_prices[item] for item in product
])

# Add variation to product prices
price = np.round(
    price * np.random.uniform(0.90, 1.10, n),
    2
)

quantity = np.random.randint(1, 6, n)

discount = np.random.choice(
    [0, 5, 10, 15, 20, 25],
    n,
    p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05]
)

revenue = np.round(
    price * quantity * (1 - discount / 100),
    2
)

df = pd.DataFrame({
    "Customer Age": np.random.randint(18, 71, n),

    "Gender": np.random.choice(
        ["Male", "Female"],
        n
    ),

    "Product": product,

    "Category": [
        categories[item] for item in product
    ],

    "Price": price,

    "Quantity": quantity,

    "Revenue": revenue,

    "Discount": discount,

    "Rating": np.round(
        np.random.uniform(1, 5, n),
        1
    ),

    "City": np.random.choice(
        [
            "Colombo",
            "Jaffna",
            "Kandy",
            "Galle",
            "Negombo",
            "Batticaloa",
            "Matara"
        ],
        n
    )
})


# ==========================================
# 2. BASIC EXPLORATORY DATA ANALYSIS
# ==========================================

print("\nECOMMERCE DATASET")
print("=" * 60)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nTotal Revenue:")
print(f"{df['Revenue'].sum():,.2f}")

print("\nAverage Purchase Revenue:")
print(f"{df['Revenue'].mean():,.2f}")


# ==========================================
# 3. REVENUE DISTRIBUTION
# Histogram
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Revenue",
    bins=30,
    kde=True
)

plt.title("E-Commerce Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Transaction Count")

plt.tight_layout()
plt.savefig("revenue_distribution.png", dpi=300)
plt.show()


# ==========================================
# 4. PRODUCT / CATEGORY COMPARISON
# Bar Chart
# ==========================================

revenue_by_category = (
    df.groupby("Category")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))

sns.barplot(
    x=revenue_by_category.index,
    y=revenue_by_category.values
)

plt.title("Total Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")

plt.tight_layout()
plt.savefig("revenue_by_category.png", dpi=300)
plt.show()


# Product-level revenue comparison
revenue_by_product = (
    df.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=revenue_by_product.index,
    y=revenue_by_product.values
)

plt.title("Total Revenue by Product")
plt.xlabel("Product")
plt.ylabel("Total Revenue")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("revenue_by_product.png", dpi=300)
plt.show()


# ==========================================
# 5. PRICE VS RATING
# Scatter Plot
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Price",
    y="Rating",
    hue="Category",
    alpha=0.6
)

plt.title("Product Price vs Customer Rating")
plt.xlabel("Price")
plt.ylabel("Customer Rating")
plt.legend(title="Category")

plt.tight_layout()
plt.savefig("price_vs_rating.png", dpi=300)
plt.show()


# ==========================================
# 6. REVENUE BOX PLOT
# ==========================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    y="Revenue"
)

plt.title("Revenue Distribution and Outliers")
plt.ylabel("Revenue")

plt.tight_layout()
plt.savefig("revenue_boxplot.png", dpi=300)
plt.show()


# ==========================================
# 7. CUSTOMER AGE DISTRIBUTION
# Histogram
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Customer Age",
    bins=15,
    kde=True
)

plt.title("Customer Age Distribution")
plt.xlabel("Customer Age")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.savefig("customer_age_distribution.png", dpi=300)
plt.show()


# ==========================================
# 8. CATEGORY COUNT PLOT
# ==========================================

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="Category",
    order=df["Category"].value_counts().index
)

plt.title("Number of Transactions by Category")
plt.xlabel("Category")
plt.ylabel("Transaction Count")

plt.tight_layout()
plt.savefig("category_countplot.png", dpi=300)
plt.show()


# ==========================================
# 9. NUMERICAL CORRELATION HEATMAP
# ==========================================

numerical_columns = [
    "Customer Age",
    "Price",
    "Quantity",
    "Revenue",
    "Discount",
    "Rating"
]

correlation_matrix = df[numerical_columns].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("E-Commerce Numerical Feature Correlation")

plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()


# ==========================================
# 10. PAIR PLOT
# ==========================================

pairplot_columns = [
    "Price",
    "Quantity",
    "Revenue",
    "Rating"
]

pair_plot = sns.pairplot(
    df[pairplot_columns].sample(
        n=500,
        random_state=42
    ),
    diag_kind="hist"
)

pair_plot.fig.suptitle(
    "Pair Plot of E-Commerce Numerical Features",
    y=1.02
)

pair_plot.savefig(
    "ecommerce_pairplot.png",
    dpi=300
)

plt.show()


# ==========================================
# 11. BUSINESS ANALYSIS
# ==========================================

print("\nBUSINESS ANALYSIS")
print("=" * 60)

# Highest revenue category
best_category = revenue_by_category.idxmax()

print(f"Highest Revenue Category: {best_category}")

print(
    f"Category Revenue: "
    f"{revenue_by_category.max():,.2f}"
)

# Highest revenue product
best_product = revenue_by_product.idxmax()

print(f"\nHighest Revenue Product: {best_product}")

print(
    f"Product Revenue: "
    f"{revenue_by_product.max():,.2f}"
)

# Most frequently purchased category
most_popular_category = df["Category"].mode()[0]

print(
    f"\nMost Frequent Category: "
    f"{most_popular_category}"
)

# Highest-rated category
average_rating = (
    df.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

print(
    f"\nHighest Average Rating Category: "
    f"{average_rating.idxmax()}"
)

print("\nAverage Rating by Category:")
print(average_rating.round(2))

# Highest revenue city
revenue_by_city = (
    df.groupby("City")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print(
    f"\nHighest Revenue City: "
    f"{revenue_by_city.idxmax()}"
)

# Correlation values
print("\nCORRELATION WITH REVENUE:")
print(
    correlation_matrix["Revenue"]
    .sort_values(ascending=False)
    .round(3)
)


# ==========================================
# 12. EXPORT REPORTS
# ==========================================

df.to_csv(
    "ecommerce_transactions.csv",
    index=False
)

revenue_by_category.to_csv(
    "revenue_by_category.csv"
)

revenue_by_product.to_csv(
    "revenue_by_product.csv"
)

correlation_matrix.to_csv(
    "ecommerce_correlation.csv"
)

average_rating.to_csv(
    "average_rating_by_category.csv"
)

print("\nFILES GENERATED")
print("=" * 60)

print("1. ecommerce_transactions.csv")
print("2. revenue_by_category.csv")
print("3. revenue_by_product.csv")
print("4. ecommerce_correlation.csv")
print("5. average_rating_by_category.csv")
print("6. revenue_distribution.png")
print("7. revenue_by_category.png")
print("8. revenue_by_product.png")
print("9. price_vs_rating.png")
print("10. revenue_boxplot.png")
print("11. customer_age_distribution.png")
print("12. category_countplot.png")
print("13. correlation_heatmap.png")
print("14. ecommerce_pairplot.png")

print("\nE-Commerce EDA completed successfully!")