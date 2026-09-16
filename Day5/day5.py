import numpy as np


# -----------------------------------------
# Create Sales Dataset
# -----------------------------------------
def create_dataset():

    sales_data = np.array([
        [101, 120, 15000, 3000],
        [102, 95, 12000, 2400],
        [103, 180, 22500, 5000],
        [104, 75, 9000, 1500],
        [105, 210, 28000, 6500],
        [106, 130, 16500, 3200],
        [107, 160, 20000, 4200],
        [108, 90, 11000, 1800],
        [109, 250, 32000, 8000],
        [110, 145, 18500, 3500],
        [111, 110, 14000, 2700],
        [112, 195, 25000, 5500],
        [113, 85, 10500, 1700],
        [114, 170, 21500, 4600],
        [115, 220, 29000, 7000],
        [116, 105, 13500, 2500],
        [117, 155, 19500, 4000],
        [118, 65, 8000, 1200],
        [119, 200, 26000, 5800],
        [120, 135, 17500, 3300]
    ])

    return sales_data


# -----------------------------------------
# Display Dataset
# -----------------------------------------
def display_dataset(data):

    print("\n========== SALES DATASET ==========")

    print(
        "Product ID | Monthly Sales | Revenue | Profit"
    )

    print("-" * 50)

    for row in data:
        print(
            f"{int(row[0]):10} | "
            f"{int(row[1]):13} | "
            f"{row[2]:7.2f} | "
            f"{row[3]:6.2f}"
        )


# -----------------------------------------
# Total Revenue
# -----------------------------------------
def calculate_total_revenue(data):

    revenue = data[:, 2]

    return np.sum(revenue)


# -----------------------------------------
# Average Revenue
# -----------------------------------------
def calculate_average_revenue(data):

    revenue = data[:, 2]

    return np.mean(revenue)


# -----------------------------------------
# Highest Sales
# -----------------------------------------
def calculate_highest_sales(data):

    sales = data[:, 1]

    return np.max(sales)


# -----------------------------------------
# Lowest Sales
# -----------------------------------------
def calculate_lowest_sales(data):

    sales = data[:, 1]

    return np.min(sales)


# -----------------------------------------
# Profit Percentage
# -----------------------------------------
def calculate_profit_percentage(data):

    total_revenue = np.sum(data[:, 2])
    total_profit = np.sum(data[:, 3])

    profit_percentage = (total_profit / total_revenue) * 100

    return profit_percentage


# -----------------------------------------
# Best Selling Product
# -----------------------------------------
def find_best_product(data):

    sales = data[:, 1]

    index = np.argmax(sales)

    product_id = data[index, 0]

    return int(product_id)


# -----------------------------------------
# Worst Selling Product
# -----------------------------------------
def find_worst_product(data):

    sales = data[:, 1]

    index = np.argmin(sales)

    product_id = data[index, 0]

    return int(product_id)


# -----------------------------------------
# Top 5 Products Based on Revenue
# -----------------------------------------
def top_5_products(data):

    revenue = data[:, 2]

    # Get indexes sorted by revenue
    sorted_indexes = np.argsort(revenue)[::-1]

    top_indexes = sorted_indexes[:5]

    print("\n========== TOP 5 PRODUCTS BY REVENUE ==========")

    print("Rank | Product ID | Revenue")
    print("-" * 35)

    for rank, index in enumerate(top_indexes, start=1):

        product_id = int(data[index, 0])
        product_revenue = data[index, 2]

        print(
            f"{rank:4} | "
            f"{product_id:10} | "
            f"{product_revenue:,.2f}"
        )


# -----------------------------------------
# Display Analytics
# -----------------------------------------
def display_analytics(data):

    total_revenue = calculate_total_revenue(data)
    average_revenue = calculate_average_revenue(data)

    highest_sales = calculate_highest_sales(data)
    lowest_sales = calculate_lowest_sales(data)

    profit_percentage = calculate_profit_percentage(data)

    best_product = find_best_product(data)
    worst_product = find_worst_product(data)

    print("\n==========================================")
    print("           SALES ANALYTICS")
    print("==========================================")

    print(f"Total Revenue      : ${total_revenue:,.2f}")
    print(f"Average Revenue    : ${average_revenue:,.2f}")
    print(f"Highest Sales      : {highest_sales:.0f}")
    print(f"Lowest Sales       : {lowest_sales:.0f}")

    print(f"Profit Percentage  : {profit_percentage:.2f}%")

    print(f"Best Selling Product  : {best_product}")
    print(f"Worst Selling Product : {worst_product}")


# -----------------------------------------
# Main Function
# -----------------------------------------
def main():

    print("==========================================")
    print("       SALES ANALYTICS DASHBOARD")
    print("==========================================")

    # Create dataset
    data = create_dataset()

    # Display dataset
    display_dataset(data)

    # Display analytics
    display_analytics(data)

    # Bonus feature
    top_5_products(data)


# -----------------------------------------
# Start Program
# -----------------------------------------
if __name__ == "__main__":
    main()