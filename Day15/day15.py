import pandas as pd


# ==================================================
# LOAD CSV
# ==================================================

def load_sales_data(filename):

    try:

        df = pd.read_csv(filename)

        print("\nCSV loaded successfully!")

        return df

    except FileNotFoundError:

        print("\nFile not found.")

        return None


# ==================================================
# DISPLAY DATASET INFORMATION
# ==================================================

def display_information(df):

    print("\n========================================")
    print("        DATASET INFORMATION")
    print("========================================")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nDataset Info:")

    df.info()


# ==================================================
# FIRST RECORDS
# ==================================================

def display_first_records(df):

    print("\n========================================")
    print("        FIRST 5 RECORDS")
    print("========================================")

    print(df.head())


# ==================================================
# LAST RECORDS
# ==================================================

def display_last_records(df):

    print("\n========================================")
    print("        LAST 5 RECORDS")
    print("========================================")

    print(df.tail())


# ==================================================
# TOTAL REVENUE
# ==================================================

def calculate_total_revenue(df):

    total_revenue = df["Revenue"].sum()

    print("\n========================================")
    print("        TOTAL REVENUE")
    print("========================================")

    print(f"Total Revenue: ${total_revenue:,.2f}")


# ==================================================
# HIGHEST VALUE ORDER
# ==================================================

def highest_value_order(df):

    index = df["Revenue"].idxmax()

    order = df.loc[index]

    print("\n========================================")
    print("        HIGHEST-VALUE ORDER")
    print("========================================")

    print(f"Order ID: {order['Order ID']}")
    print(f"Product: {order['Product']}")
    print(f"Category: {order['Category']}")
    print(f"Quantity: {order['Quantity']}")
    print(f"Price: ${order['Price']:,.2f}")
    print(f"Revenue: ${order['Revenue']:,.2f}")
    print(f"City: {order['City']}")
    print(f"Date: {order['Date']}")


# ==================================================
# SAVE ANALYZED CSV
# ==================================================

def save_analyzed_csv(df):

    output_file = "analyzed_sales_data.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("\n========================================")
    print("        FILE SAVED")
    print("========================================")

    print(
        f"Analyzed dataset saved as: {output_file}"
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print("========================================")
    print("         SALES CSV ANALYZER")
    print("========================================")

    # Bonus: User enters filename

    filename = input(
        "\nEnter CSV filename "
        "(default: sales_data.csv): "
    ).strip()

    if filename == "":
        filename = "sales_data.csv"

    # Load data

    df = load_sales_data(filename)

    if df is None:
        return

    # Display information

    display_information(df)

    # First records

    display_first_records(df)

    # Last records

    display_last_records(df)

    # Total revenue

    calculate_total_revenue(df)

    # Highest-value order

    highest_value_order(df)

    # Save analyzed data

    save_analyzed_csv(df)


# ==================================================
# PROGRAM START
# ==================================================

if __name__ == "__main__":
    main()