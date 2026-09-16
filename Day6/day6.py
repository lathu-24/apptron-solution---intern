import numpy as np


# -----------------------------------------
# Create Employee Dataset
# -----------------------------------------
def create_dataset():

    employees = np.array([
        [101, 25, 1, 45000, 2, 78],
        [102, 32, 2, 62000, 5, 88],
        [103, 28, 3, -50000, 3, 92],
        [104, 40, 1, 75000, 10, 95],
        [105, 35, 4, 58000, 7, 85],
        [106, 29, 2, 52000, 4, 91],
        [107, 45, 3, 90000, 15, 97],
        [108, 31, 1, -42000, 6, 76],
        [109, 27, 4, 48000, 3, 89],
        [110, 38, 2, 70000, 9, 94],
        [111, 42, 3, 82000, 12, 96],
        [112, 26, 1, 40000, 1, 72],
        [113, 34, 4, 61000, 6, 93],
        [114, 30, 2, 55000, 4, 87],
        [115, 48, 3, 95000, 20, 99]
    ], dtype=float)

    return employees


# -----------------------------------------
# Display Dataset
# -----------------------------------------
def display_dataset(data, title="EMPLOYEE DATASET"):

    print(f"\n========== {title} ==========")

    print(
        "ID | Age | Department | Salary | Experience | Performance"
    )

    print("-" * 65)

    for row in data:
        print(
            f"{int(row[0]):3} | "
            f"{int(row[1]):3} | "
            f"{int(row[2]):10} | "
            f"{row[3]:8.2f} | "
            f"{int(row[4]):10} | "
            f"{row[5]:11.2f}"
        )


# -----------------------------------------
# Replace Negative Salaries
# -----------------------------------------
def clean_negative_salaries(data):

    cleaned_data = data.copy()

    salary_column = cleaned_data[:, 3]

    # Find average of valid salaries
    valid_salaries = salary_column[salary_column >= 0]

    average_salary = np.mean(valid_salaries)

    # Replace negative salaries with average salary
    cleaned_data[:, 3] = np.where(
        cleaned_data[:, 3] < 0,
        average_salary,
        cleaned_data[:, 3]
    )

    return cleaned_data


# -----------------------------------------
# Increase Salary by 10%
# -----------------------------------------
def increase_salary(data):

    updated_data = data.copy()

    updated_data[:, 3] = updated_data[:, 3] * 1.10

    return updated_data


# -----------------------------------------
# Filter Performance > 90
# -----------------------------------------
def high_performers(data):

    result = data[data[:, 5] > 90]

    return result


# -----------------------------------------
# Employees Above Average Salary
# -----------------------------------------
def above_average_salary(data):

    average_salary = np.mean(data[:, 3])

    result = data[data[:, 3] > average_salary]

    print(f"\nAverage Salary: {average_salary:.2f}")

    return result


# -----------------------------------------
# Reshape Dataset
# -----------------------------------------
def reshape_dataset(data):

    rows, columns = data.shape

    reshaped_data = data.reshape(rows, columns, 1)

    return reshaped_data


# -----------------------------------------
# Transpose Dataset
# -----------------------------------------
def transpose_dataset(data):

    return data.T


# -----------------------------------------
# Generate Cleaned Dataset
# -----------------------------------------
def generate_cleaned_dataset(data):

    # Step 1: Fix negative salaries
    cleaned_data = clean_negative_salaries(data)

    # Step 2: Increase salaries by 10%
    cleaned_data = increase_salary(cleaned_data)

    return cleaned_data


# -----------------------------------------
# Main Function
# -----------------------------------------
def main():

    print("==========================================")
    print("     EMPLOYEE SALARY DATA CLEANING")
    print("==========================================")

    # Create original dataset
    employees = create_dataset()

    display_dataset(employees, "ORIGINAL DATASET")

    # -------------------------------------
    # Clean negative salaries
    # -------------------------------------
    cleaned_data = clean_negative_salaries(employees)

    display_dataset(
        cleaned_data,
        "AFTER REPLACING NEGATIVE SALARIES"
    )

    # -------------------------------------
    # Increase salary by 10%
    # -------------------------------------
    updated_data = increase_salary(cleaned_data)

    display_dataset(
        updated_data,
        "AFTER 10% SALARY INCREASE"
    )

    # -------------------------------------
    # Performance > 90
    # -------------------------------------
    high_performance = high_performers(updated_data)

    display_dataset(
        high_performance,
        "EMPLOYEES WITH PERFORMANCE > 90"
    )

    # -------------------------------------
    # Above average salary
    # -------------------------------------
    above_average = above_average_salary(updated_data)

    display_dataset(
        above_average,
        "EMPLOYEES ABOVE AVERAGE SALARY"
    )

    # -------------------------------------
    # Reshape
    # -------------------------------------
    reshaped = reshape_dataset(updated_data)

    print("\n========== RESHAPED DATASET ==========")
    print("Original Shape :", updated_data.shape)
    print("New Shape      :", reshaped.shape)

    # -------------------------------------
    # Transpose
    # -------------------------------------
    transposed = transpose_dataset(updated_data)

    print("\n========== TRANSPOSED DATASET ==========")
    print("Original Shape :", updated_data.shape)
    print("Transposed Shape:", transposed.shape)

    print("\nTransposed Data:")
    print(transposed)

    # -------------------------------------
    # Bonus: Final cleaned dataset
    # -------------------------------------
    final_data = generate_cleaned_dataset(employees)

    print("\n========== FINAL CLEANED DATASET ==========")
    print(final_data)

    print("\nDataset is ready for further AI/ML analysis.")


# -----------------------------------------
# Run Program
# -----------------------------------------
if __name__ == "__main__":
    main()