import pandas as pd


# ==================================================
# CREATE DATASET
# ==================================================

data = {
    "Employee ID": ["E001", "E002", "E003", "E004", "E005",
                    "E006", "E007", "E008", "E009", "E010"],

    "Name": ["Kamal", "Arun", "Siva", "John", "David",
             "Ravi", "Nimal", "Alex", "Kevin", "Daniel"],

    "Age": [24, 28, 32, 26, 30, 35, 27, 29, 31, 25],

    "Department": [
        "AI", "IT", "HR", "AI", "Finance",
        "IT", "Marketing", "AI", "Finance", "IT"
    ],

    "Salary": [
        75000, 85000, 92000, 78000, 95000,
        88000, 70000, 100000, 90000, 82000
    ],

    "Experience": [
        1, 3, 6, 2, 5,
        7, 2, 4, 6, 2
    ],

    "Performance Score": [
        85, 90, 88, 92, 79,
        95, 75, 91, 87, 84
    ]
}


df = pd.DataFrame(data)


# ==================================================
# 1. DISPLAY COMPLETE DATASET
# ==================================================

print("\n===== COMPLETE DATASET =====")

print(df)


# ==================================================
# 2. DISPLAY SELECTED COLUMNS
# ==================================================

print("\n===== SELECTED COLUMNS =====")

print(
    df[
        ["Name", "Department", "Salary"]
    ]
)


# ==================================================
# 3. FIRST 5 EMPLOYEES
# ==================================================

print("\n===== FIRST 5 EMPLOYEES =====")

print(df.head(5))


# ==================================================
# 4. LAST 5 EMPLOYEES
# ==================================================

print("\n===== LAST 5 EMPLOYEES =====")

print(df.tail(5))


# ==================================================
# 5. ADD BONUS COLUMN
# ==================================================

# Bonus = 10% of salary

df["Bonus"] = df["Salary"] * 0.10

print("\n===== AFTER ADDING BONUS =====")

print(df)


# ==================================================
# 6. RENAME SALARY
# ==================================================

df.rename(
    columns={
        "Salary": "Base_Salary"
    },
    inplace=True
)

print("\n===== AFTER RENAMING SALARY =====")

print(df)


# ==================================================
# 7. REMOVE UNNECESSARY COLUMN
# ==================================================

# Remove Age as an example

df.drop(
    columns=["Age"],
    inplace=True
)

print("\n===== AFTER REMOVING AGE =====")

print(df)


# ==================================================
# 8. SELECT ONE EMPLOYEE USING LOC
# ==================================================

print("\n===== EMPLOYEE USING LOC =====")

print(df.loc[2])


# ==================================================
# 9. SELECT ONE EMPLOYEE USING ILOC
# ==================================================

print("\n===== EMPLOYEE USING ILOC =====")

print(df.iloc[4])


# ==================================================
# 10. CALCULATED TOTAL SALARY
# ==================================================

df["Total_Salary"] = (
    df["Base_Salary"] + df["Bonus"]
)

print("\n===== FINAL DATASET =====")

print(df)


# ==================================================
# MAIN INFORMATION
# ==================================================

print("\n===== DATASET INFORMATION =====")

print(f"Number of employees: {len(df)}")

print(f"Number of columns: {len(df.columns)}")

print("\nColumn Names:")

print(df.columns.tolist())