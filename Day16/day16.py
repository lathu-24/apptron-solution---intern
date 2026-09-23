import pandas as pd
import numpy as np


# ==================================================
# 1. CREATE DATASET WITH INTENTIONAL PROBLEMS
# ==================================================

data = {
    "Patient ID": [
        "P001", "P002", "P003", "P004", "P005",
        "P006", "P007", "P008", "P009", "P010",
        "P005",  # Duplicate patient
        "P012"
    ],

    "Age": [
        25, 34, 45, 150, 29,
        np.nan, 52, -5, 41, 63,
        29, "thirty"  # Incorrect age values/types
    ],

    "Gender": [
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female"
    ],

    "Blood Pressure": [
        120, 130, 125, 140, np.nan,
        118, 135, 128, 122, 145,
        np.nan, 130
    ],

    "Glucose": [
        95, 110, 105, 130, 100,
        np.nan, 115, 108, 98, 140,
        100, 120
    ],

    "Weight": [
        65, 70, 75, 80, 68,
        72, np.nan, 60, 77, 85,
        68, 74
    ],

    "Appointment Date": [
        "2026-01-10",
        "2026-01-12",
        "2026-01-15",
        "2026-01-18",
        "2026-01-20",
        "2026-01-22",
        "2026-01-25",
        "2026-01-28",
        "2026-02-01",
        "2026-02-05",
        "2026-01-20",
        "wrong-date"  # Wrong data
    ]
}


df = pd.DataFrame(data)


# ==================================================
# 2. DISPLAY DATASET
# ==================================================

print("=" * 70)
print("             HEALTHCARE DATASET AUDITOR")
print("=" * 70)

print("\n===== ORIGINAL DATASET =====")

print(df)


# ==================================================
# 3. DATASET SHAPE
# ==================================================

print("\n===== DATASET SHAPE =====")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Shape: {df.shape}")


# ==================================================
# 4. DATA TYPES
# ==================================================

print("\n===== DATA TYPES =====")

print(df.dtypes)


# ==================================================
# 5. MISSING VALUES
# ==================================================

print("\n===== MISSING VALUES =====")

missing_values = df.isnull().sum()

print(missing_values)

print(
    f"\nTotal missing values: "
    f"{missing_values.sum()}"
)


# ==================================================
# 6. DUPLICATE PATIENTS
# ==================================================

print("\n===== DUPLICATE PATIENTS =====")

duplicate_rows = df.duplicated(
    subset=["Patient ID"]
)

duplicate_count = duplicate_rows.sum()

print(
    f"Duplicate patient count: "
    f"{duplicate_count}"
)

if duplicate_count > 0:

    print("\nDuplicate records:")

    print(
        df[duplicate_rows]
    )


# ==================================================
# 7. INVALID AGE VALUES
# ==================================================

print("\n===== INVALID AGE VALUES =====")

# Convert age to numeric.
# Invalid text values become NaN.

numeric_age = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

invalid_age = (
    numeric_age.isna()
    | (numeric_age < 0)
    | (numeric_age > 120)
)

invalid_age_count = invalid_age.sum()

print(
    f"Invalid age records: "
    f"{invalid_age_count}"
)

print("\nInvalid age records:")

print(
    df[invalid_age]
)


# ==================================================
# 8. WRONG DATE VALUES
# ==================================================

print("\n===== INVALID APPOINTMENT DATES =====")

converted_dates = pd.to_datetime(
    df["Appointment Date"],
    errors="coerce"
)

invalid_dates = converted_dates.isna()

invalid_date_count = invalid_dates.sum()

print(
    f"Invalid date records: "
    f"{invalid_date_count}"
)

print("\nInvalid date records:")

print(
    df[invalid_dates]
)


# ==================================================
# 9. INVALID BLOOD PRESSURE
# ==================================================

print("\n===== INVALID BLOOD PRESSURE =====")

invalid_bp = (
    df["Blood Pressure"].notna()
    & (
        (df["Blood Pressure"] <= 0)
        | (df["Blood Pressure"] > 250)
    )
)

print(
    f"Invalid blood pressure records: "
    f"{invalid_bp.sum()}"
)


# ==================================================
# 10. INVALID GLUCOSE
# ==================================================

print("\n===== INVALID GLUCOSE =====")

invalid_glucose = (
    df["Glucose"].notna()
    & (
        (df["Glucose"] <= 0)
        | (df["Glucose"] > 500)
    )
)

print(
    f"Invalid glucose records: "
    f"{invalid_glucose.sum()}"
)


# ==================================================
# 11. INVALID WEIGHT
# ==================================================

print("\n===== INVALID WEIGHT =====")

invalid_weight = (
    df["Weight"].notna()
    & (
        (df["Weight"] <= 0)
        | (df["Weight"] > 300)
    )
)

print(
    f"Invalid weight records: "
    f"{invalid_weight.sum()}"
)


# ==================================================
# 12. DATA QUALITY REPORT
# ==================================================

print("\n")
print("=" * 70)
print("                 DATA QUALITY REPORT")
print("=" * 70)

print(
    f"\nDataset Shape: {df.shape}"
)

print(
    f"Total Records: {len(df)}"
)

print(
    f"Total Columns: {len(df.columns)}"
)

print(
    f"Total Missing Values: "
    f"{missing_values.sum()}"
)

print(
    f"Duplicate Patients: "
    f"{duplicate_count}"
)

print(
    f"Invalid Age Records: "
    f"{invalid_age_count}"
)

print(
    f"Invalid Date Records: "
    f"{invalid_date_count}"
)

print(
    f"Invalid Blood Pressure Records: "
    f"{invalid_bp.sum()}"
)

print(
    f"Invalid Glucose Records: "
    f"{invalid_glucose.sum()}"
)

print(
    f"Invalid Weight Records: "
    f"{invalid_weight.sum()}"
)


# ==================================================
# 13. OVERALL QUALITY STATUS
# ==================================================

total_problems = (
    missing_values.sum()
    + duplicate_count
    + invalid_age_count
    + invalid_date_count
    + invalid_bp.sum()
    + invalid_glucose.sum()
    + invalid_weight.sum()
)

print("\n===== OVERALL DATA QUALITY =====")

if total_problems == 0:

    print("Data Quality: GOOD")

else:

    print("Data Quality: REQUIRES CLEANING")

    print(
        f"Total detected problems: "
        f"{total_problems}"
    )


# ==================================================
# 14. SAVE REPORT
# ==================================================

report = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Missing Values",
        "Duplicate Patients",
        "Invalid Age Records",
        "Invalid Date Records",
        "Invalid Blood Pressure",
        "Invalid Glucose",
        "Invalid Weight",
        "Total Problems"
    ],

    "Value": [
        df.shape[0],
        df.shape[1],
        missing_values.sum(),
        duplicate_count,
        invalid_age_count,
        invalid_date_count,
        invalid_bp.sum(),
        invalid_glucose.sum(),
        invalid_weight.sum(),
        total_problems
    ]
})


report.to_csv(
    "data_quality_report.csv",
    index=False
)


print("\nData quality report saved as:")
print("data_quality_report.csv")