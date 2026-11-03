# ============================================================
# DAY 45 - STUDENT PASS/FAIL PREDICTION
# Practical Task: Build a Model Comparison Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ============================================================
# TASK 1: CREATE A SYNTHETIC STUDENT DATASET
# ============================================================

np.random.seed(42)

number_of_students = 1000

# Generate student information

study_hours = np.random.uniform(1, 10, number_of_students)

attendance = np.random.uniform(40, 100, number_of_students)

assignment_score = np.random.uniform(20, 100, number_of_students)

previous_exam_score = np.random.uniform(15, 100, number_of_students)


# Generate student performance score

performance_score = (
    study_hours * 3
    + attendance * 0.20
    + assignment_score * 0.25
    + previous_exam_score * 0.30
    + np.random.normal(0, 5, number_of_students)
)


# Determine Pass or Fail

results = []

for score in performance_score:

    if score >= 45:
        results.append("Pass")

    else:
        results.append("Fail")


# Create DataFrame

df = pd.DataFrame({

    "Study Hours": study_hours.round(2),

    "Attendance Percentage": attendance.round(2),

    "Assignment Score": assignment_score.round(2),

    "Previous Exam Score": previous_exam_score.round(2),

    "Result": results

})


# Save raw dataset

df.to_csv("student_dataset.csv", index=False)

print("\nStudent dataset created successfully!")


# ============================================================
# TASK 2: LOAD THE DATASET USING PANDAS
# ============================================================

df = pd.read_csv("student_dataset.csv")


print("\n========== DATASET INFORMATION ==========")

print("\nFirst 5 Records:")

print(df.head())


print("\nLast 5 Records:")

print(df.tail())


print("\nDataset Shape:")

print(df.shape)


print("\nColumn Names:")

print(df.columns.tolist())


print("\nData Types:")

print(df.dtypes)


# ============================================================
# TASK 3: CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUE INSPECTION ==========")

print(df.isnull().sum())

total_missing = df.isnull().sum().sum()

print("\nTotal Missing Values:", total_missing)


# ============================================================
# TASK 4: CHECK DUPLICATE RECORDS
# ============================================================

print("\n========== DUPLICATE INSPECTION ==========")

duplicate_count = df.duplicated().sum()

print("Duplicate Records:", duplicate_count)


# ============================================================
# TASK 5: CLEAN THE DATASET
# ============================================================

print("\n========== DATA CLEANING ==========")

# Remove duplicate records

df = df.drop_duplicates()

# Remove rows containing missing values

df = df.dropna()

print("Dataset Shape After Cleaning:", df.shape)

print("\nRemaining Missing Values:")

print(df.isnull().sum())

print("\nRemaining Duplicate Records:")

print(df.duplicated().sum())


# Save cleaned dataset

df.to_csv("clean_student_dataset.csv", index=False)


# ============================================================
# TASK 6: EXPLORE PASS/FAIL CLASS DISTRIBUTION
# ============================================================

print("\n========== CLASS DISTRIBUTION ==========")

class_distribution = df["Result"].value_counts()

print(class_distribution)


print("\nClass Distribution Percentage:")

print(df["Result"].value_counts(normalize=True) * 100)


# ============================================================
# BONUS: VISUALIZE PASS/FAIL DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

plt.bar(
    class_distribution.index,
    class_distribution.values
)

plt.title("Student Pass/Fail Distribution")

plt.xlabel("Student Result")

plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig("pass_fail_distribution.png")

plt.show()


# ============================================================
# TASK 7: SEPARATE FEATURES AND TARGET
# ============================================================

print("\n========== FEATURE AND TARGET PREPARATION ==========")


# Select input features

X = df[
    [
        "Study Hours",
        "Attendance Percentage",
        "Assignment Score",
        "Previous Exam Score"
    ]
]


# Select target variable

y = df["Result"]


print("\nFeature Columns:")

print(X.head())


print("\nTarget Column:")

print(y.head())


print("\nFeature Shape:", X.shape)

print("Target Shape:", y.shape)


# ============================================================
# TASK 8: ENCODE TARGET LABELS
# ============================================================

# Fail = 0
# Pass = 1

y = y.map({

    "Fail": 0,

    "Pass": 1

})


print("\n========== TARGET ENCODING ==========")

print(y.head())

print("\nEncoded Class Distribution:")

print(y.value_counts())


# ============================================================
# TASK 9: CREATE TRAINING AND TESTING SPLIT
# ============================================================

print("\n========== TRAIN / TEST SPLITTING ==========")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nTraining Features Shape:")

print(X_train.shape)


print("\nTesting Features Shape:")

print(X_test.shape)


print("\nTraining Target Shape:")

print(y_train.shape)


print("\nTesting Target Shape:")

print(y_test.shape)


# ============================================================
# TASK 10: SAVE TRAINING AND TESTING DATA
# ============================================================

# Save training features

X_train.to_csv(
    "X_train.csv",
    index=False
)


# Save testing features

X_test.to_csv(
    "X_test.csv",
    index=False
)


# Save training target

y_train.to_csv(
    "y_train.csv",
    index=False
)


# Save testing target

y_test.to_csv(
    "y_test.csv",
    index=False
)


# Save complete training dataset

train_data = X_train.copy()

train_data["Result"] = y_train

train_data.to_csv(
    "training_dataset.csv",
    index=False
)


# Save complete testing dataset

test_data = X_test.copy()

test_data["Result"] = y_test

test_data.to_csv(
    "testing_dataset.csv",
    index=False
)


# ============================================================
# TASK 11: SAVE DATA INSPECTION REPORT
# ============================================================

report = f"""

STUDENT PASS/FAIL PREDICTION
DATASET INSPECTION REPORT
========================================

1. DATASET INFORMATION

Total Records: {len(df)}

Total Features: {len(X.columns)}

Target Variable: Result


2. FEATURE COLUMNS

{X.columns.tolist()}


3. DATA TYPES

{df.dtypes.to_string()}


4. MISSING VALUES

{df.isnull().sum().to_string()}

Total Missing Values: {df.isnull().sum().sum()}


5. DUPLICATE RECORDS

{df.duplicated().sum()}


6. CLASS DISTRIBUTION

{df["Result"].value_counts().to_string()}


7. CLASS DISTRIBUTION PERCENTAGE

{(df["Result"].value_counts(normalize=True) * 100).round(2).to_string()}


8. FEATURE AND TARGET INFORMATION

Features:
Study Hours
Attendance Percentage
Assignment Score
Previous Exam Score

Target:
Result

Fail = 0
Pass = 1


9. TRAINING AND TESTING SPLIT

Training Records: {len(X_train)}

Testing Records: {len(X_test)}

Training Percentage: 80%

Testing Percentage: 20%

Random State: 42


10. WHY THE SAME DATA SPLIT IS IMPORTANT

Using the same training and testing split for every
classification model ensures a fair comparison.

Each model receives the same training records and is
evaluated using the same unseen testing records.

This prevents differences in model performance from
being caused by different data splits.

It makes the comparison more consistent and reproducible.


11. DATASET DISCLAIMER

This dataset contains synthetically generated student
records. It is created for educational and machine
learning practice purposes.

It does not represent real student academic records.

========================================
END OF REPORT

"""


with open(
    "data_inspection_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


print("\n========== DATASET REPORT ==========")

print(report)


# ============================================================
# TASK 12: FINAL SUMMARY
# ============================================================

print("\n========== PROJECT SUMMARY ==========")

print("Dataset Creation: Completed")

print("Data Inspection: Completed")

print("Missing Value Inspection: Completed")

print("Duplicate Inspection: Completed")

print("Data Cleaning: Completed")

print("Class Distribution Analysis: Completed")

print("Feature and Target Separation: Completed")

print("Target Encoding: Completed")

print("Train/Test Split: Completed")

print("Dataset Export: Completed")

print("\nAll tasks completed successfully!")


# ============================================================
# END OF DAY 45
# ============================================================