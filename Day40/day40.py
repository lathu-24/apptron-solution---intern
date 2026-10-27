# ============================================================
# Day 40: Student Result Prediction System Using KNN
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------
# 1. Create a dataset using Pandas
# ------------------------------------------------------------

print("=" * 65)
print("STUDENT RESULT PREDICTION SYSTEM USING KNN")
print("=" * 65)

np.random.seed(42)

number_of_students = 50

student_ids = [f"S{i:03d}" for i in range(1, number_of_students + 1)]

# Generate study hours between 1 and 10
study_hours = np.round(
    np.random.uniform(1, 10, number_of_students), 1
)

# Generate attendance percentages between 40 and 100
attendance = np.random.randint(40, 101, number_of_students)

# Generate assignment scores between 20 and 100
assignment_scores = np.random.randint(20, 101, number_of_students)

# Create results based on student performance.
# This makes the synthetic dataset reproducible.
performance_score = (
    study_hours * 5
    + attendance * 0.3
    + assignment_scores * 0.4
)

results = np.where(performance_score >= 52, "Pass", "Fail")

# Create DataFrame
df = pd.DataFrame({
    "Student ID": student_ids,
    "Study Hours": study_hours,
    "Attendance Percentage": attendance,
    "Assignment Score": assignment_scores,
    "Result": results
})

# Save original dataset
df.to_csv("student_results.csv", index=False)

print("\n1. DATASET CREATED")
print("-" * 65)
print(f"Number of students: {len(df)}")
print("\nFirst five records:")
print(df.head())

# ------------------------------------------------------------
# 2. Check missing values and duplicate rows
# ------------------------------------------------------------

print("\n2. DATA QUALITY CHECK")
print("-" * 65)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Remove duplicates and missing values if present
df = df.drop_duplicates()
df = df.dropna()

print("\nDataset shape after cleaning:", df.shape)

# ------------------------------------------------------------
# 3. Select features and encode the target
# ------------------------------------------------------------

print("\n3. FEATURE SELECTION AND LABEL ENCODING")
print("-" * 65)

feature_columns = [
    "Study Hours",
    "Attendance Percentage",
    "Assignment Score"
]

X = df[feature_columns]

# Encode Fail = 0 and Pass = 1
y = df["Result"].map({
    "Fail": 0,
    "Pass": 1
})

print("\nSelected features:")
print(X.head())

print("\nEncoded labels:")
print("Fail = 0")
print("Pass = 1")

print("\nClass distribution:")
print(df["Result"].value_counts())

# ------------------------------------------------------------
# 4. Split the dataset into training and testing sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n4. TRAIN-TEST SPLIT")
print("-" * 65)
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# ------------------------------------------------------------
# 5. Scale the features
# ------------------------------------------------------------

# KNN uses distances, so scaling prevents features with larger
# numeric ranges from dominating the distance calculation.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 6. Train KNN with K = 3
# ------------------------------------------------------------

print("\n5. TRAIN KNN CLASSIFIER (K = 3)")
print("-" * 65)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_scaled, y_train)

# Predict the test set
y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)

# ------------------------------------------------------------
# 7. Display actual and predicted test results
# ------------------------------------------------------------

print("\n6. ACTUAL VS PREDICTED TEST RESULTS")
print("-" * 65)

test_results = pd.DataFrame({
    "Actual Result": y_test.map({0: "Fail", 1: "Pass"}).values,
    "Predicted Result": pd.Series(y_pred, index=y_test.index)
        .map({0: "Fail", 1: "Pass"}).values
})

test_results["Correct Prediction"] = (
    test_results["Actual Result"] == test_results["Predicted Result"]
)

print(test_results.to_string(index=False))

test_results.to_csv("test_prediction_results.csv", index=False)

# ------------------------------------------------------------
# 8. Predict results for five new students
# ------------------------------------------------------------

print("\n7. PREDICT RESULTS FOR FIVE NEW STUDENTS")
print("-" * 65)

new_students = pd.DataFrame({
    "Student ID": ["NEW001", "NEW002", "NEW003", "NEW004", "NEW005"],
    "Study Hours": [8.0, 2.0, 6.0, 3.5, 9.0],
    "Attendance Percentage": [95, 50, 80, 60, 98],
    "Assignment Score": [90, 35, 75, 45, 95]
})

# Select the same features used during training
new_X = new_students[feature_columns]

# Apply the already-fitted scaler
new_X_scaled = scaler.transform(new_X)

# Predict numerical labels
new_predictions = knn.predict(new_X_scaled)

# Convert numerical labels back to text
new_students["Predicted Result"] = pd.Series(
    new_predictions
).map({0: "Fail", 1: "Pass"}).values

print(new_students.to_string(index=False))

new_students.to_csv("new_student_predictions.csv", index=False)

# ------------------------------------------------------------
# 9. Bonus: Compare K = 1, K = 3 and K = 5
# ------------------------------------------------------------

print("\n8. BONUS: COMPARE DIFFERENT K VALUES")
print("-" * 65)

comparison_results = []

for k in [1, 3, 5]:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train_scaled, y_train)

    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    comparison_results.append({
        "K Value": k,
        "Training Accuracy (%)": round(train_accuracy * 100, 2),
        "Testing Accuracy (%)": round(test_accuracy * 100, 2)
    })

comparison_df = pd.DataFrame(comparison_results)

print(comparison_df.to_string(index=False))

comparison_df.to_csv("knn_k_comparison.csv", index=False)

# ------------------------------------------------------------
# 10. Display three observations
# ------------------------------------------------------------

print("\n9. THREE OBSERVATIONS")
print("-" * 65)

pass_count = (df["Result"] == "Pass").sum()
fail_count = (df["Result"] == "Fail").sum()

print(
    f"Observation 1: The dataset contains {pass_count} Pass records "
    f"and {fail_count} Fail records."
)

correct_count = test_results["Correct Prediction"].sum()
total_test = len(test_results)

print(
    f"Observation 2: The K=3 model correctly predicted "
    f"{correct_count} out of {total_test} test records."
)

best_row = comparison_df.loc[
    comparison_df["Testing Accuracy (%)"].idxmax()
]

print(
    f"Observation 3: Among K=1, K=3 and K=5, "
    f"K={int(best_row['K Value'])} achieved the highest testing "
    f"accuracy of {best_row['Testing Accuracy (%)']:.2f}% "
    "on this particular split."
)

# ------------------------------------------------------------
# 11. Explain KNN
# ------------------------------------------------------------

print("\n10. HOW KNN WORKS")
print("-" * 65)

print("""
K-Nearest Neighbors (KNN) is a supervised machine learning algorithm.

1. Each student is represented by three features:
   Study Hours, Attendance Percentage and Assignment Score.

2. The model calculates distances between a new student and
   students in the training dataset.

3. It identifies the K nearest students.

4. The majority result among those neighbors determines the
   predicted result.

5. In this practical, K = 3 means the three nearest students
   vote to predict Pass or Fail.

Feature scaling is used because the three features have
different numeric ranges.

The dataset is synthetic and designed for educational practice.
Predictions are not guaranteed to reflect real student outcomes.
""")

print("=" * 65)
print("STUDENT RESULT PREDICTION SYSTEM COMPLETED")
print("=" * 65)

print("\nFiles created:")
print("1. student_results.csv")
print("2. test_prediction_results.csv")
print("3. new_student_predictions.csv")
print("4. knn_k_comparison.csv")