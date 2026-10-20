# ============================================================
# STUDENT PASS/FAIL DECISION TREE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. CREATE THE DATASET (500 STUDENT RECORDS)
# ============================================================

np.random.seed(42)

number_of_students = 500

# Generate student data
study_hours = np.random.uniform(0, 10, number_of_students)

attendance = np.random.uniform(40, 100, number_of_students)

assignment_score = np.random.uniform(0, 100, number_of_students)

previous_score = np.random.uniform(0, 100, number_of_students)

# Generate Pass/Fail labels using a weighted performance score
performance_score = (
    study_hours * 4
    + attendance * 0.25
    + assignment_score * 0.25
    + previous_score * 0.20
)

# Pass = 1, Fail = 0
pass_fail = (performance_score >= 50).astype(int)

# Create a Pandas DataFrame
data = pd.DataFrame({
    "Study Hours": np.round(study_hours, 2),
    "Attendance": np.round(attendance, 2),
    "Assignment Score": np.round(assignment_score, 2),
    "Previous Score": np.round(previous_score, 2),
    "Pass": pass_fail
})

# Save the dataset to a CSV file
data.to_csv("student_pass_fail.csv", index=False)

print("Dataset created successfully!")
print("File saved as student_pass_fail.csv")


# ============================================================
# 2. EXPLORE THE DATASET
# ============================================================

print("\n========== DATASET EXPLORATION ==========")

# Display first five records
print("\nFirst five student records:")
print(data.head())

# Display last five records
print("\nLast five student records:")
print(data.tail())

# Display dataset dimensions
print("\nDataset shape:")
print(data.shape)

# Display column names
print("\nColumn names:")
print(data.columns)

# Display data types
print("\nData types:")
print(data.dtypes)

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check duplicate records
print("\nDuplicate records:")
print(data.duplicated().sum())

# Statistical summary
print("\nStatistical summary:")
print(data.describe())

# Count Pass and Fail students
print("\nPass/Fail distribution:")
print(data["Pass"].value_counts().sort_index())


# ============================================================
# 3. IDENTIFY FEATURES AND TARGET
# ============================================================

# Features: input variables
X = data[
    [
        "Study Hours",
        "Attendance",
        "Assignment Score",
        "Previous Score"
    ]
]

# Target: output variable
y = data["Pass"]

print("\n========== FEATURES AND TARGET ==========")

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature names:")
print(X.columns.tolist())

print("\nTarget values:")
print("0 = Fail")
print("1 = Pass")


# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLITTING ==========")

print("Total records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 5. CREATE AND TRAIN THE DECISION TREE CLASSIFIER
# ============================================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\n========== PREDICTIONS ==========")

# Display actual and predicted values
results = pd.DataFrame({
    "Actual Pass/Fail": y_test.values,
    "Predicted Pass/Fail": y_pred
})

print(results.head(10))

# Save predictions
results.to_csv("student_predictions.csv", index=False)

print("\nPredictions saved to student_predictions.csv")


# ============================================================
# 7. CALCULATE MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)


# ============================================================
# 8. VISUALIZE THE TRAINED DECISION TREE
# ============================================================

plt.figure(figsize=(22, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Fail (0)", "Pass (1)"],
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=9
)

plt.title("Student Pass/Fail Decision Tree")

plt.tight_layout()

# Save the visualization
plt.savefig("decision_tree.png", dpi=200, bbox_inches="tight")

plt.show()


# ============================================================
# 9. DISPLAY THE DECISION TREE RULES
# ============================================================

print("\n========== DECISION TREE RULES ==========")

tree_rules = export_text(
    model,
    feature_names=list(X.columns)
)

print(tree_rules)


# ============================================================
# 10. BONUS: MANUALLY TRACE FIVE STUDENTS THROUGH THE TREE
# ============================================================

print("\n========== FIVE STUDENT DECISION PATHS ==========")

# Select five test students
sample_students = X_test.head(5)

# Get the path through the tree for each student
decision_paths = model.decision_path(sample_students)

# Find the final leaf node for each student
leaf_nodes = model.apply(sample_students)

# Get predictions
sample_predictions = model.predict(sample_students)

for i, (index, student) in enumerate(sample_students.iterrows()):

    print(f"\nStudent {i + 1}")
    print("-" * 40)

    print("Study Hours:", student["Study Hours"])
    print("Attendance:", student["Attendance"])
    print("Assignment Score:", student["Assignment Score"])
    print("Previous Score:", student["Previous Score"])

    # Get nodes visited by this student
    start = decision_paths.indptr[i]
    end = decision_paths.indptr[i + 1]

    nodes = decision_paths.indices[start:end]

    print("\nDecision path:")

    for node in nodes:

        # Check whether this node is a leaf
        if node == leaf_nodes[i]:

            if sample_predictions[i] == 1:
                print(f"Leaf Node {node}: PASS")
            else:
                print(f"Leaf Node {node}: FAIL")

        else:

            feature_index = model.tree_.feature[node]

            threshold = model.tree_.threshold[node]

            feature_name = X.columns[feature_index]

            feature_value = student[feature_name]

            if feature_value <= threshold:

                print(
                    f"{feature_name} = {feature_value:.2f} "
                    f"<= {threshold:.2f} -> YES"
                )

            else:

                print(
                    f"{feature_name} = {feature_value:.2f} "
                    f"> {threshold:.2f} -> NO"
                )

    print(
        "Final Prediction:",
        "PASS" if sample_predictions[i] == 1 else "FAIL"
    )


# ============================================================
# 11. EXPLAIN THE DECISION TREE
# ============================================================

print("\n========== TREE EXPLANATION ==========")

# The root is node 0
root_feature_index = model.tree_.feature[0]

root_feature_name = X.columns[root_feature_index]

root_threshold = model.tree_.threshold[0]

print("\nRoot Node:")
print(f"First question: Is {root_feature_name} <= {root_threshold:.2f}?")

print("\nDecision Nodes:")
print("Each decision node asks a question about a feature and threshold.")

print("\nBranches:")
print("Branches represent the outcomes of each question.")

print("\nLeaf Nodes:")
print("Leaf nodes provide the final prediction: Pass (1) or Fail (0).")

print("\nHow a prediction is made:")
print("1. Start at the root node.")
print("2. Check the student's feature value.")
print("3. Follow the matching branch.")
print("4. Continue until reaching a leaf node.")
print("5. The leaf node gives the Pass/Fail prediction.")


# ============================================================
# 12. FINISH
# ============================================================

print("\n========== PRACTICAL COMPLETED ==========")

print("Dataset: student_pass_fail.csv")
print("Predictions: student_predictions.csv")
print("Tree visualization: decision_tree.png")

print("\nAll tasks completed successfully!")