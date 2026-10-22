# ============================================================
# DAY 36: LOAN APPROVAL DECISION TREE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. CREATE THE DATASET (1,000 LOAN APPLICATIONS)
# ============================================================

np.random.seed(42)

number_of_applications = 1000

# Generate loan application data
age = np.random.randint(18, 66, number_of_applications)

income = np.random.randint(20000, 250001, number_of_applications)

credit_score = np.random.randint(300, 851, number_of_applications)

loan_amount = np.random.randint(5000, 200001, number_of_applications)

employment_years = np.random.randint(0, 41, number_of_applications)

existing_loans = np.random.randint(0, 6, number_of_applications)


# Generate loan approval labels using a simple decision rule.
# This is synthetic data for educational purposes.

loan_approved = (
    (credit_score >= 650)
    & (income >= loan_amount * 0.25)
    & (employment_years >= 2)
    & (existing_loans <= 3)
).astype(int)


# Create a Pandas DataFrame
data = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Credit Score": credit_score,
    "Loan Amount": loan_amount,
    "Employment Years": employment_years,
    "Existing Loans": existing_loans,
    "Loan Approved": loan_approved
})


# Save the dataset
data.to_csv("loan_approval_dataset.csv", index=False)

print("Dataset created successfully!")
print("Total loan applications:", len(data))
print("Saved as loan_approval_dataset.csv")


# ============================================================
# 2. LOAD AND INSPECT THE DATASET
# ============================================================

# Load the CSV file using Pandas
data = pd.read_csv("loan_approval_dataset.csv")

print("\n========== DATASET INSPECTION ==========")

# First five records
print("\nFirst five records:")
print(data.head())

# Dataset dimensions
print("\nDataset shape:")
print(data.shape)

# Column names
print("\nColumns:")
print(data.columns.tolist())

# Data types
print("\nData types:")
print(data.dtypes)

# Statistical summary
print("\nStatistical summary:")
print(data.describe())

# Missing values
print("\nMissing values:")
print(data.isnull().sum())

# Duplicate records
print("\nDuplicate records:")
print(data.duplicated().sum())

# Loan approval distribution
print("\nLoan approval distribution:")
print(data["Loan Approved"].value_counts().sort_index())


# ============================================================
# 3. CLEAN THE DATASET
# ============================================================

print("\n========== DATA CLEANING ==========")

# Remove duplicate records
data = data.drop_duplicates()

# Handle missing numerical values
numeric_columns = data.select_dtypes(include=np.number).columns

data[numeric_columns] = data[numeric_columns].fillna(
    data[numeric_columns].median()
)

# Remove records containing invalid values
data = data[
    (data["Age"].between(18, 100))
    & (data["Income"] > 0)
    & (data["Credit Score"].between(300, 850))
    & (data["Loan Amount"] > 0)
    & (data["Employment Years"] >= 0)
    & (data["Existing Loans"] >= 0)
    & (data["Loan Approved"].isin([0, 1]))
].copy()

print("Dataset cleaned successfully!")
print("Records after cleaning:", len(data))

print("\nMissing values after cleaning:")
print(data.isnull().sum())


# ============================================================
# 4. SEPARATE FEATURES (X) AND TARGET (y)
# ============================================================

X = data[
    [
        "Age",
        "Income",
        "Credit Score",
        "Loan Amount",
        "Employment Years",
        "Existing Loans"
    ]
]

y = data["Loan Approved"]

print("\n========== FEATURES AND TARGET ==========")

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nTarget meaning:")
print("1 = Loan Approved")
print("0 = Loan Rejected")


# ============================================================
# 5. SPLIT INTO TRAINING AND TESTING DATA
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
# 6. TRAIN THE DECISION TREE CLASSIFIER
# ============================================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nDecision Tree trained successfully!")


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\n========== LOAN APPROVAL PREDICTIONS ==========")

results = pd.DataFrame({
    "Actual Approval": y_test.to_numpy(),
    "Predicted Approval": y_pred
})

print(results.head(10))

# Save predictions
results.to_csv("loan_approval_predictions.csv", index=False)

print("\nPredictions saved to loan_approval_predictions.csv")


# ============================================================
# 8. CALCULATE ACCURACY
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
        labels=[0, 1],
        target_names=["Rejected", "Approved"],
        zero_division=0
    )
)


# ============================================================
# 9. VISUALIZE THE TRAINED DECISION TREE
# ============================================================

plt.figure(figsize=(24, 14))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Rejected (0)", "Approved (1)"],
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=9
)

plt.title("Loan Approval Decision Tree")

plt.tight_layout()

# Save visualization
plt.savefig(
    "loan_approval_decision_tree.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. DISPLAY DECISION TREE RULES
# ============================================================

print("\n========== DECISION TREE RULES ==========")

tree_rules = export_text(
    model,
    feature_names=list(X.columns)
)

print(tree_rules)

# Save rules to a text file
with open(
    "loan_approval_tree_rules.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(tree_rules)


# ============================================================
# 11. IDENTIFY THE ROOT FEATURE
# ============================================================

root_feature_index = model.tree_.feature[0]

root_feature = X.columns[root_feature_index]

root_threshold = model.tree_.threshold[0]

print("\n========== ROOT FEATURE ==========")

print("Root feature:", root_feature)

print(
    f"Root decision: Is {root_feature} <= {root_threshold:.2f}?"
)


# ============================================================
# 12. IDENTIFY IMPORTANT FEATURES
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance - Loan Approval")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 13. IDENTIFY THE FINAL LEAF PREDICTIONS
# ============================================================

print("\n========== FINAL LEAF PREDICTIONS ==========")

tree = model.tree_

for node_id in range(tree.node_count):

    # A leaf node has no left child
    if tree.children_left[node_id] == tree.children_right[node_id]:

        class_index = np.argmax(tree.value[node_id][0])

        if class_index == 1:
            prediction = "APPROVED"
        else:
            prediction = "REJECTED"

        print(f"Leaf Node {node_id}: {prediction}")


# ============================================================
# 14. DISPLAY FEATURES USED NEAR THE ROOT
# ============================================================

print("\n========== FEATURES USED EARLY IN THE TREE ==========")

# Display decision features in node order
seen_features = []

for node_id in range(tree.node_count):

    if tree.children_left[node_id] != tree.children_right[node_id]:

        feature_index = tree.feature[node_id]
        feature_name = X.columns[feature_index]

        if feature_name not in seen_features:
            seen_features.append(feature_name)

        print(
            f"Node {node_id}: {feature_name} "
            f"<= {tree.threshold[node_id]:.2f}"
        )

print("\nUnique features used in tree traversal order:")
print(seen_features)


# ============================================================
# 15. PRACTICAL SUMMARY
# ============================================================

print("\n========== PRACTICAL SUMMARY ==========")

print("1. Dataset created with 1,000 loan applications.")
print("2. Dataset loaded, inspected, and cleaned.")
print("3. Features and target separated.")
print("4. Data split into training and testing sets.")
print("5. Decision Tree Classifier trained.")
print(f"6. Model accuracy: {accuracy * 100:.2f}%")
print("7. Decision Tree visualized.")
print("8. Root feature identified.")
print("9. Decision rules and leaf predictions displayed.")
print("10. Important features analyzed.")

print("\nDAY 36 PRACTICAL COMPLETED!")