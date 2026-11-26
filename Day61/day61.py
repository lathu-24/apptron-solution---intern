# Day 60 - Hyperparameter Experiment Lab
# Decision Tree Parameter Experiment

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Load the Breast Cancer Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", X.shape)
print()


# --------------------------------------------------
# 2. Split the Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data Size:", X_train.shape[0])
print("Testing Data Size:", X_test.shape[0])
print()


# --------------------------------------------------
# 3. Experiment with Different max_depth Values
# --------------------------------------------------

max_depth_values = [2, 3, 5, 8, 10, None]

results = []

for depth in max_depth_values:

    # Create Decision Tree Model
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    # Train the Model
    model.fit(X_train, y_train)

    # Make Predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate Accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Store Results
    results.append({
        "max_depth": depth,
        "training_accuracy": train_accuracy,
        "testing_accuracy": test_accuracy
    })


# --------------------------------------------------
# 4. Create Results DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df["training_accuracy"] = (
    results_df["training_accuracy"] * 100
).round(2)

results_df["testing_accuracy"] = (
    results_df["testing_accuracy"] * 100
).round(2)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# --------------------------------------------------
# 5. Identify the Best Model
# --------------------------------------------------

best_model_index = results_df["testing_accuracy"].idxmax()

best_model = results_df.loc[best_model_index]

print("\n========== BEST MODEL ==========")

print("Best max_depth:", best_model["max_depth"])
print("Best Testing Accuracy:", best_model["testing_accuracy"], "%")


# --------------------------------------------------
# 6. Save Results
# --------------------------------------------------

results_df.to_csv("decision_tree_experiment_results.csv", index=False)

print("\nResults saved to decision_tree_experiment_results.csv")