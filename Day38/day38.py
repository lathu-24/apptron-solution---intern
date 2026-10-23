# ============================================================
# DAY 38: DECISION TREE OVERFITTING EXPERIMENT
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD OR CREATE CUSTOMER PURCHASE DATASET
# ============================================================

np.random.seed(42)

dataset_file = "customer_purchase_dataset.csv"

if os.path.exists(dataset_file):

    # Load the Day 37 dataset
    data = pd.read_csv(dataset_file)

    print("Existing Customer Purchase Dataset loaded.")

else:

    # Create 2,000 customer records if the file does not exist
    number_of_customers = 2000

    age = np.random.randint(18, 66, number_of_customers)
    income = np.random.randint(20000, 200001, number_of_customers)
    website_visits = np.random.randint(1, 31, number_of_customers)
    time_spent = np.random.uniform(1, 60, number_of_customers)
    previous_purchases = np.random.randint(0, 21, number_of_customers)
    product_views = np.random.randint(1, 51, number_of_customers)

    purchase_score = (
        website_visits * 1.5
        + time_spent * 0.5
        + previous_purchases * 2
        + product_views * 0.8
        + income / 10000
        - age * 0.1
    )

    purchase_score += np.random.normal(
        0, 15, number_of_customers
    )

    purchased = (purchase_score >= 65).astype(int)

    data = pd.DataFrame({
        "Age": age,
        "Income": income,
        "Website Visits": website_visits,
        "Time Spent": np.round(time_spent, 2),
        "Previous Purchases": previous_purchases,
        "Product Views": product_views,
        "Purchased": purchased
    })

    data.to_csv(dataset_file, index=False)

    print("New Customer Purchase Dataset created.")


# ============================================================
# 2. EXPLORE THE DATASET
# ============================================================

print("\n========== DATASET EXPLORATION ==========")

print("\nFirst five records:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate records:")
print(data.duplicated().sum())


# ============================================================
# 3. CLEAN THE DATASET
# ============================================================

# Remove duplicate records
data = data.drop_duplicates().copy()

# Fill missing numeric values with their median
numeric_columns = data.select_dtypes(include=np.number).columns

data[numeric_columns] = data[numeric_columns].fillna(
    data[numeric_columns].median()
)

# Validate the target values
data = data[data["Purchased"].isin([0, 1])].copy()

print("\nRecords after cleaning:", len(data))


# ============================================================
# 4. IDENTIFY FEATURES AND TARGET
# ============================================================

feature_columns = [
    "Age",
    "Income",
    "Website Visits",
    "Time Spent",
    "Previous Purchases",
    "Product Views"
]

X = data[feature_columns]
y = data["Purchased"]

print("\nFeatures:")
print(feature_columns)

print("\nTarget:")
print("Purchased = 1")
print("Not Purchased = 0")


# ============================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN/TEST SPLIT ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. TRAIN DECISION TREES WITH DIFFERENT DEPTHS
# ============================================================

depth_values = [1, 2, 3, 5, 8, 12, 15]

results = []

trained_models = {}

for depth in depth_values:

    print(f"\nTraining Decision Tree with max_depth = {depth}")

    # Create model
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict training data
    train_predictions = model.predict(X_train)

    # Predict testing data
    test_predictions = model.predict(X_test)

    # Calculate accuracy
    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    # Calculate difference in percentage points
    accuracy_difference = (
        train_accuracy - test_accuracy
    ) * 100

    # Store results
    results.append({
        "Tree Depth": depth,
        "Training Accuracy (%)": train_accuracy * 100,
        "Testing Accuracy (%)": test_accuracy * 100,
        "Accuracy Difference (%)": accuracy_difference
    })

    trained_models[depth] = model


# ============================================================
# 7. BONUS: TRAIN UNRESTRICTED TREE
# ============================================================

unrestricted_model = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

unrestricted_model.fit(X_train, y_train)

unrestricted_train_accuracy = accuracy_score(
    y_train,
    unrestricted_model.predict(X_train)
)

unrestricted_test_accuracy = accuracy_score(
    y_test,
    unrestricted_model.predict(X_test)
)

unrestricted_difference = (
    unrestricted_train_accuracy
    - unrestricted_test_accuracy
) * 100

results.append({
    "Tree Depth": "None",
    "Training Accuracy (%)": unrestricted_train_accuracy * 100,
    "Testing Accuracy (%)": unrestricted_test_accuracy * 100,
    "Accuracy Difference (%)": unrestricted_difference
})


# ============================================================
# 8. DISPLAY AND SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n========== OVERFITTING EXPERIMENT RESULTS ==========")

print(
    results_df.round(2).to_string(index=False)
)

results_df.to_csv(
    "day38_overfitting_results.csv",
    index=False
)

print("\nResults saved to day38_overfitting_results.csv")


# ============================================================
# 9. GRAPH: TREE DEPTH VS TRAINING ACCURACY
# ============================================================

depth_results = results_df[
    results_df["Tree Depth"] != "None"
].copy()

depth_results["Tree Depth"] = depth_results[
    "Tree Depth"
].astype(int)

depth_results = depth_results.sort_values("Tree Depth")

plt.figure(figsize=(10, 6))

plt.plot(
    depth_results["Tree Depth"],
    depth_results["Training Accuracy (%)"],
    marker="o"
)

plt.title("Tree Depth vs Training Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Training Accuracy (%)")
plt.xticks(depth_values)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "day38_training_accuracy.png",
    dpi=200
)

plt.show()


# ============================================================
# 10. GRAPH: TREE DEPTH VS TESTING ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    depth_results["Tree Depth"],
    depth_results["Testing Accuracy (%)"],
    marker="o"
)

plt.title("Tree Depth vs Testing Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Testing Accuracy (%)")
plt.xticks(depth_values)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "day38_testing_accuracy.png",
    dpi=200
)

plt.show()


# ============================================================
# 11. GRAPH: TRAINING VS TESTING ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    depth_results["Tree Depth"],
    depth_results["Training Accuracy (%)"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depth_results["Tree Depth"],
    depth_results["Testing Accuracy (%)"],
    marker="o",
    label="Testing Accuracy"
)

plt.title("Training vs Testing Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy (%)")
plt.xticks(depth_values)
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "day38_training_vs_testing.png",
    dpi=200
)

plt.show()


# ============================================================
# 12. IDENTIFY THE BEST TREE DEPTH
# ============================================================

best_row = depth_results.loc[
    depth_results["Testing Accuracy (%)"].idxmax()
]

best_depth = int(best_row["Tree Depth"])
best_test_accuracy = best_row["Testing Accuracy (%)"]

print("\n========== BEST MODEL ==========")

print("Best tree depth:", best_depth)

print(
    f"Best testing accuracy: {best_test_accuracy:.2f}%"
)


# ============================================================
# 13. ANALYZE UNDERFITTING AND OVERFITTING
# ============================================================

print("\n========== MODEL ANALYSIS ==========")

print("\nResults by depth:")

for _, row in depth_results.iterrows():

    depth = int(row["Tree Depth"])
    train_acc = row["Training Accuracy (%)"]
    test_acc = row["Testing Accuracy (%)"]
    difference = row["Accuracy Difference (%)"]

    print(
        f"Depth {depth}: "
        f"Training = {train_acc:.2f}%, "
        f"Testing = {test_acc:.2f}%, "
        f"Difference = {difference:.2f}%"
    )


print("\nUnderfitting explanation:")
print(
    "A shallow tree may underfit when both training "
    "and testing accuracy are relatively low."
)

print("\nBalanced model explanation:")
print(
    "A balanced model has strong testing accuracy "
    "and a reasonably small training-testing gap."
)

print("\nOverfitting explanation:")
print(
    "Overfitting occurs when training accuracy is much "
    "higher than testing accuracy, suggesting the model "
    "may be learning details specific to the training data."
)

print("\nUnrestricted tree comparison:")
print(
    f"Training accuracy: "
    f"{unrestricted_train_accuracy * 100:.2f}%"
)

print(
    f"Testing accuracy: "
    f"{unrestricted_test_accuracy * 100:.2f}%"
)

print(
    f"Accuracy difference: {unrestricted_difference:.2f} "
    "percentage points"
)


# ============================================================
# 14. PRACTICAL COMPLETED
# ============================================================

print("\n========== DAY 38 PRACTICAL COMPLETED ==========")

print("1. Customer dataset loaded or created.")
print("2. Dataset cleaned.")
print("3. Seven controlled tree depths tested.")
print("4. Unrestricted tree tested.")
print("5. Training and testing accuracy compared.")
print("6. Accuracy difference calculated.")
print("7. Graphs generated.")
print("8. Best depth identified using testing accuracy.")