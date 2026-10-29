# ============================================================
# Day 42: Customer Purchase Prediction Experiment Using KNN
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# 1. Create a customer dataset
# ------------------------------------------------------------

print("=" * 70)
print("CUSTOMER PURCHASE PREDICTION EXPERIMENT")
print("=" * 70)

np.random.seed(42)

n_customers = 500

age = np.random.randint(18, 66, n_customers)
monthly_spending = np.random.randint(20, 1001, n_customers)
website_visits = np.random.randint(1, 51, n_customers)
time_on_website = np.round(
    np.random.uniform(1, 60, n_customers), 1
)
previous_purchases = np.random.randint(0, 21, n_customers)

# Generate synthetic purchase labels.
# These rules are only for educational dataset generation.
purchase_score = (
    website_visits * 0.8
    + time_on_website * 0.5
    + previous_purchases * 2.0
    + monthly_spending * 0.015
    - age * 0.1
)

purchase_probability = 1 / (
    1 + np.exp(-(purchase_score - 35) / 8)
)

purchased = np.where(
    np.random.random(n_customers) < purchase_probability,
    "Yes",
    "No"
)

df = pd.DataFrame({
    "Age": age,
    "Monthly Spending": monthly_spending,
    "Website Visits": website_visits,
    "Time on Website": time_on_website,
    "Previous Purchases": previous_purchases,
    "Purchased": purchased
})

# Save original dataset
df.to_csv("customer_purchase_dataset.csv", index=False)

print("\n1. DATASET CREATED")
print("-" * 70)
print(f"Number of customers: {len(df)}")
print("\nFirst five records:")
print(df.head())

# ------------------------------------------------------------
# 2. Inspect and clean the dataset
# ------------------------------------------------------------

print("\n2. DATA INSPECTION")
print("-" * 70)

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicates and rows with missing values
df = df.drop_duplicates().dropna().reset_index(drop=True)

print("\nDataset shape after cleaning:", df.shape)

print("\nPurchase class distribution:")
print(df["Purchased"].value_counts())

# ------------------------------------------------------------
# 3. Separate features and target
# ------------------------------------------------------------

print("\n3. FEATURE SELECTION")
print("-" * 70)

features = [
    "Age",
    "Monthly Spending",
    "Website Visits",
    "Time on Website",
    "Previous Purchases"
]

X = df[features]
y = df["Purchased"]

# ------------------------------------------------------------
# 4. Split into training, validation, and testing sets
# ------------------------------------------------------------

# First, reserve 20% for final testing.
X_train_validation, X_test, y_train_validation, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

# Split the remaining 80% into training and validation sets.
# Final proportions: 60% train, 20% validation, 20% test.
X_train, X_validation, y_train, y_validation = (
    train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=42,
        stratify=y_train_validation
    )
)

print("\n4. DATA SPLIT")
print("-" * 70)
print(f"Training records:   {len(X_train)}")
print(f"Validation records: {len(X_validation)}")
print(f"Testing records:    {len(X_test)}")

# ------------------------------------------------------------
# 5. Scale numerical features
# ------------------------------------------------------------

# Fit scaler using training data only to avoid data leakage.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 6. Train KNN models using different K values
# ------------------------------------------------------------

print("\n5. KNN ACCURACY EXPERIMENT")
print("-" * 70)

k_values = [1, 3, 5, 7, 9, 11]

results = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean",
        weights="uniform"
    )

    model.fit(X_train_scaled, y_train)

    validation_predictions = model.predict(X_validation_scaled)

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    results.append({
        "K": k,
        "Validation Accuracy (%)": round(
            validation_accuracy * 100, 2
        )
    })

accuracy_df = pd.DataFrame(results)

print("\nValidation accuracy comparison:")
print(accuracy_df.to_string(index=False))

accuracy_df.to_csv("knn_accuracy_comparison.csv", index=False)

# ------------------------------------------------------------
# 7. Select the best K using validation accuracy
# ------------------------------------------------------------

best_row = accuracy_df.loc[
    accuracy_df["Validation Accuracy (%)"].idxmax()
]

best_k = int(best_row["K"])
best_validation_accuracy = best_row["Validation Accuracy (%)"]

print("\n6. BEST K VALUE")
print("-" * 70)
print(f"Best K: {best_k}")
print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")

# ------------------------------------------------------------
# 8. Create a line chart
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    accuracy_df["K"],
    accuracy_df["Validation Accuracy (%)"],
    marker="o"
)

plt.xlabel("K Value")
plt.ylabel("Validation Accuracy (%)")
plt.title("KNN Experiment: K Value vs Validation Accuracy")
plt.xticks(k_values)
plt.grid(True)

plt.tight_layout()
plt.savefig("knn_accuracy_chart.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 9. Evaluate the selected K on the test set
# ------------------------------------------------------------

print("\n7. FINAL TEST EVALUATION")
print("-" * 70)

# Combine training and validation data after selecting K.
X_final_train = np.vstack([
    X_train_scaled,
    X_validation_scaled
])

y_final_train = pd.concat([
    y_train,
    y_validation
], ignore_index=True)

final_model = KNeighborsClassifier(
    n_neighbors=best_k,
    metric="euclidean",
    weights="uniform"
)

final_model.fit(X_final_train, y_final_train)

test_predictions = final_model.predict(X_test_scaled)

test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Selected K: {best_k}")
print(f"Final test accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, test_predictions, zero_division=0))

# Save actual and predicted test results
test_results = pd.DataFrame({
    "Actual Purchase": y_test.reset_index(drop=True),
    "Predicted Purchase": test_predictions
})

test_results.to_csv("test_prediction_results.csv", index=False)

# ------------------------------------------------------------
# 10. Bonus: Compare uniform and distance-weighted voting
# ------------------------------------------------------------

print("\n8. BONUS: UNIFORM VS DISTANCE-WEIGHTED VOTING")
print("-" * 70)

voting_results = []

for weight_type in ["uniform", "distance"]:

    model = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean",
        weights=weight_type
    )

    model.fit(X_final_train, y_final_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    voting_results.append({
        "Voting Method": weight_type,
        "Test Accuracy (%)": round(accuracy * 100, 2)
    })

voting_df = pd.DataFrame(voting_results)

print(voting_df.to_string(index=False))

voting_df.to_csv("voting_comparison.csv", index=False)

# ------------------------------------------------------------
# 11. Predict purchase outcomes for ten new customers
# ------------------------------------------------------------

print("\n9. PREDICTIONS FOR TEN NEW CUSTOMERS")
print("-" * 70)

new_customers = pd.DataFrame({
    "Age": [25, 42, 31, 55, 28, 38, 47, 22, 60, 35],
    "Monthly Spending": [
        300, 600, 450, 200, 800, 350, 500, 250, 150, 700
    ],
    "Website Visits": [20, 12, 30, 5, 40, 18, 15, 25, 3, 35],
    "Time on Website": [
        25, 10, 35, 5, 45, 20, 15, 30, 3, 40
    ],
    "Previous Purchases": [5, 3, 8, 1, 12, 4, 6, 2, 0, 10]
})

new_customers_scaled = scaler.transform(new_customers[features])

new_predictions = final_model.predict(new_customers_scaled)

new_customers["Predicted Purchase"] = new_predictions

print(new_customers.to_string(index=False))

new_customers.to_csv("new_customer_predictions.csv", index=False)

# ------------------------------------------------------------
# 12. Explain why the best K performed better
# ------------------------------------------------------------

print("\n10. OBSERVATIONS AND CONCLUSION")
print("-" * 70)

best_accuracy = accuracy_df[
    accuracy_df["K"] == best_k
]["Validation Accuracy (%)"].iloc[0]

print(
    f"Observation 1: K={best_k} achieved the highest validation "
    f"accuracy of {best_accuracy:.2f}% among the tested K values."
)

k1_accuracy = accuracy_df[
    accuracy_df["K"] == 1
]["Validation Accuracy (%)"].iloc[0]

print(
    f"Observation 2: K=1 achieved validation accuracy of "
    f"{k1_accuracy:.2f}%, compared with {best_accuracy:.2f}% "
    f"for K={best_k}."
)

print(
    "Observation 3: A small K can be sensitive to individual "
    "customers, while a larger K averages more neighbors. "
    "The best K balances these effects for this dataset."
)

print(
    f"\nConclusion: K={best_k} was selected using validation accuracy. "
    f"Its final test accuracy was {test_accuracy * 100:.2f}%. "
    "The test set was kept separate during K selection."
)

print("\nFiles created:")
print("1. customer_purchase_dataset.csv")
print("2. knn_accuracy_comparison.csv")
print("3. knn_accuracy_chart.png")
print("4. test_prediction_results.csv")
print("5. voting_comparison.csv")
print("6. new_customer_predictions.csv")

print("\n" + "=" * 70)
print("CUSTOMER PURCHASE PREDICTION EXPERIMENT COMPLETED")
print("=" * 70)