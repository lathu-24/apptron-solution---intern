# ============================================================
# DAY 37: CUSTOMER PURCHASE PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. CREATE DATASET (2,000 CUSTOMERS)
# ============================================================

np.random.seed(42)

number_of_customers = 2000

# Generate customer information
age = np.random.randint(18, 66, number_of_customers)

income = np.random.randint(20000, 200001, number_of_customers)

website_visits = np.random.randint(1, 31, number_of_customers)

time_spent = np.random.uniform(1, 60, number_of_customers)

previous_purchases = np.random.randint(0, 21, number_of_customers)

product_views = np.random.randint(1, 51, number_of_customers)


# Generate purchase labels using a simulated customer behavior rule.
# Purchased = 1
# Not Purchased = 0

purchase_score = (
    website_visits * 1.5
    + time_spent * 0.5
    + previous_purchases * 2
    + product_views * 0.8
    + income / 10000
    - age * 0.1
)

# Add randomness so the model does not learn a perfectly deterministic rule
purchase_score += np.random.normal(0, 15, number_of_customers)

purchased = (purchase_score >= 65).astype(int)


# Create DataFrame
data = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Website Visits": website_visits,
    "Time Spent": np.round(time_spent, 2),
    "Previous Purchases": previous_purchases,
    "Product Views": product_views,
    "Purchased": purchased
})


# Save dataset
data.to_csv("customer_purchase_dataset.csv", index=False)

print("Customer dataset created successfully!")
print("Total customer records:", len(data))


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = pd.read_csv("customer_purchase_dataset.csv")

print("\n========== DATASET LOADED ==========")

print("\nFirst five records:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)


# ============================================================
# 3. CLEAN DATASET
# ============================================================

print("\n========== DATA CLEANING ==========")

print("\nMissing values before cleaning:")
print(data.isnull().sum())

print("\nDuplicate records before cleaning:")
print(data.duplicated().sum())

# Remove duplicate records
data = data.drop_duplicates().copy()

# Fill missing numeric values with column medians
numeric_columns = data.select_dtypes(include=np.number).columns

data[numeric_columns] = data[numeric_columns].fillna(
    data[numeric_columns].median()
)

# Remove invalid records
data = data[
    data["Age"].between(18, 100)
    & (data["Income"] > 0)
    & (data["Website Visits"] >= 0)
    & (data["Time Spent"] >= 0)
    & (data["Previous Purchases"] >= 0)
    & (data["Product Views"] >= 0)
    & data["Purchased"].isin([0, 1])
].copy()

print("\nDataset cleaned successfully!")
print("Records after cleaning:", len(data))

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nPurchase distribution:")
print(data["Purchased"].value_counts().sort_index())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = data[
    [
        "Age",
        "Income",
        "Website Visits",
        "Time Spent",
        "Previous Purchases",
        "Product Views"
    ]
]

y = data["Purchased"]

print("\n========== FEATURES AND TARGET ==========")

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nTarget meaning:")
print("1 = Purchased")
print("0 = Not Purchased")


# ============================================================
# 5. TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN/TEST SPLIT ==========")

print("Total records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. TRAIN DECISION TREE CLASSIFIER
# ============================================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("\nDecision Tree trained successfully!")


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\n========== MODEL PREDICTIONS ==========")

prediction_output = pd.DataFrame({
    "Customer": X_test.index + 1,
    "Actual": y_test.map({
        1: "Purchased",
        0: "Not Purchased"
    }).to_numpy(),
    "Predicted": [
        "Purchased" if value == 1 else "Not Purchased"
        for value in y_pred
    ]
})

print(prediction_output.head(15).to_string(index=False))

prediction_output.to_csv(
    "customer_purchase_predictions.csv",
    index=False
)

print("\nPredictions saved to customer_purchase_predictions.csv")


# ============================================================
# 8. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Purchased", "Purchased"],
        zero_division=0
    )
)


# ============================================================
# 9. IDENTIFY INCORRECTLY CLASSIFIED CUSTOMERS
# ============================================================

print("\n========== INCORRECT PREDICTIONS ==========")

incorrect = X_test.copy()

incorrect["Actual"] = y_test
incorrect["Predicted"] = y_pred

incorrect = incorrect[
    incorrect["Actual"] != incorrect["Predicted"]
].copy()

incorrect["Actual"] = incorrect["Actual"].map({
    1: "Purchased",
    0: "Not Purchased"
})

incorrect["Predicted"] = incorrect["Predicted"].map({
    1: "Purchased",
    0: "Not Purchased"
})

print("Number of incorrect predictions:", len(incorrect))

if len(incorrect) > 0:
    print(incorrect.head(20).to_string())
else:
    print("No incorrect predictions.")

incorrect.to_csv(
    "incorrect_customer_predictions.csv",
    index=True
)


# ============================================================
# 10. VISUALIZE DECISION TREE
# ============================================================

plt.figure(figsize=(24, 14))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Purchased", "Purchased"],
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=8
)

plt.title("Customer Purchase Prediction Decision Tree")

plt.tight_layout()

plt.savefig(
    "customer_purchase_decision_tree.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. DISPLAY DECISION RULES
# ============================================================

print("\n========== DECISION TREE RULES ==========")

tree_rules = export_text(
    model,
    feature_names=list(X.columns)
)

print(tree_rules)

with open(
    "customer_purchase_tree_rules.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(tree_rules)


# ============================================================
# 12. IDENTIFY ROOT FEATURE
# ============================================================

root_feature_index = model.tree_.feature[0]

root_feature = X.columns[root_feature_index]

root_threshold = model.tree_.threshold[0]

print("\n========== ROOT FEATURE ==========")

print("Root feature:", root_feature)

print(
    f"Root question: Is {root_feature} <= {root_threshold:.2f}?"
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))

plt.figure(figsize=(10, 6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance - Customer Purchase Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("customer_purchase_feature_importance.png", dpi=200)
plt.show()


# ============================================================
# 14. TEST 10 NEW CUSTOMERS
# ============================================================

print("\n========== TEST 10 NEW CUSTOMERS ==========")

new_customers = pd.DataFrame({
    "Age": [22, 35, 45, 28, 52, 31, 40, 24, 60, 38],
    "Income": [
        35000, 85000, 120000, 45000, 65000,
        95000, 70000, 40000, 55000, 110000
    ],
    "Website Visits": [3, 15, 8, 20, 5, 18, 12, 25, 2, 22],
    "Time Spent": [5, 35, 20, 45, 10, 40, 25, 50, 3, 55],
    "Previous Purchases": [0, 8, 4, 10, 1, 12, 5, 15, 0, 18],
    "Product Views": [4, 25, 15, 35, 8, 30, 20, 40, 3, 45]
})

# Predict for new customers
new_predictions = model.predict(new_customers)

# Create prediction output
new_customer_results = new_customers.copy()

new_customer_results.insert(
    0,
    "Customer",
    [f"New Customer {i}" for i in range(1, 11)]
)

new_customer_results["Predicted"] = [
    "Purchased" if value == 1 else "Not Purchased"
    for value in new_predictions
]

print(
    new_customer_results[
        ["Customer", "Predicted"]
    ].to_string(index=False)
)

new_customer_results.to_csv(
    "new_customer_predictions.csv",
    index=False
)


# ============================================================
# 15. BONUS: PREDICT CUSTOMER PURCHASE FUNCTION
# ============================================================

def predict_customer_purchase(
    age,
    income,
    website_visits,
    time_spent,
    previous_purchases,
    product_views
):
    """
    Predict whether a customer will purchase.

    Returns:
        'Purchased' or 'Not Purchased'
    """

    customer = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Website Visits": website_visits,
        "Time Spent": time_spent,
        "Previous Purchases": previous_purchases,
        "Product Views": product_views
    }])

    prediction = model.predict(customer)[0]

    if prediction == 1:
        return "Purchased"
    else:
        return "Not Purchased"


# Example function call
print("\n========== BONUS FUNCTION ==========")

result = predict_customer_purchase(
    age=29,
    income=75000,
    website_visits=18,
    time_spent=35,
    previous_purchases=6,
    product_views=28
)

print("Customer Purchase Prediction:", result)


# ============================================================
# 16. PRACTICAL SUMMARY
# ============================================================

print("\n========== PRACTICAL SUMMARY ==========")

print("Dataset created: 2,000 customer records")
print("Dataset cleaned successfully")
print("Decision Tree Classifier trained")
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("Root feature:", root_feature)
print("Incorrect predictions:", len(incorrect))
print("Ten new customers tested")
print("Bonus prediction function implemented")

print("\nDAY 37 PRACTICAL COMPLETED!")