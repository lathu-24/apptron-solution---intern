# ============================================================
# Day 41: Distance Calculator and Neighbor Finder
# ============================================================

import numpy as np
import pandas as pd

from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# 1. Create a dataset containing 50 customers
# ------------------------------------------------------------

print("=" * 70)
print("DISTANCE CALCULATOR AND NEIGHBOR FINDER")
print("=" * 70)

np.random.seed(42)

number_of_customers = 50

customer_ids = [
    f"C{i:03d}" for i in range(1, number_of_customers + 1)
]

ages = np.random.randint(18, 66, number_of_customers)
monthly_spending = np.random.randint(20, 501, number_of_customers)
website_visits = np.random.randint(1, 31, number_of_customers)

# Create synthetic purchase labels for demonstration.
# Purchase probability depends on spending and website visits.
purchase_score = (
    monthly_spending * 0.06
    + website_visits * 1.5
    - ages * 0.1
)

purchased = np.where(purchase_score >= 25, "Yes", "No")

df = pd.DataFrame({
    "Customer ID": customer_ids,
    "Age": ages,
    "Monthly Spending": monthly_spending,
    "Website Visits": website_visits,
    "Purchased": purchased
})

# Save dataset
df.to_csv("customer_dataset.csv", index=False)

print("\n1. CUSTOMER DATASET")
print("-" * 70)

print("First five customer records:")
print(df.head())

print(f"\nTotal customers: {len(df)}")

# ------------------------------------------------------------
# 2. Select one new customer for testing
# ------------------------------------------------------------

print("\n2. NEW CUSTOMER")
print("-" * 70)

new_customer = {
    "Customer ID": "NEW001",
    "Age": 30,
    "Monthly Spending": 250,
    "Website Visits": 12
}

print(pd.DataFrame([new_customer]).to_string(index=False))

# ------------------------------------------------------------
# 3. Create a Euclidean distance function
# ------------------------------------------------------------

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two numerical points.

    Supports two or more numerical features.
    Both points must contain the same number of values.
    """
    point1 = np.array(point1, dtype=float)
    point2 = np.array(point2, dtype=float)

    if point1.shape != point2.shape:
        raise ValueError(
            "Both points must contain the same number of features."
        )

    return np.sqrt(np.sum((point1 - point2) ** 2))


print("\n3. EUCLIDEAN DISTANCE FUNCTION")
print("-" * 70)

example_distance = euclidean_distance(
    [30, 250],
    [35, 270]
)

print("Example point 1: [30, 250]")
print("Example point 2: [35, 270]")
print(f"Euclidean distance: {example_distance:.2f}")

# ------------------------------------------------------------
# 4. Calculate distances using Age and Monthly Spending
# ------------------------------------------------------------

print("\n4. CALCULATE DISTANCES")
print("-" * 70)

new_point = [
    new_customer["Age"],
    new_customer["Monthly Spending"]
]

df["Distance"] = df.apply(
    lambda row: euclidean_distance(
        [row["Age"], row["Monthly Spending"]],
        new_point
    ),
    axis=1
)

print("Distances calculated for all 50 customers.")

# ------------------------------------------------------------
# 5. Sort customers by distance
# ------------------------------------------------------------

print("\n5. SORTED CUSTOMER TABLE")
print("-" * 70)

sorted_customers = df.sort_values(
    by="Distance",
    ascending=True
).reset_index(drop=True)

print(
    sorted_customers[
        [
            "Customer ID",
            "Age",
            "Monthly Spending",
            "Website Visits",
            "Purchased",
            "Distance"
        ]
    ].round({"Distance": 2}).to_string(index=False)
)

# Save sorted customer table
sorted_customers.to_csv("sorted_customer_distances.csv", index=False)

# ------------------------------------------------------------
# 6. Display the three nearest customers
# ------------------------------------------------------------

print("\n6. THREE NEAREST CUSTOMERS")
print("-" * 70)

nearest_neighbors = sorted_customers.head(3)

print(
    nearest_neighbors[
        [
            "Customer ID",
            "Age",
            "Monthly Spending",
            "Purchased",
            "Distance"
        ]
    ].round({"Distance": 2}).to_string(index=False)
)

# ------------------------------------------------------------
# 7. Predict purchase class using majority voting
# ------------------------------------------------------------

print("\n7. MANUAL KNN PREDICTION")
print("-" * 70)

neighbor_labels = nearest_neighbors["Purchased"].tolist()

vote_counts = Counter(neighbor_labels)

manual_prediction = vote_counts.most_common(1)[0][0]

print(f"Neighbor purchase labels: {neighbor_labels}")
print(f"Vote counts: {dict(vote_counts)}")
print(f"Manual KNN prediction: {manual_prediction}")

# ------------------------------------------------------------
# 8. Compare with Scikit-learn KNN (K = 3)
# ------------------------------------------------------------

print("\n8. SCIKIT-LEARN KNN PREDICTION")
print("-" * 70)

features = ["Age", "Monthly Spending"]
X = df[features]
y = df["Purchased"]

# StandardScaler is included so that Age and Monthly Spending
# contribute on comparable scales when calculating distances.
# The manual comparison below uses the same scaled features.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

new_customer_df = pd.DataFrame([{
    "Age": new_customer["Age"],
    "Monthly Spending": new_customer["Monthly Spending"]
}])

new_customer_scaled = scaler.transform(new_customer_df)

knn_model = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean",
    weights="uniform"
)

knn_model.fit(X_scaled, y)

sklearn_prediction = knn_model.predict(new_customer_scaled)[0]

print(f"Scikit-learn KNN prediction: {sklearn_prediction}")

# ------------------------------------------------------------
# 9. Compare manual and Scikit-learn predictions fairly
# ------------------------------------------------------------

print("\n9. PREDICTION COMPARISON")
print("-" * 70)

# The manual calculation above used raw features.
# For a fair comparison, recalculate the manual neighbors using
# the same scaled features as Scikit-learn.

scaled_new_point = new_customer_scaled[0]

scaled_distances = np.sqrt(
    np.sum((X_scaled - scaled_new_point) ** 2, axis=1)
)

scaled_df = df.copy()
scaled_df["Scaled Distance"] = scaled_distances

scaled_neighbors = scaled_df.nsmallest(3, "Scaled Distance")

scaled_neighbor_labels = scaled_neighbors["Purchased"].tolist()

scaled_vote_counts = Counter(scaled_neighbor_labels)

scaled_manual_prediction = scaled_vote_counts.most_common(1)[0][0]

print("\nThree nearest neighbors using scaled features:")
print(
    scaled_neighbors[
        ["Customer ID", "Age", "Monthly Spending", "Purchased",
         "Scaled Distance"]
    ].round({"Scaled Distance": 3}).to_string(index=False)
)

print(f"\nManual KNN prediction (scaled features): {scaled_manual_prediction}")
print(f"Scikit-learn KNN prediction:              {sklearn_prediction}")

if scaled_manual_prediction == sklearn_prediction:
    print("Result: Both predictions match.")
else:
    print("Result: Predictions differ. Check neighbor selection and voting.")

comparison = pd.DataFrame({
    "Method": [
        "Manual KNN (raw features)",
        "Manual KNN (scaled features)",
        "Scikit-learn KNN (scaled features)"
    ],
    "Predicted Purchase": [
        manual_prediction,
        scaled_manual_prediction,
        sklearn_prediction
    ]
})

print("\nPrediction comparison table:")
print(comparison.to_string(index=False))

comparison.to_csv("prediction_comparison.csv", index=False)

# ------------------------------------------------------------
# 10. Bonus: Distance with three or more features
# ------------------------------------------------------------

print("\n10. BONUS: MULTI-FEATURE DISTANCE")
print("-" * 70)

point_a = [30, 250, 12]
point_b = [35, 270, 15]

bonus_distance = euclidean_distance(point_a, point_b)

print(f"Point A (Age, Spending, Visits): {point_a}")
print(f"Point B (Age, Spending, Visits): {point_b}")
print(f"Three-feature Euclidean distance: {bonus_distance:.2f}")

# ------------------------------------------------------------
# 11. Display three observations
# ------------------------------------------------------------

print("\n11. OBSERVATIONS")
print("-" * 70)

print(
    f"Observation 1: The three nearest customers have purchase "
    f"labels {neighbor_labels}."
)

print(
    f"Observation 2: Manual majority voting predicts "
    f"'{manual_prediction}' for the new customer using raw features."
)

print(
    f"Observation 3: After using the same scaled features, "
    f"manual KNN and Scikit-learn KNN "
    f"{'agree' if scaled_manual_prediction == sklearn_prediction else 'disagree'}."
)

print("\nFiles created:")
print("1. customer_dataset.csv")
print("2. sorted_customer_distances.csv")
print("3. prediction_comparison.csv")

print("\n" + "=" * 70)
print("PRACTICAL COMPLETED")
print("=" * 70)