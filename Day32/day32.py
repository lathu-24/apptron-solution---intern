from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "customer_purchase_data.csv"

FEATURES = ["Age", "Income", "Website Visits", "Time Spent",
            "Previous Purchases", "Product Views"]
TARGET = "Purchase"

# Load and inspect the prepared dataset
df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)
print("\\nFirst five rows:\\n", df.head())
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isnull().sum())
print("\\nTarget counts (0 = No purchase, 1 = Purchased):\\n", df[TARGET].value_counts())

df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES]
y = df[TARGET].astype(int)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features and train Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)

# Test predictions and probabilities
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= 0.50).astype(int)
results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": test_predictions,
    "Probability": (test_probabilities * 100).round(2).astype(str) + "%"
})
print("\\nTest results (first 20 rows):")
print(results.head(20).to_string(index=False))
print(f"\\nTest accuracy at threshold 0.50: {accuracy_score(y_test, test_predictions):.2%}")
print("\\nClassification report:")
print(classification_report(y_test, test_predictions, zero_division=0))

# Predict at least 10 new customers
new_customers = pd.DataFrame([
    [28, 120000, 8, 25, 3, 12],
    [22, 45000, 2, 8, 0, 3],
    [41, 165000, 14, 42, 8, 22],
    [35, 90000, 6, 18, 2, 9],
    [29, 75000, 12, 35, 1, 18],
    [52, 60000, 3, 10, 4, 5],
    [31, 200000, 18, 55, 12, 28],
    [24, 52000, 7, 15, 0, 8],
    [46, 135000, 10, 30, 6, 16],
    [38, 110000, 4, 12, 5, 7],
], columns=FEATURES)

new_probabilities = model.predict_proba(new_customers)[:, 1]
new_predictions = (new_probabilities >= 0.50).astype(int)
new_results = new_customers.copy()
new_results["Purchase Probability"] = (new_probabilities * 100).round(2).astype(str) + "%"
new_results["Prediction"] = [
    "Purchased" if value == 1 else "Not Purchased" for value in new_predictions
]
print("\\nNew customer predictions (threshold 0.50):")
print(new_results.to_string(index=False))

# Bonus: compare different decision thresholds
print("\\nThreshold experiment:")
for threshold in [0.30, 0.50, 0.70]:
    predictions = (test_probabilities >= threshold).astype(int)
    print(f"Threshold={threshold:.2f} | Predicted purchases={predictions.sum()}/{len(predictions)}"
          f" | Accuracy={accuracy_score(y_test, predictions):.2%}")

print("\\nLower thresholds usually predict more purchases; higher thresholds are stricter.")
