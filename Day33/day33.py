from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "customer_purchase_data.csv"
PLOT_FILE = BASE_DIR / "confusion_matrix.png"
RESULTS_FILE = BASE_DIR / "actual_vs_predicted.csv"

FEATURES = ["Age", "Income", "Website Visits", "Time Spent",
            "Previous Purchases", "Product Views"]
TARGET = "Purchase"

# Load and inspect the dataset
df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

df = df.dropna(subset=FEATURES + [TARGET])
X, y = df[FEATURES], df[TARGET].astype(int)

# Same reproducible 80/20 split used in Day 32
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Train Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred,
    "Correct": y_test.to_numpy() == y_pred,
})
results.to_csv(RESULTS_FILE, index=False)

print("\n--- Classification Performance ---")
print(f"Accuracy: {accuracy:.2%}")
print(f"Correct predictions: {int(results['Correct'].sum())}")
print(f"Incorrect predictions: {int((~results['Correct']).sum())}")
print(f"True negatives (TN): {tn}")
print(f"False positives (FP): {fp}")
print(f"False negatives (FN): {fn}")
print(f"True positives (TP): {tp}")
print("\nActual vs Predicted (first 20 rows):")
print(results.head(20).to_string(index=False))

# Confusion matrix visualization
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Purchased (0)", "Purchased (1)"],
)
display.plot(values_format="d")
plt.title("Customer Purchase Model - Confusion Matrix")
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()

# Business interpretation
print("\n--- Business Analysis ---")
print(f"False positives: {fp} customers were predicted to purchase but did not.")
print(f"False negatives: {fn} customers were predicted not to purchase but did.")
print("False positives may waste marketing budget; false negatives may miss potential buyers.")
print("Accuracy alone may be insufficient; consider precision, recall, F1-score, and class balance.")

# Bonus: compare two feature sets using the same split strategy
feature_sets = {
    "Model 1 - All features": FEATURES,
    "Model 2 - Engagement and purchase history": [
        "Website Visits", "Time Spent", "Previous Purchases", "Product Views"
    ],
}
print("\n--- Bonus: Feature Set Comparison ---")
for name, cols in feature_sets.items():
    X_variant = df[cols]
    Xtr, Xte, ytr, yte = train_test_split(
        X_variant, y, test_size=0.20, random_state=42, stratify=y
    )
    variant = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    variant.fit(Xtr, ytr)
    pred = variant.predict(Xte)
    print(f"{name}: accuracy = {accuracy_score(yte, pred):.2%} ({len(cols)} features)")

print(f"\nSaved results table: {RESULTS_FILE}")
print(f"Saved confusion matrix: {PLOT_FILE}")
