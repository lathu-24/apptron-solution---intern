from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "customer_churn_dataset.csv"
NUM = ["Age", "Tenure", "Monthly Charges", "Total Charges", "Support Calls", "Internet Usage", "Previous Complaints"]
CAT = ["Contract Type", "Payment Method"]
FEATURES = NUM + CAT
TARGET = "Churn"

# Part 1: load, inspect and clean
raw = pd.read_csv(DATA)
print("Raw shape:", raw.shape)
print(raw.head())
print("\nMissing values:\n", raw.isna().sum())
print("Duplicate rows:", raw.duplicated().sum())
print("Raw class distribution:\n", raw[TARGET].value_counts(dropna=False))

df = raw.drop_duplicates().dropna(subset=[TARGET]).copy()
for col in NUM:
    df[col] = pd.to_numeric(df[col], errors="coerce")
ranges = {"Age": (18, 100), "Tenure": (0, 120), "Monthly Charges": (0, 10000),
          "Total Charges": (0, 1000000), "Support Calls": (0, 100),
          "Internet Usage": (0, 10000), "Previous Complaints": (0, 100)}
for col, (lo, hi) in ranges.items():
    df.loc[~df[col].between(lo, hi), col] = np.nan
for col in CAT:
    df[col] = df[col].apply(lambda value: str(value).strip() if pd.notna(value) else np.nan)
    df[col] = df[col].replace("", np.nan)
df[TARGET] = df[TARGET].astype(str).str.strip().str.title()
df = df[df[TARGET].isin(["Yes", "No"])].copy()
print("\nClean shape:", df.shape)
print("Clean class distribution:\n", df[TARGET].value_counts())

X = df[FEATURES]
y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

prep = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUM),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CAT)
])
model = Pipeline([("prep", prep), ("logreg", LogisticRegression(max_iter=1500))])
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# Parts 5-6: test predictions and evaluation
table = pd.DataFrame({
    "Customer ID": df.loc[X_test.index, "Customer ID"].to_numpy(),
    "Actual": np.where(y_test.to_numpy() == 1, "Yes", "No"),
    "Predicted": np.where(pred == 1, "Yes", "No"),
    "Churn Probability": (prob * 100).round(2)
})
table["Correct"] = table["Actual"] == table["Predicted"]
table.to_csv(BASE / "actual_vs_predicted.csv", index=False)
acc = accuracy_score(y_test, pred)
tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
correct = int(table["Correct"].sum())
incorrect = len(table) - correct
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)

print("\n--- Evaluation ---")
print("Total Test Records:", len(y_test))
print("Correct Predictions:", correct)
print("Incorrect Predictions:", incorrect)
print(f"Accuracy: {acc:.2%}")
print("True Positives:", tp, "| True Negatives:", tn)
print("False Positives:", fp, "| False Negatives:", fn)
print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
print("\nActual | Predicted | Churn Probability (first 25):\n", table.head(25).to_string(index=False))
print("\nClassification report:\n", classification_report(y_test, pred, target_names=["No Churn", "Churn"], zero_division=0))

cm = confusion_matrix(y_test, pred, labels=[0, 1])
ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(values_format="d")
plt.title("Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.savefig(BASE / "confusion_matrix.png", dpi=160)
plt.close()

# Part 5: ten new customers
new = pd.DataFrame([
[28,4,120,480,6,55,3,"Month-to-month","Electronic check"],
[45,48,75,3600,1,25,0,"One year","Credit card"],
[62,8,145,1160,7,80,4,"Month-to-month","Electronic check"],
[35,60,55,3300,0,18,0,"Two year","Bank transfer"],
[24,3,95,285,5,65,2,"Month-to-month","Mailed check"],
[51,36,110,3960,2,42,1,"One year","Bank transfer"],
[39,12,160,1920,8,90,5,"Month-to-month","Electronic check"],
[29,54,68,3672,1,30,0,"Two year","Credit card"],
[70,6,130,780,4,70,2,"Month-to-month","Electronic check"],
[42,24,85,2040,2,35,1,"One year","Mailed check"],
], columns=NUM+CAT)
new["Churn Probability (%)"] = (model.predict_proba(new)[:, 1] * 100).round(2)
new["Predicted Churn"] = np.where(new["Churn Probability (%)"] >= 50, "Yes", "No")
new.to_csv(BASE / "new_customer_churn_predictions.csv", index=False)
print("\nNew customer predictions:\n", new.to_string(index=False))

# Part 7: Error analysis in the terminal
wrong = table[~table["Correct"]]
class_dist = df[TARGET].value_counts(normalize=True)
print("\n--- Error Analysis ---")
print("Misclassified customers:", len(wrong))
print("Misclassified customer details:")
print(wrong.to_string(index=False))
print("False positives are more common." if fp > fn else
      "False negatives are more common." if fn > fp else
      "False positives and false negatives are equal.")
print("Class distribution:", class_dist.to_dict())
print("Accuracy alone is not sufficient; consider precision, recall, F1-score and the confusion matrix.")
print("\n--- Business Recommendation ---")
performance = "Good" if acc >= 0.85 and recall >= 0.70 else "Moderate" if acc >= 0.70 else "Poor"
recommendation = "Yes, with monitoring" if performance == "Good" else "With improvements"
print("Model Performance:", performance)
print("Can the company use this model?", recommendation)
print("Reason: Validate the model on real, recent customer data and review the cost of false positives and false negatives.")
print("Recommended Improvements: Tune the decision threshold, check precision/recall/F1, test class weights, and monitor performance.")

print("\nSaved actual_vs_predicted.csv, new_customer_churn_predictions.csv and confusion_matrix.png")
