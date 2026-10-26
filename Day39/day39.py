# ============================================================
# DAY 39: HEART DISEASE PREDICTION USING DECISION TREE
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. CREATE THE DATASET
# ============================================================

np.random.seed(42)

n = 1200

age = np.random.randint(29, 78, n)
sex = np.random.randint(0, 2, n)
cp = np.random.randint(0, 4, n)

trestbps = np.clip(
    np.random.normal(130, 17, n), 90, 200
)

chol = np.clip(
    np.random.normal(245, 48, n), 120, 500
)

thalach = np.clip(
    np.random.normal(150, 22, n), 70, 210
)

exang = np.random.randint(0, 2, n)
fbs = np.random.randint(0, 2, n)

oldpeak = np.clip(
    np.random.normal(1.0, 1.1, n), 0, 6
)

slope = np.random.randint(0, 3, n)
ca = np.random.randint(0, 4, n)
thal = np.random.randint(0, 3, n)

# Generate synthetic labels for educational purposes
risk = (
    0.035 * (age - 45)
    + 0.7 * (cp == 0)
    + 0.65 * exang
    + 0.35 * (oldpeak > 1.5)
    + 0.35 * (ca >= 2)
    + 0.20 * (chol > 240)
    + 0.25 * (trestbps > 140)
    - 0.018 * (thalach - 140)
    + np.random.normal(0, 0.7, n)
)

target = (
    risk > np.quantile(risk, 0.5)
).astype(int)

data = pd.DataFrame({
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": np.round(trestbps, 1),
    "chol": np.round(chol, 1),
    "thalach": np.round(thalach, 1),
    "exang": exang,
    "fbs": fbs,
    "oldpeak": np.round(oldpeak, 2),
    "slope": slope,
    "ca": ca,
    "thal": thal,
    "target": target
})

data.to_csv("heart_disease.csv", index=False)

print("Dataset created successfully!")
print("Dataset size:", data.shape)


# ============================================================
# 2. LOAD AND INSPECT DATASET
# ============================================================

data = pd.read_csv("heart_disease.csv")

print("\n========== DATA INSPECTION ==========")

print("\nFirst five records:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate records:")
print(data.duplicated().sum())

print("\nStatistical summary:")
print(data.describe())


# ============================================================
# 3. CLEAN THE DATASET
# ============================================================

data = data.drop_duplicates().copy()

data.replace(["?", "NA", "N/A", ""], np.nan, inplace=True)

for column in data.columns:
    data[column] = pd.to_numeric(
        data[column], errors="coerce"
    )

# Remove records without target labels
data = data.dropna(subset=["target"])

# Fill missing feature values using medians
for column in data.columns:
    if column != "target":
        data[column] = data[column].fillna(
            data[column].median()
        )

# Keep only valid binary targets
data = data[data["target"].isin([0, 1])].copy()

print("\nDataset cleaned successfully!")
print("Records after cleaning:", len(data))


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

output_dir = "heart_disease_outputs"
os.makedirs(output_dir, exist_ok=True)

# Target distribution
plt.figure(figsize=(7, 5))

data["target"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Heart Disease Class Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "target_distribution.png")
)
plt.show()


# Age distribution
plt.figure(figsize=(7, 5))

plt.hist(data["age"], bins=20, edgecolor="black")

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "age_distribution.png")
)
plt.show()


# Cholesterol distribution
plt.figure(figsize=(7, 5))

plt.hist(data["chol"], bins=20, edgecolor="black")

plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "cholesterol_distribution.png")
)
plt.show()


# Age versus target
plt.figure(figsize=(7, 5))

data.boxplot(
    column="age",
    by="target"
)

plt.title("Age vs Heart Disease")
plt.suptitle("")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "age_vs_target.png")
)
plt.show()


# Correlation heatmap using Matplotlib
correlation = data.corr(numeric_only=True)

plt.figure(figsize=(10, 8))

plt.imshow(
    correlation,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "correlation_heatmap.png")
)
plt.show()


# ============================================================
# 5. FIVE DATA INSIGHTS
# ============================================================

print("\n========== FIVE DATA INSIGHTS ==========")

print(
    "1. Class distribution:\n",
    data["target"].value_counts()
)

print(
    "\n2. Average age by target:\n",
    data.groupby("target")["age"].mean()
)

print(
    "\n3. Average cholesterol by target:\n",
    data.groupby("target")["chol"].mean()
)

print(
    "\n4. Average maximum heart rate by target:\n",
    data.groupby("target")["thalach"].mean()
)

print(
    "\n5. Features most correlated with target:\n",
    correlation["target"]
    .drop("target")
    .sort_values(key=abs, ascending=False)
    .head(5)
)


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["target"])
y = data["target"].astype(int)

print("\nNumber of features:", X.shape[1])


# ============================================================
# 7. TRAIN/TEST SPLIT (80/20)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 8. OVERFITTING EXPERIMENT
# ============================================================

depths = [2, 3, 5, 8, 10]

results = []
models = {}

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train, train_pred
    )

    test_accuracy = accuracy_score(
        y_test, test_pred
    )

    gap = (
        train_accuracy - test_accuracy
    ) * 100

    results.append({
        "Depth": depth,
        "Training Accuracy (%)": train_accuracy * 100,
        "Testing Accuracy (%)": test_accuracy * 100,
        "Accuracy Gap (%)": gap
    })

    models[depth] = model


results_df = pd.DataFrame(results)

print("\n========== DEPTH COMPARISON ==========")

print(results_df.round(2).to_string(index=False))

results_df.to_csv(
    os.path.join(output_dir, "depth_comparison.csv"),
    index=False
)


# ============================================================
# 9. SELECT BEST MODEL BASED ON TEST ACCURACY
# ============================================================

best_index = results_df[
    "Testing Accuracy (%)"
].idxmax()

best_depth = int(
    results_df.loc[best_index, "Depth"]
)

final_model = models[best_depth]

y_pred = final_model.predict(X_test)

training_accuracy = accuracy_score(
    y_train,
    final_model.predict(X_train)
)

testing_accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== FINAL MODEL ==========")

print("Best tree depth:", best_depth)

print(
    f"Training accuracy: {training_accuracy * 100:.2f}%"
)

print(
    f"Testing accuracy: {testing_accuracy * 100:.2f}%"
)


# ============================================================
# 10. CONFUSION MATRIX AND ACTUAL VS PREDICTED
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Disease", "Disease"],
        zero_division=0
    )
)

actual_vs_predicted = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(actual_vs_predicted.head(15))

actual_vs_predicted.to_csv(
    os.path.join(output_dir, "actual_vs_predicted.csv"),
    index=False
)


# ============================================================
# 11. VISUALIZE THE OVERFITTING EXPERIMENT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Depth"],
    results_df["Training Accuracy (%)"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    results_df["Depth"],
    results_df["Testing Accuracy (%)"],
    marker="o",
    label="Testing Accuracy"
)

plt.title("Decision Tree Overfitting Experiment")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "overfitting_comparison.png")
)

plt.show()


# ============================================================
# 12. VISUALIZE FINAL DECISION TREE
# ============================================================

plt.figure(figsize=(22, 12))

plot_tree(
    final_model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=8
)

plt.title(
    f"Final Heart Disease Decision Tree (Depth {best_depth})"
)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "final_decision_tree.png"),
    dpi=180,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. DISPLAY FIVE DECISION RULES
# ============================================================

rules = export_text(
    final_model,
    feature_names=list(X.columns)
)

print("\n========== DECISION TREE RULES ==========")

print(rules)

with open(
    os.path.join(output_dir, "decision_rules.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(rules)


# ============================================================
# 14. FINAL REPORT SUMMARY
# ============================================================

print("\n========== FINAL REPORT ==========")

print("Dataset Size:", len(data))
print("Number of Features:", X.shape[1])
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print(
    f"Training Accuracy: {training_accuracy * 100:.2f}%"
)

print(
    f"Testing Accuracy: {testing_accuracy * 100:.2f}%"
)

print("Best Tree Depth:", best_depth)

best_row = results_df.loc[best_index]

gap = best_row["Accuracy Gap (%)"]

print(
    "Training-testing accuracy gap:",
    f"{gap:.2f} percentage points"
)

if gap >= 15:
    print("Overfitting Observed: Yes, a substantial gap exists.")
else:
    print("Overfitting Observed: No substantial gap at the selected depth.")

print("Final Model: Decision Tree Classifier")

print(
    "Why selected? It achieved the highest testing accuracy "
    "among the tested depths."
)

print(
    "Important observations: Review the EDA charts, "
    "depth comparison, and decision rules."
)

print("\nDAY 39 PRACTICAL COMPLETED!")