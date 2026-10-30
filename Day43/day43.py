# ============================================================
# PRACTICAL TASK: KNN SCALING AND EVALUATION LAB
# Objective: Test whether scaling improves customer purchase
# prediction using K-Nearest Neighbors (KNN).
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# TASK 1: LOAD THE CUSTOMER DATASET
# ============================================================

df = pd.read_csv("customer_purchase_dataset.csv")

print("\n========== DATASET INFORMATION ==========")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# TASK 2: INSPECT MISSING VALUES, DUPLICATES AND CLASSES
# ============================================================

print("\n========== DATA INSPECTION ==========")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nClass Distribution:")
print(df["Purchased"].value_counts())

print("\nClass Distribution Percentage:")
print(df["Purchased"].value_counts(normalize=True) * 100)


# ============================================================
# TASK 3: DATA CLEANING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

print("\nDataset Shape After Cleaning:", df.shape)


# ============================================================
# TASK 4: SEPARATE FEATURES AND TARGET
# ============================================================

# Features
X = df.drop(columns=["Purchased"])

# Target
y = df["Purchased"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# TASK 5: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLITTING ==========")

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# TASK 6: FIND THE BEST K USING CROSS-VALIDATION
# ============================================================

k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

scalers = {
    "No Scaling": "passthrough",
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler()
}

for scaler_name, scaler in scalers.items():

    for k in k_values:

        model = Pipeline([
            ("scaler", scaler),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        cv_results.append({
            "Scaler": scaler_name,
            "K": k,
            "Mean CV F1": scores.mean(),
            "Std CV F1": scores.std()
        })


cv_df = pd.DataFrame(cv_results)

print("\n========== CROSS-VALIDATION RESULTS ==========")

print(cv_df)

cv_df.to_csv("cross_validation_results.csv", index=False)


# ============================================================
# TASK 7: SELECT THE BEST K
# ============================================================

# Select the best K using StandardScaler cross-validation F1
standard_results = cv_df[
    cv_df["Scaler"] == "StandardScaler"
]

best_row = standard_results.loc[
    standard_results["Mean CV F1"].idxmax()
]

best_k = int(best_row["K"])

print("\nBest K Value:", best_k)

print(
    "Best Cross-Validation F1:",
    round(best_row["Mean CV F1"], 4)
)


# ============================================================
# TASK 8: TRAIN KNN WITHOUT SCALING
# ============================================================

knn_without_scaling = Pipeline([
    ("scaler", "passthrough"),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

knn_without_scaling.fit(X_train, y_train)

y_pred_without = knn_without_scaling.predict(X_test)


# ============================================================
# TASK 9: TRAIN KNN WITH STANDARDSCALER
# ============================================================

knn_standard = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

knn_standard.fit(X_train, y_train)

y_pred_standard = knn_standard.predict(X_test)


# ============================================================
# BONUS: TRAIN KNN WITH MINMAXSCALER
# ============================================================

knn_minmax = Pipeline([
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

knn_minmax.fit(X_train, y_train)

y_pred_minmax = knn_minmax.predict(X_test)


# ============================================================
# TASK 10: EVALUATE MODELS
# ============================================================

def evaluate_model(model_name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    print(f"\n========== {model_name} ==========")

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, zero_division=0
    ))

    return {
        "Model": model_name,
        "K": best_k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


# Evaluate all models

results = []

results.append(
    evaluate_model(
        "KNN Without Scaling",
        y_test,
        y_pred_without
    )
)

results.append(
    evaluate_model(
        "KNN with StandardScaler",
        y_test,
        y_pred_standard
    )
)

results.append(
    evaluate_model(
        "KNN with MinMaxScaler",
        y_test,
        y_pred_minmax
    )
)


# ============================================================
# TASK 11: COMPARE MODEL PERFORMANCE
# ============================================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison_metrics.csv",
    index=False
)


# ============================================================
# TASK 12: DISPLAY CONFUSION MATRICES
# ============================================================

predictions = {
    "KNN Without Scaling": y_pred_without,
    "KNN with StandardScaler": y_pred_standard,
    "KNN with MinMaxScaler": y_pred_minmax
}

for name, predictions_values in predictions.items():

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions_values,
        display_labels=["Not Purchased", "Purchased"],
        values_format="d"
    )

    plt.title(f"Confusion Matrix - {name}")

    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        f"{filename}_confusion_matrix.png",
        dpi=150
    )

    plt.show()


# ============================================================
# TASK 13: VISUALIZE CROSS-VALIDATION RESULTS
# ============================================================

for scaler_name in scalers.keys():

    subset = cv_df[
        cv_df["Scaler"] == scaler_name
    ]

    plt.plot(
        subset["K"],
        subset["Mean CV F1"],
        marker="o",
        label=scaler_name
    )

plt.xlabel("Number of Neighbors (K)")

plt.ylabel("Mean Cross-Validation F1-Score")

plt.title("KNN Performance Comparison")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "knn_cross_validation_comparison.png",
    dpi=150
)

plt.show()


# ============================================================
# TASK 14: ANALYZE WHETHER SCALING IMPROVED PERFORMANCE
# ============================================================

unscaled_f1 = results_df.loc[
    results_df["Model"] == "KNN Without Scaling",
    "F1-score"
].iloc[0]

standard_f1 = results_df.loc[
    results_df["Model"] == "KNN with StandardScaler",
    "F1-score"
].iloc[0]

print("\n========== SCALING ANALYSIS ==========")

if standard_f1 > unscaled_f1:

    print("StandardScaler improved the model performance.")

    print(
        "F1-score improvement:",
        round(standard_f1 - unscaled_f1, 4)
    )

elif standard_f1 < unscaled_f1:

    print("StandardScaler did not improve the F1-score.")

    print(
        "F1-score difference:",
        round(unscaled_f1 - standard_f1, 4)
    )

else:

    print("Both models achieved the same F1-score.")


# ============================================================
# TASK 15: GENERATE A SHORT REPORT
# ============================================================

best_model = results_df.loc[
    results_df["F1-score"].idxmax()
]

if standard_f1 > unscaled_f1:

    explanation = (
        "StandardScaler improved performance. KNN uses distance "
        "calculations, so features with larger numeric ranges can "
        "dominate the distance. Scaling gives features a more "
        "comparable influence."
    )

elif standard_f1 < unscaled_f1:

    explanation = (
        "StandardScaler did not improve performance on this test "
        "split. Scaling does not guarantee better results for every "
        "dataset; the effect depends on feature distributions."
    )

else:

    explanation = (
        "StandardScaler and the unscaled model achieved the same "
        "F1-score on this test split."
    )


report = f"""
KNN SCALING AND EVALUATION LAB
================================

Dataset Information
-------------------
Total Records: {len(df)}
Features: {", ".join(X.columns)}
Target Variable: Purchased

Training Records: {len(X_train)}
Testing Records: {len(X_test)}

Best K Value: {best_k}

Model Comparison
----------------
{results_df.to_string(index=False)}

Best Model: {best_model['Model']}

Best Model Accuracy: {best_model['Accuracy']:.4f}
Best Model Precision: {best_model['Precision']:.4f}
Best Model Recall: {best_model['Recall']:.4f}
Best Model F1-Score: {best_model['F1-score']:.4f}

Scaling Analysis
----------------
{explanation}

Conclusion
----------
KNN is a distance-based algorithm. Feature scaling is important
because features with larger numeric ranges can dominate distance
calculations.

StandardScaler standardizes features using their mean and standard
deviation, while MinMaxScaler scales values into a 0-to-1 range.

The best scaling method should be selected using cross-validation
on the training data. The test set is reserved for final evaluation.

Note: This is an experimental customer dataset. Results should not
be interpreted as evidence of real customer purchasing behavior.
"""

with open(
    "knn_scaling_lab_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


print("\n========== FINAL REPORT ==========")

print(report)


# ============================================================
# END OF PRACTICAL TASK
# ============================================================

print("\nKNN Scaling and Evaluation Lab Completed Successfully!")