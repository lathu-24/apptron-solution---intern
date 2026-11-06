# ============================================================
# DAY 48 - MODEL RESULTS ANALYSIS DASHBOARD
# Student Pass/Fail Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. CREATE FOLDERS
# ============================================================

print("\n========== MODEL RESULTS DASHBOARD ==========")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD STUDENT DATASET
# ============================================================

print("\n========== LOADING DATASET ==========")

data = pd.read_csv("student_dataset.csv")

print("Dataset Shape:", data.shape)

print("\nDataset Columns:")
print(data.columns.tolist())

print("\nFirst Five Records:")
print(data.head())

# Remove missing and duplicate records
data = data.drop_duplicates()
data = data.dropna()

# Separate features and target
X = data.drop("Result", axis=1)

y = data["Result"].astype(str).str.strip().str.capitalize()

# Encode Fail = 0, Pass = 1
y = y.map({
    "Fail": 0,
    "Pass": 1
})

if y.isnull().any():
    raise ValueError("Unexpected values in Result column.")

y = y.astype(int)

print("\nClass Distribution:")
print(y.value_counts())


# ============================================================
# 3. LOAD SAME DAY47 TRAIN-TEST SPLIT
# ============================================================

print("\n========== LOADING TRAIN-TEST SPLIT ==========")

split_files = [
    Path("X_train.csv"),
    Path("X_test.csv"),
    Path("y_train.csv"),
    Path("y_test.csv")
]

if all(file.exists() for file in split_files):

    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")

    y_train = pd.read_csv("y_train.csv").squeeze()
    y_test = pd.read_csv("y_test.csv").squeeze()

    print("Loaded saved Day47 split.")

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Created a new reproducible split.")

# Ensure target values are integers
def encode_target(series):

    if series.dtype == "object":
        series = series.astype(str).str.strip().str.capitalize()
        series = series.map({
            "Fail": 0,
            "Pass": 1
        })

    if series.isnull().any():
        raise ValueError("Invalid target values in split files.")

    return series.astype(int)


y_train = encode_target(y_train)
y_test = encode_target(y_test)

# Validate feature order and columns
X_train = X_train[X.columns]
X_test = X_test[X.columns]

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# ============================================================
# 4. LOAD TRAINED MODELS
# ============================================================

print("\n========== LOADING TRAINED MODELS ==========")

model_files = {
    "Logistic Regression":
        model_dir / "logistic_regression_model.pkl",

    "Decision Tree":
        model_dir / "decision_tree_model.pkl",

    "KNN":
        model_dir / "knn_model.pkl"
}

models = {}

for name, path in model_files.items():

    if path.exists():

        models[name] = joblib.load(path)

        print(name, "loaded successfully.")

    else:

        print(name, "model file not found.")

        print("A replacement model will be trained.")

        if name == "Logistic Regression":

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ))
            ])

        elif name == "Decision Tree":

            model = DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )

        else:

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(
                    n_neighbors=5
                ))
            ])

        model.fit(X_train, y_train)

        models[name] = model

        joblib.dump(model, path)

        print(name, "trained and saved.")


# ============================================================
# 5. CALCULATE TRAINING AND TESTING METRICS
# ============================================================

print("\n========== MODEL PERFORMANCE ANALYSIS ==========")

results = []

predictions = {}

for name, model in models.items():

    # Training predictions
    train_pred = model.predict(X_train)

    # Testing predictions
    test_pred = model.predict(X_test)

    predictions[name] = test_pred

    # Training accuracy
    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    # Testing accuracy
    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    # Test metrics for Fail class (0)
    precision = precision_score(
        y_test,
        test_pred,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_pred,
        pos_label=0,
        zero_division=0
    )

    # Training/testing performance gap
    performance_gap = train_accuracy - test_accuracy

    results.append({

        "Model": name,

        "Training Accuracy": round(
            train_accuracy, 4
        ),

        "Testing Accuracy": round(
            test_accuracy, 4
        ),

        "Precision (Fail)": round(
            precision, 4
        ),

        "Recall (Fail)": round(
            recall, 4
        ),

        "F1-Score (Fail)": round(
            f1, 4
        ),

        "Performance Gap": round(
            performance_gap, 4
        )

    })


# ============================================================
# 6. CREATE METRICS TABLE
# ============================================================

metrics_df = pd.DataFrame(results)

print("\n========== FINAL METRICS TABLE ==========")

print(metrics_df.to_string(index=False))

metrics_df.to_csv(
    output_dir / "model_metrics.csv",
    index=False
)


# ============================================================
# 7. BAR CHART - TEST ACCURACY
# ============================================================

print("\nGenerating Accuracy Chart...")

plt.figure(figsize=(8, 5))

plt.bar(
    metrics_df["Model"],
    metrics_df["Testing Accuracy"]
)

plt.title("Test Accuracy Comparison")

plt.xlabel("Machine Learning Models")

plt.ylabel("Testing Accuracy")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    output_dir / "accuracy_comparison.png"
)

plt.close()


# ============================================================
# 8. GROUPED BAR CHART - PRECISION, RECALL, F1
# ============================================================

print("Generating Metrics Comparison Chart...")

metrics_plot = metrics_df.set_index("Model")[
    [
        "Precision (Fail)",
        "Recall (Fail)",
        "F1-Score (Fail)"
    ]
]

metrics_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Precision, Recall and F1-Score Comparison")

plt.xlabel("Machine Learning Models")

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    output_dir / "metrics_comparison.png"
)

plt.close()


# ============================================================
# 9. CONFUSION MATRIX HEATMAPS
# ============================================================

print("\nGenerating Confusion Matrices...")

for name, y_pred in predictions.items():

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print(f"\n{name} Confusion Matrix:")

    print(cm)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"]
    )

    plt.title(f"{name} Confusion Matrix")

    plt.xlabel("Predicted Result")

    plt.ylabel("Actual Result")

    filename = name.lower().replace(" ", "_")

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{filename}_confusion_matrix.png"
    )

    plt.close()


# ============================================================
# 10. TRAINING VS TESTING ACCURACY CHART
# ============================================================

print("\nGenerating Overfitting Analysis Chart...")

gap_plot = metrics_df.set_index("Model")[
    [
        "Training Accuracy",
        "Testing Accuracy"
    ]
]

gap_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Training vs Testing Accuracy")

plt.xlabel("Machine Learning Models")

plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    output_dir / "training_testing_comparison.png"
)

plt.close()


# ============================================================
# 11. OVERFITTING ANALYSIS
# ============================================================

print("\n========== OVERFITTING ANALYSIS ==========")

overfitting_models = []

for _, row in metrics_df.iterrows():

    gap = row["Performance Gap"]

    if gap > 0.10:

        status = "Possible overfitting"

        overfitting_models.append(row["Model"])

    elif gap > 0.05:

        status = "Moderate performance gap"

    elif gap < -0.05:

        status = "Testing score exceeds training score"

    else:

        status = "Small performance gap"

    print(
        f"{row['Model']}: {status} "
        f"(Gap: {gap:.4f})"
    )


# ============================================================
# 12. BONUS - PREDICTION TIME COMPARISON
# ============================================================

print("\n========== PREDICTION TIME ANALYSIS ==========")

timing_results = []

for name, model in models.items():

    # Warm-up prediction
    model.predict(X_test)

    # Average time across repeated predictions
    repetitions = 10

    start_time = time.perf_counter()

    for _ in range(repetitions):

        model.predict(X_test)

    end_time = time.perf_counter()

    average_time = (
        (end_time - start_time) / repetitions
    ) * 1000

    timing_results.append({

        "Model": name,

        "Prediction Time (ms)": round(
            average_time, 4
        )

    })

timing_df = pd.DataFrame(timing_results)

print(timing_df.to_string(index=False))

timing_df.to_csv(
    output_dir / "prediction_time.csv",
    index=False
)

# Prediction time chart
plt.figure(figsize=(8, 5))

plt.bar(
    timing_df["Model"],
    timing_df["Prediction Time (ms)"]
)

plt.title("Prediction Time Comparison")

plt.xlabel("Machine Learning Models")

plt.ylabel("Average Prediction Time (ms)")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    output_dir / "prediction_time_comparison.png"
)

plt.close()


# ============================================================
# 13. IDENTIFY BEST MODEL
# ============================================================

print("\n========== MODEL RECOMMENDATION ==========")

best_accuracy = metrics_df.loc[
    metrics_df["Testing Accuracy"].idxmax()
]

best_recall = metrics_df.loc[
    metrics_df["Recall (Fail)"].idxmax()
]

print(
    "Highest Test Accuracy:",
    best_accuracy["Model"]
)

print(
    "Best Fail Recall:",
    best_recall["Model"]
)

print(
    "Fail Recall Score:",
    best_recall["Recall (Fail)"]
)


# ============================================================
# 14. GENERATE FIVE OBSERVATIONS
# ============================================================

highest_accuracy = metrics_df.loc[
    metrics_df["Testing Accuracy"].idxmax()
]

highest_precision = metrics_df.loc[
    metrics_df["Precision (Fail)"].idxmax()
]

highest_f1 = metrics_df.loc[
    metrics_df["F1-Score (Fail)"].idxmax()
]

largest_gap = metrics_df.loc[
    metrics_df["Performance Gap"].idxmax()
]

fastest_model = timing_df.loc[
    timing_df["Prediction Time (ms)"].idxmin()
]

observations = [

    f"1. {highest_accuracy['Model']} achieved the "
    f"highest test accuracy of "
    f"{highest_accuracy['Testing Accuracy']:.4f}.",

    f"2. {best_recall['Model']} achieved the highest "
    f"Fail Recall of {best_recall['Recall (Fail)']:.4f}, "
    "making it the strongest model for identifying "
    "students who may fail.",

    f"3. {highest_precision['Model']} achieved the "
    f"highest Fail Precision of "
    f"{highest_precision['Precision (Fail)']:.4f}.",

    f"4. {largest_gap['Model']} had the largest "
    f"training/testing accuracy gap of "
    f"{largest_gap['Performance Gap']:.4f}.",

    f"5. {fastest_model['Model']} had the shortest "
    f"average prediction time of "
    f"{fastest_model['Prediction Time (ms)']:.4f} ms."
]


print("\n========== FIVE OBSERVATIONS ==========")

for observation in observations:

    print(observation)


# ============================================================
# 15. GENERATE FINAL ANALYSIS REPORT
# ============================================================

recommendation = f"""

MODEL RESULTS ANALYSIS DASHBOARD
================================

DATASET
-------
Student Pass/Fail Prediction Dataset

Models:
- Logistic Regression
- Decision Tree
- KNN


1. MODEL PERFORMANCE

{metrics_df.to_string(index=False)}


2. OVERFITTING ANALYSIS

{metrics_df[[
    "Model",
    "Training Accuracy",
    "Testing Accuracy",
    "Performance Gap"
]].to_string(index=False)}


3. FIVE OBSERVATIONS

{chr(10).join(observations)}


4. MODEL RECOMMENDATION

Recommended Model: {best_recall['Model']}

Test Accuracy: {best_recall['Testing Accuracy']}

Fail Precision: {best_recall['Precision (Fail)']}

Fail Recall: {best_recall['Recall (Fail)']}

Fail F1-Score: {best_recall['F1-Score (Fail)']}


5. CONFUSION MATRIX ANALYSIS

Fail is treated as the positive class.

False Positive:
A student actually passes but is predicted to fail.

False Negative:
A student actually fails but is predicted to pass.

For student risk identification, false negatives
are particularly important because the system
fails to identify students who need academic support.


6. FINAL CONCLUSION

The three models were compared using test accuracy,
precision, recall and F1-score.

The highest accuracy model is not automatically
the best choice for every situation.

For identifying students who may fail, Fail Recall
is particularly important.

The recommended model is the one with the highest
Fail Recall, provided its accuracy and precision
are also acceptable.

The training and testing accuracy comparison helps
identify possible overfitting.

The prediction time comparison provides additional
information about model efficiency.
"""


with open(
    output_dir / "model_analysis_report.txt",
    "w"
) as file:

    file.write(recommendation)


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n========== DAY48 COMPLETED ==========")

print("\nGenerated Files:")

for file in output_dir.iterdir():

    print("-", file.name)

print("\nDashboard analysis completed successfully!")