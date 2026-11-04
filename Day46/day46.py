import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split


# ------------------------------------------------
# 1. CREATE OUTPUT FOLDER
# ------------------------------------------------

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


# ------------------------------------------------
# 2. LOAD DATASET AND TRAIN-TEST SPLIT
# ------------------------------------------------

print("\n========== DATASET LOADING ==========")

if all(Path(file).exists() for file in [
    "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"
]):

    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")

    y_train = pd.read_csv("y_train.csv").squeeze()
    y_test = pd.read_csv("y_test.csv").squeeze()

    print("Using the saved Day45 train-test split.")

else:

    data = pd.read_csv("student_dataset.csv")

    X = data.drop("Result", axis=1)
    y = data["Result"]

    # Encode Fail = 0 and Pass = 1
    if y.dtype == "object":
        y = y.str.strip().str.capitalize().map({
            "Fail": 0,
            "Pass": 1
        })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Created a reproducible train-test split.")


# Ensure target values are integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------
# 3. DEFINE CLASSIFICATION MODELS
# ------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

}


# ------------------------------------------------
# 4. TRAIN MODELS AND GENERATE PREDICTIONS
# ------------------------------------------------

print("\n========== MODEL TRAINING ==========")

results = []
predictions = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Class 0 represents Fail
    precision = precision_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision (Fail)": round(precision, 4),
        "Recall (Fail)": round(recall, 4),
        "F1-Score (Fail)": round(f1, 4)
    })

    print(f"{name} completed.")


# ------------------------------------------------
# 5. DISPLAY METRICS DATAFRAME
# ------------------------------------------------

print("\n========== MODEL COMPARISON ==========")

metrics_df = pd.DataFrame(results)

print(metrics_df.to_string(index=False))

metrics_df.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)


# ------------------------------------------------
# 6. GENERATE CONFUSION MATRICES
# ------------------------------------------------

print("\n========== CONFUSION MATRICES ==========")

for name, y_pred in predictions.items():

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print(f"\n{name} Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 4))

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


# ------------------------------------------------
# 7. IDENTIFY THE BEST MODEL FOR FAIL DETECTION
# ------------------------------------------------

print("\n========== FAIL DETECTION ANALYSIS ==========")

best_model = metrics_df.loc[
    metrics_df["Recall (Fail)"].idxmax()
]

print(
    "Best model for identifying students who may fail:",
    best_model["Model"]
)

print(
    "Fail Recall:",
    best_model["Recall (Fail)"]
)


# ------------------------------------------------
# 8. FALSE POSITIVE AND FALSE NEGATIVE EXPLANATION
# ------------------------------------------------

print("\n========== CONFUSION MATRIX EXPLANATION ==========")

print("""
False Positive:
A student who actually passes but is predicted to fail.

False Negative:
A student who actually fails but is predicted to pass.

For this project:
False Negatives are particularly important because
the system fails to identify students who may fail.
""")


# ------------------------------------------------
# 9. GENERATE AUTOMATIC RESULTS ANALYSIS
# ------------------------------------------------

best_accuracy = metrics_df.loc[
    metrics_df["Accuracy"].idxmax()
]

best_recall = metrics_df.loc[
    metrics_df["Recall (Fail)"].idxmax()
]

analysis = f"""
STUDENT RESULT PREDICTION - RESULTS ANALYSIS
=============================================

1. MODEL COMPARISON

{metrics_df.to_string(index=False)}


2. FALSE POSITIVE

A false positive occurs when a student who actually
passes is incorrectly predicted to fail.


3. FALSE NEGATIVE

A false negative occurs when a student who actually
fails is incorrectly predicted to pass.

False negatives are important because the system
fails to identify students who need additional support.


4. BEST MODEL BASED ON ACCURACY

{best_accuracy['Model']}

Accuracy: {best_accuracy['Accuracy']}


5. BEST MODEL FOR IDENTIFYING STUDENTS WHO MAY FAIL

{best_recall['Model']}

Fail Recall: {best_recall['Recall (Fail)']}


6. CONCLUSION

The models were compared using accuracy, precision,
recall and F1-score.

The model with the highest accuracy provides the
best overall percentage of correct predictions.

However, identifying students who may fail is also
important. Therefore, recall for the Fail class is
particularly important.

The model with the highest Fail recall is the best
at identifying students who may need additional
academic support.

The final model selection should consider both
accuracy and Fail recall rather than accuracy alone.
"""

with open(
    output_dir / "results_analysis.txt",
    "w"
) as file:

    file.write(analysis)


# ------------------------------------------------
# 10. FINAL OUTPUT
# ------------------------------------------------

print("\n========== PROJECT COMPLETED ==========")

print("Generated files:")

for file in output_dir.iterdir():
    print("-", file.name)

print("\nResults analysis:")
print(analysis)