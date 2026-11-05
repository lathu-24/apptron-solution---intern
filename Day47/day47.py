# ============================================================
# DAY 47 - THREE-MODEL COMPARISON SYSTEM
# Student Pass/Fail Prediction
# Models: Logistic Regression, Decision Tree, KNN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)

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
    confusion_matrix,
    make_scorer
)


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("\n========================================")
print(" THREE-MODEL COMPARISON SYSTEM")
print("========================================")


# ============================================================
# 2. LOAD STUDENT DATASET
# ============================================================

print("\n========== LOADING DATASET ==========")

data = pd.read_csv("student_dataset.csv")

print("Dataset Shape:", data.shape)

print("\nDataset Columns:")
print(data.columns.tolist())

print("\nFirst 5 Records:")
print(data.head())

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Records:", data.duplicated().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n========== DATA CLEANING ==========")

data = data.drop_duplicates()

data = data.dropna()

# Separate features and target
X = data.drop("Result", axis=1)

y = data["Result"].str.strip().str.capitalize()

# Encode target variable
# Fail = 0
# Pass = 1

y = y.map({
    "Fail": 0,
    "Pass": 1
})

if y.isnull().any():
    raise ValueError("Unexpected values found in Result column.")

y = y.astype(int)

print("Cleaned Dataset Shape:", data.shape)

print("\nClass Distribution:")
print(data["Result"].value_counts())


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

print("\n========== TRAIN-TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# Save split files for future reuse

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_frame(name="Result").to_csv(
    "y_train.csv", index=False
)

y_test.to_frame(name="Result").to_csv(
    "y_test.csv", index=False
)

print("Train-test split saved successfully.")


# ============================================================
# 5. CROSS-VALIDATION SETTINGS
# ============================================================

print("\n========== CROSS-VALIDATION ==========")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Fail = 0 is the positive class for evaluation.
# We prioritize identifying students who may fail.

fail_recall_scorer = make_scorer(
    recall_score,
    pos_label=0,
    zero_division=0
)


# ============================================================
# 6. DEFINE THREE MODEL PIPELINES
# ============================================================

models = {

    "Logistic Regression": {

        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "parameters": {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }

    },


    "Decision Tree": {

        "pipeline": Pipeline([
            ("model", DecisionTreeClassifier(
                random_state=42
            ))
        ]),

        "parameters": {
            "model__max_depth": [2, 3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5]
        }

    },


    "KNN": {

        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),

        "parameters": {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"]
        }

    }

}


# ============================================================
# 7. TRAIN AND TUNE ALL THREE MODELS
# ============================================================

print("\n========== MODEL TRAINING ==========")

best_models = {}

validation_results = []

for name, settings in models.items():

    print(f"\nTraining {name}...")

    grid_search = GridSearchCV(
        estimator=settings["pipeline"],
        param_grid=settings["parameters"],
        scoring=fail_recall_scorer,
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    best_models[name] = grid_search.best_estimator_

    validation_results.append({

        "Model": name,

        "CV Fail Recall": round(
            grid_search.best_score_, 4
        ),

        "Best Parameters": str(
            grid_search.best_params_
        )

    })

    print("Best Parameters:")
    print(grid_search.best_params_)

    print(
        "Best CV Fail Recall:",
        round(grid_search.best_score_, 4)
    )


# ============================================================
# 8. SAVE VALIDATION RESULTS
# ============================================================

validation_df = pd.DataFrame(validation_results)

print("\n========== VALIDATION RESULTS ==========")

print(validation_df.to_string(index=False))

validation_df.to_csv(
    output_dir / "validation_results.csv",
    index=False
)


# ============================================================
# 9. EVALUATE MODELS ON SAME TEST DATA
# ============================================================

print("\n========== FINAL MODEL EVALUATION ==========")

comparison_results = {}

predictions = {}

for name, model in best_models.items():

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    accuracy = accuracy_score(
        y_test, y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    comparison_results[name] = {

        "Accuracy": round(accuracy, 4),

        "Fail Precision": round(precision, 4),

        "Fail Recall": round(recall, 4),

        "Fail F1-Score": round(f1, 4)

    }

    print(f"\n{name}")

    print("Accuracy:", round(accuracy, 4))

    print("Fail Precision:", round(precision, 4))

    print("Fail Recall:", round(recall, 4))

    print("Fail F1-Score:", round(f1, 4))


# ============================================================
# 10. CREATE MODEL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame.from_dict(
    comparison_results,
    orient="index"
)

comparison_df.index.name = "Model"

comparison_df = comparison_df.reset_index()

print("\n========== MODEL COMPARISON TABLE ==========")

print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)


# ============================================================
# 11. GENERATE CONFUSION MATRICES
# ============================================================

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


# ============================================================
# 12. SAVE TRAINED MODELS
# ============================================================

print("\n========== SAVING TRAINED MODELS ==========")

for name, model in best_models.items():

    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        output_dir / f"{filename}_model.pkl"
    )

    print(f"{name} saved successfully.")


# ============================================================
# 13. BONUS - DECISION TREE DEPTH COMPARISON
# ============================================================

print("\n========== DECISION TREE DEPTH COMPARISON ==========")

depth_values = [1, 2, 3, 5, 7, 10, None]

depth_results = []

for depth in depth_values:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    scores = cross_val_score(
        tree,
        X_train,
        y_train,
        cv=cv,
        scoring=fail_recall_scorer,
        n_jobs=-1
    )

    depth_results.append({

        "Maximum Depth": str(depth),

        "Average CV Fail Recall": round(
            scores.mean(), 4
        )

    })

depth_df = pd.DataFrame(depth_results)

print(depth_df.to_string(index=False))

depth_df.to_csv(
    output_dir / "decision_tree_depth_comparison.csv",
    index=False
)


# Plot Decision Tree depth comparison

plt.figure(figsize=(8, 5))

plt.plot(
    depth_df["Maximum Depth"],
    depth_df["Average CV Fail Recall"],
    marker="o"
)

plt.title("Decision Tree Maximum Depth Comparison")

plt.xlabel("Maximum Depth")

plt.ylabel("Average CV Fail Recall")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    output_dir / "decision_tree_depth_comparison.png"
)

plt.close()


# ============================================================
# 14. IDENTIFY BEST MODELS
# ============================================================

print("\n========== MODEL RECOMMENDATION ==========")

best_accuracy = comparison_df.loc[
    comparison_df["Accuracy"].idxmax()
]

best_recall = comparison_df.loc[
    comparison_df["Fail Recall"].idxmax()
]

print(
    "Best Model Based on Accuracy:",
    best_accuracy["Model"]
)

print(
    "Highest Accuracy:",
    best_accuracy["Accuracy"]
)

print(
    "\nBest Model for Identifying Students Who May Fail:",
    best_recall["Model"]
)

print(
    "Highest Fail Recall:",
    best_recall["Fail Recall"]
)


# ============================================================
# 15. GENERATE WRITTEN RESULTS ANALYSIS
# ============================================================

analysis = f"""

THREE-MODEL COMPARISON SYSTEM
STUDENT PASS/FAIL PREDICTION
==============================================

DATASET INFORMATION

Total Records: {len(data)}

Training Records: {len(X_train)}

Testing Records: {len(X_test)}

Features:
- Study Hours
- Attendance Percentage
- Assignment Score
- Previous Exam Score

Target:
- Result (Pass/Fail)


MODEL COMPARISON

{comparison_df.to_string(index=False)}


1. LOGISTIC REGRESSION

Logistic Regression is a classification algorithm
that predicts the probability of a student passing
or failing.

StandardScaler is used because the features have
different numerical ranges.


2. DECISION TREE

Decision Tree makes predictions using a sequence
of decision rules.

Maximum depth and other parameters are tuned
using cross-validation.


3. K-NEAREST NEIGHBORS

KNN predicts the result based on the nearest
training records.

StandardScaler is important because KNN uses
distance calculations.


4. FALSE POSITIVE

A false positive occurs when a student who actually
passes is incorrectly predicted to fail.


5. FALSE NEGATIVE

A false negative occurs when a student who actually
fails is incorrectly predicted to pass.

False negatives are important because the system
fails to identify students who need academic support.


6. BEST MODEL BASED ON ACCURACY

Model: {best_accuracy['Model']}

Accuracy: {best_accuracy['Accuracy']}


7. BEST MODEL FOR IDENTIFYING STUDENTS WHO MAY FAIL

Model: {best_recall['Model']}

Fail Recall: {best_recall['Fail Recall']}


8. FINAL CONCLUSION

The three classification algorithms were trained
and evaluated using the same training and testing
records.

Accuracy measures the overall percentage of
correct predictions.

However, accuracy alone is not sufficient for
student risk identification.

Fail Recall is particularly important because
it measures how many actual failing students
are correctly identified.

The model with the highest Fail Recall is the
strongest candidate for identifying students
who may need additional academic support.

The final model recommendation should consider
both Accuracy and Fail Recall, along with
Precision and F1-Score.

"""


with open(
    output_dir / "model_recommendation.txt",
    "w"
) as file:

    file.write(analysis)


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n========== PROJECT COMPLETED ==========")

print(analysis)

print("\nGenerated Files:")

for file in output_dir.iterdir():

    print("-", file.name)

print("\nDay47 Three-Model Comparison System Completed!")