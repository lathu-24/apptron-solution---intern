from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "student_dataset.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
SEED = 42
TARGET = "Result (Pass/Fail)"
FEATURES = ["Study Hours", "Attendance Percentage",
            "Assignment Score", "Previous Exam Score"]


def make_demo_data(path, n=1000):
    """Fallback demo data only; use an approved real dataset for final results."""
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame({
        "Study Hours": np.clip(rng.normal(4.5, 2, n), 0, 10),
        "Attendance Percentage": np.clip(rng.normal(78, 13, n), 30, 100),
        "Assignment Score": np.clip(rng.normal(70, 15, n), 0, 100),
        "Previous Exam Score": np.clip(rng.normal(68, 16, n), 0, 100)
    })
    risk = (0.34*(10-x["Study Hours"])/10
            + 0.25*(100-x["Attendance Percentage"])/100
            + 0.21*(100-x["Assignment Score"])/100
            + 0.20*(100-x["Previous Exam Score"])/100
            + rng.normal(0, 0.055, n))
    x[TARGET] = np.where(risk >= 0.39, "Fail", "Pass")
    if x[TARGET].nunique() < 2:
        x.loc[:n//2, TARGET] = "Fail"
        x.loc[n//2+1:, TARGET] = "Pass"
    x.to_csv(path, index=False)
    return x


def load_data():
    if not DATA.exists():
        print("No dataset found. Creating a synthetic demo dataset with 1,000 rows.")
        df = make_demo_data(DATA)
    else:
        df = pd.read_csv(DATA)

    # Support common alternative column names.
    aliases = {
        "Study_Hours": "Study Hours",
        "Attendance_Percentage": "Attendance Percentage",
        "Assignment_Score": "Assignment Score",
        "Previous_Exam_Score": "Previous Exam Score",
        "Result": TARGET,
        "Pass/Fail": TARGET,
        "Result (Pass / Fail)": TARGET
    }
    df.columns = [str(c).strip() for c in df.columns]
    df.rename(columns={c: aliases[c] for c in df.columns if c in aliases}, inplace=True)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {FEATURES + [TARGET]}")

    df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = df[TARGET].astype("string").str.strip().str.title()
    df[TARGET] = df[TARGET].replace({"Failed": "Fail", "Passed": "Pass"})
    df = df[df[TARGET].isin(["Fail", "Pass"])]
    before = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = before - len(df)

    limits = {"Study Hours": (0, 24), "Attendance Percentage": (0, 100),
              "Assignment Score": (0, 100), "Previous Exam Score": (0, 100)}
    for col, (low, high) in limits.items():
        df.loc[~df[col].between(low, high), col] = np.nan

    if len(df) < 500:
        raise ValueError(f"Need at least 500 valid records; found {len(df)}.")
    if df[TARGET].nunique() != 2:
        raise ValueError("Target must include both Pass and Fail.")

    report = [
        "DATA INSPECTION", "="*40, f"Shape: {df.shape}",
        f"Duplicate rows removed: {duplicates_removed}", "",
        "Data types:", df.dtypes.to_string(), "",
        "Missing values:", df.isna().sum().to_string(), "",
        "Target counts:", df[TARGET].value_counts().to_string(), "",
        "Descriptive statistics:", df[FEATURES].describe().round(2).to_string()
    ]
    (OUT / "data_inspection_report.txt").write_text("\n".join(report), encoding="utf-8")
    df.to_csv(OUT / "clean_student_dataset.csv", index=False)
    return df


def run_eda(df):
    sns.set_theme()
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Student Result Distribution")
    plt.tight_layout()
    plt.savefig(OUT / "target_distribution.png", dpi=150)
    plt.close()

    for col in FEATURES:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue=TARGET, kde=True)
        plt.title(f"{col} by Student Result")
        plt.tight_layout()
        plt.savefig(OUT / f"eda_{col.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
    plt.close()


def metrics(y, pred):
    # Fail=1 is the positive class, so Fail recall measures at-risk students found.
    return {
        "Accuracy": accuracy_score(y, pred),
        "Fail Precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "Fail Recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "Fail F1": f1_score(y, pred, pos_label=1, zero_division=0)
    }


def make_models():
    scaled = ColumnTransformer([("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), FEATURES)])
    unscaled = ColumnTransformer([("numeric", SimpleImputer(strategy="median"), FEATURES)])

    return {
        "Logistic Regression": (
            Pipeline([("prep", scaled),
                      ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   random_state=SEED))]),
            {"model__C": [0.01, 0.1, 1, 10, 100], "model__penalty": ["l2"]}
        ),
        "Decision Tree": (
            Pipeline([("prep", unscaled),
                      ("model", DecisionTreeClassifier(class_weight="balanced",
                                                        random_state=SEED))]),
            {"model__max_depth": [2, 3, 5, 7, 10, None],
             "model__min_samples_split": [2, 5, 10],
             "model__min_samples_leaf": [1, 2, 5]}
        ),
        "KNN": (
            Pipeline([("prep", scaled), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
             "model__weights": ["uniform", "distance"], "model__p": [1, 2]}
        )
    }


def save_cm(y, pred, name):
    cm = confusion_matrix(y, pred, labels=[1, 0])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Fail", "Predicted Pass"],
                yticklabels=["Actual Fail", "Actual Pass"])
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(OUT / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
    plt.close()


def main():
    df = load_data()
    run_eda(df)
    X = df[FEATURES]
    # Fail=1 (positive/risk class); Pass=0.
    y = df[TARGET].map({"Pass": 0, "Fail": 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    # Save exact split for reproducibility.
    for name, values in [("X_train", X_train), ("X_test", X_test),
                         ("y_train", y_train), ("y_test", y_test)]:
        pd.DataFrame(values).to_csv(OUT / f"{name}.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scorers = {"accuracy": "accuracy", "precision": "precision",
               "recall": "recall", "f1": "f1"}
    validation, test_rows, gap_rows, time_rows = [], [], [], []
    fitted = {}

    for name, (pipe, grid) in make_models().items():
        print(f"Tuning {name}...")
        search = GridSearchCV(pipe, grid, scoring="recall", cv=cv,
                              n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        fitted[name] = model

        cv_result = cross_validate(model, X_train, y_train, cv=cv,
                                   scoring=scorers, n_jobs=-1)
        validation.append({
            "Model": name,
            "CV Accuracy": cv_result["test_accuracy"].mean(),
            "CV Precision": cv_result["test_precision"].mean(),
            "CV Recall": cv_result["test_recall"].mean(),
            "CV F1": cv_result["test_f1"].mean(),
            "Best Settings": str(search.best_params_)
        })

        train_pred = model.predict(X_train)
        start = time.perf_counter()
        test_pred = model.predict(X_test)
        elapsed = time.perf_counter() - start
        train_m, test_m = metrics(y_train, train_pred), metrics(y_test, test_pred)

        test_rows.append({"Model": name, **test_m})
        gap_rows.append({
            "Model": name,
            "Train Accuracy": train_m["Accuracy"],
            "Test Accuracy": test_m["Accuracy"],
            "Accuracy Gap": train_m["Accuracy"] - test_m["Accuracy"],
            "Train Fail Recall": train_m["Fail Recall"],
            "Test Fail Recall": test_m["Fail Recall"],
            "Fail Recall Gap": train_m["Fail Recall"] - test_m["Fail Recall"]
        })
        time_rows.append({
            "Model": name,
            "Prediction Time (seconds)": elapsed,
            "Prediction Time (ms per record)": elapsed / len(X_test) * 1000
        })
        save_cm(y_test, test_pred, name)
        joblib.dump(model, MODELS / f"{name.lower().replace(' ', '_')}.joblib")
        (OUT / f"{name.lower().replace(' ', '_')}_classification_report.txt").write_text(
            classification_report(y_test, test_pred, target_names=["Pass", "Fail"],
                                  zero_division=0), encoding="utf-8"
        )

    val_df = pd.DataFrame(validation)
    test_df = pd.DataFrame(test_rows)
    gap_df = pd.DataFrame(gap_rows)
    time_df = pd.DataFrame(time_rows)
    val_df.to_csv(OUT / "validation_comparison.csv", index=False)
    test_df.to_csv(OUT / "test_model_comparison.csv", index=False)
    gap_df.to_csv(OUT / "training_testing_comparison.csv", index=False)
    time_df.to_csv(OUT / "prediction_time_comparison.csv", index=False)

    # Dashboard charts.
    plt.figure(figsize=(8, 5))
    sns.barplot(data=test_df, x="Model", y="Accuracy")
    plt.ylim(0, 1); plt.title("Test Accuracy Comparison")
    plt.xticks(rotation=10); plt.tight_layout()
    plt.savefig(OUT / "dashboard_accuracy.png", dpi=150); plt.close()

    long = test_df.melt(id_vars="Model",
                        value_vars=["Fail Precision", "Fail Recall", "Fail F1"],
                        var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1); plt.title("Test Precision, Recall and F1")
    plt.tight_layout(); plt.savefig(OUT / "dashboard_metrics.png", dpi=150); plt.close()

    gap_long = gap_df.melt(id_vars="Model",
                           value_vars=["Train Accuracy", "Test Accuracy"],
                           var_name="Dataset", value_name="Accuracy")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=gap_long, x="Model", y="Accuracy", hue="Dataset")
    plt.ylim(0, 1); plt.title("Training vs Testing Accuracy")
    plt.tight_layout(); plt.savefig(OUT / "dashboard_train_test_gap.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=time_df, x="Model", y="Prediction Time (ms per record)")
    plt.title("Prediction Time per Record")
    plt.tight_layout(); plt.savefig(OUT / "dashboard_prediction_time.png", dpi=150); plt.close()

    # Bonus 1: Decision Tree depth comparison using training-set CV only.
    depth_rows = []
    for depth in [1, 2, 3, 4, 5, 7, 10, None]:
        candidate = Pipeline([
            ("prep", ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median"), FEATURES)])),
            ("model", DecisionTreeClassifier(max_depth=depth, class_weight="balanced",
                                             random_state=SEED))
        ])
        scores = cross_validate(candidate, X_train, y_train, cv=cv,
                                scoring=scorers, n_jobs=-1)
        depth_rows.append({"Max Depth": str(depth),
                           **{f"CV {k.title()}": scores[f"test_{k}"].mean()
                              for k in scorers}})
    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(OUT / "decision_tree_depth_comparison.csv", index=False)

    # Recommendation prioritises Fail recall; F1 and accuracy break ties.
    winner = test_df.sort_values(
        ["Fail Recall", "Fail F1", "Accuracy"], ascending=False
    ).iloc[0]
    recommended = winner["Model"]
    observations = []
    for _, row in test_df.iterrows():
        observations.append(
            f"{row['Model']}: accuracy {row['Accuracy']:.3f}, "
            f"Fail precision {row['Fail Precision']:.3f}, "
            f"Fail recall {row['Fail Recall']:.3f}, Fail F1 {row['Fail F1']:.3f}."
        )
    for _, row in gap_df.iterrows():
        gap = row["Accuracy Gap"]
        note = ("large gap: possible overfitting" if gap > .10 else
                "moderate gap: inspect possible overfitting" if gap > .05 else
                "small gap: no strong accuracy-gap warning")
        observations.append(f"{row['Model']}: train-test accuracy gap {gap:.3f} ({note}).")

    source_note = ("Synthetic demonstration dataset generated by this script."
                   if not DATA.exists() else
                   "Dataset loaded from data/student_dataset.csv; add its original source details.")
    report = f"""FINAL PRACTICAL TASK – RESULTS ANALYSIS
========================================

Objective
Predict students at risk of failing from academic indicators.

Dataset and source
{source_note}
Cleaned records: {len(df)}
Features: {', '.join(FEATURES)}
Target: {TARGET}
Target definition: Fail=1 (student is at risk/has a failing result);
Pass=0 (student has a passing result).

Preprocessing and EDA
Removed duplicate rows, converted feature values to numeric, treated
out-of-range values as missing, and used median imputation inside pipelines.
EDA charts and the data inspection report are saved in outputs/.

Experiment setup
Stratified 80/20 train/test split; random_state={SEED}.
Five-fold StratifiedKFold cross-validation on training data only.
Grid search selected settings by Fail recall. Test set was held out
from model selection.

Validation comparison
{val_df.to_string(index=False)}

Held-out test comparison
{test_df.to_string(index=False)}

Training/testing comparison
{gap_df.to_string(index=False)}

Observations
""" + "\n".join(f"{i+1}. {x}" for i, x in enumerate(observations)) + f"""

Recommendation
Recommended model: {recommended}
It achieved the highest held-out Fail recall (with Fail F1 and accuracy
as tie-breakers). Fail recall matters because a false negative is a student
who actually fails but is not identified for support. Also review precision:
false positives can lead to unnecessary interventions.

Limitations
Model performance depends on data quality and representativeness. Synthetic
data is for demonstrating the workflow only, not evidence of real-world
student outcomes. Use predictions to support, not replace, educator judgement.
"""
    (OUT / "final_results_analysis_report.txt").write_text(report, encoding="utf-8")
    (OUT / "dashboard_summary.json").write_text(json.dumps({
        "recommended_model": recommended,
        "train_records": len(X_train), "test_records": len(X_test),
        "test_metrics": test_rows
    }, indent=2), encoding="utf-8")

    print("\nTEST RESULTS\n", test_df.to_string(index=False))
    print("\nTRAIN/TEST GAPS\n", gap_df.to_string(index=False))
    print(f"\nRecommended model: {recommended}")
    print(f"All outputs saved in: {OUT}")


if __name__ == "__main__":
    main()
