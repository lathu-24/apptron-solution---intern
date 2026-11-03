

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. PROJECT SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR /"customer_purchase_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Purchased"
FEATURES = [
    "Age",
    "Income",
    "Website Visits",
    "Time Spent",
    "Previous Purchases",
    "Product Views"
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
K_VALUES = [3, 5, 7, 9, 11, 13, 15, 17, 19]


# ------------------------------------------------------------
# 2. LOAD AND INSPECT DATA
# ------------------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place customer_purchase_dataset.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 65)
    print("1. DATASET INSPECTION")
    print("=" * 65)
    print("Shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nColumn data types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    required = FEATURES + [TARGET]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from dataset: {missing_columns}")

    return df


# ------------------------------------------------------------
# 3. CLEAN DATA AND PREPARE TARGET
# ------------------------------------------------------------

def clean_dataset(df):
    df = df[FEATURES + [TARGET]].copy()
    rows_before = len(df)

    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURES + [TARGET])

    # Convert target values such as Yes/No or 1/0 to numeric labels.
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        target_map = {
            "yes": 1, "no": 0,
            "true": 1, "false": 0,
            "purchased": 1, "not purchased": 0
        }
        normalized = df[TARGET].astype(str).str.strip().str.lower()
        mapped = normalized.map(target_map)
        if mapped.isna().any():
            raise ValueError(
                "Target must contain 0/1 or Yes/No (or Purchased/Not Purchased)."
            )
        df[TARGET] = mapped.astype(int)
    else:
        unique_values = set(df[TARGET].unique())
        if unique_values.issubset({0, 1, 0.0, 1.0}):
            df[TARGET] = df[TARGET].astype(int)
        else:
            raise ValueError(
                f"Unexpected target values: {sorted(unique_values)}. "
                "Expected binary values 0 and 1."
            )

    # Ensure all input features are numeric for KNN.
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)

    print("\nRows before cleaning:", rows_before)
    print("Rows after cleaning:", len(df))
    print("Rows removed:", rows_before - len(df))
    print("\nTarget distribution:\n", df[TARGET].value_counts().sort_index())
    print("\nTarget distribution (%):\n",
          (df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2))

    if df[TARGET].nunique() != 2:
        raise ValueError("The target must contain both classes (0 and 1).")

    return df


# ------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

def perform_eda(df):
    print("\n" + "=" * 65)
    print("2. EXPLORATORY DATA ANALYSIS")
    print("=" * 65)
    print("\nDescriptive statistics:\n", df[FEATURES].describe().round(2))

    # Target class distribution
    counts = df[TARGET].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    plt.bar(["Not Purchased (0)", "Purchased (1)"],
            [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Customer Purchase Class Distribution")
    plt.xlabel("Purchase Outcome")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Income vs time spent, grouped by purchase outcome
    plt.figure(figsize=(8, 6))
    for label, name in [(0, "Not Purchased"), (1, "Purchased")]:
        subset = df[df[TARGET] == label]
        plt.scatter(
            subset["Income"], subset["Time Spent"],
            alpha=0.45, label=name
        )
    plt.title("Income vs Time Spent on Website")
    plt.xlabel("Monthly Income")
    plt.ylabel("Time Spent on Website")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "income_vs_time_spent.png", dpi=150)
    plt.close()

    # Feature correlation heatmap using Matplotlib only
    corr = df[FEATURES + [TARGET]].corr()
    plt.figure(figsize=(10, 8))
    image = plt.imshow(corr, aspect="auto")
    plt.colorbar(image, label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close()

    print("EDA charts saved in:", OUTPUT_DIR)


# ------------------------------------------------------------
# 5. SPLIT FEATURES AND TARGET
# ------------------------------------------------------------

def split_data(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n" + "=" * 65)
    print("3. TRAIN / TEST SPLIT")
    print("=" * 65)
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Training class distribution:\n", y_train.value_counts().sort_index())
    print("Testing class distribution:\n", y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# 6. SELECT K USING CROSS-VALIDATION ON TRAINING DATA ONLY
# ------------------------------------------------------------

def select_best_k(X_train, y_train):
    print("\n" + "=" * 65)
    print("4. K VALUE SELECTION USING CROSS-VALIDATION")
    print("=" * 65)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    comparison_rows = []

    for k in K_VALUES:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        comparison_rows.append({
            "K": k,
            "Mean CV F1": scores.mean(),
            "Std CV F1": scores.std(),
            "Mean CV Accuracy": cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="accuracy"
            ).mean()
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "k_value_comparison.csv", index=False)

    print(comparison.round(4).to_string(index=False))

    best_row = comparison.loc[comparison["Mean CV F1"].idxmax()]
    best_k = int(best_row["K"])
    print(f"\nSelected K: {best_k}")
    print(f"Mean validation F1-score: {best_row['Mean CV F1']:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        comparison["K"],
        comparison["Mean CV F1"],
        marker="o"
    )
    plt.title("KNN: K Value vs 5-Fold Cross-Validation F1-score")
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Mean Cross-Validation F1-score")
    plt.xticks(K_VALUES)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "k_value_comparison.png", dpi=150)
    plt.close()

    return best_k, comparison


# ------------------------------------------------------------
# 7. TRAIN FINAL PIPELINE AND EVALUATE ON TEST SET ONCE
# ------------------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, best_k):
    print("\n" + "=" * 65)
    print("5. FINAL MODEL TEST EVALUATION")
    print("=" * 65)

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k))
    ])

    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0)
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Not Purchased", "Purchased"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion Matrix:\n", cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Purchased", "Purchased"]
    )
    display.plot(values_format="d")
    plt.title("Final KNN Model - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "Selected K", best_k)
    metrics_df.to_csv(OUTPUT_DIR / "final_evaluation_metrics.csv", index=False)

    # Error review: false positives and false negatives
    error_review = X_test.copy()
    error_review["Actual"] = y_test
    error_review["Predicted"] = y_pred
    error_review["Error Type"] = np.where(
        error_review["Actual"] == error_review["Predicted"],
        "Correct",
        np.where(
            (error_review["Actual"] == 0) & (error_review["Predicted"] == 1),
            "False Positive",
            "False Negative"
        )
    )
    error_review.to_csv(OUTPUT_DIR / "test_prediction_error_review.csv", index=False)

    print("\nError counts:\n", error_review["Error Type"].value_counts())

    return final_model, metrics, cm


# ------------------------------------------------------------
# 8. PREDICT AT LEAST TEN NEW CUSTOMERS
# ------------------------------------------------------------

def predict_new_customers(model):
    print("\n" + "=" * 65)
    print("6. PREDICTIONS FOR NEW CUSTOMERS")
    print("=" * 65)

    # Illustrative customer profiles. These are example inputs,
    # not records from the training or test dataset.
    new_customers = pd.DataFrame([
        [22, 35000, 3, 5.5, 0, 4],
        [28, 55000, 8, 12.0, 2, 10],
        [35, 80000, 15, 25.0, 5, 20],
        [42, 95000, 20, 35.0, 8, 30],
        [31, 45000, 5, 8.0, 1, 7],
        [50, 120000, 25, 42.0, 12, 35],
        [26, 60000, 10, 18.0, 3, 15],
        [38, 70000, 12, 22.0, 4, 18],
        [45, 30000, 2, 3.0, 0, 2],
        [29, 110000, 18, 30.0, 9, 28],
    ], columns=FEATURES)

    predictions = model.predict(new_customers)
    probabilities = model.predict_proba(new_customers)

    results = new_customers.copy()
    results["Predicted Purchase"] = np.where(
        predictions == 1, "Yes", "No"
    )
    results["Purchase Probability"] = probabilities[:, 1].round(4)

    print(results.to_string(index=False))
    results.to_csv(OUTPUT_DIR / "new_customer_predictions.csv", index=False)

    return results


# ------------------------------------------------------------
# 9. GENERATE PROJECT REPORT
# ------------------------------------------------------------

def generate_report(df, best_k, metrics, cm, comparison):
    report = f"""
CUSTOMER PURCHASE PREDICTION SYSTEM
===================================

1. PROJECT OVERVIEW
The goal is to predict whether a customer will purchase a product
using customer demographic and website engagement features.

2. DATASET
Dataset file: customer_purchase_dataset.csv
Dataset size after cleaning: {len(df)} rows
Features: {", ".join(FEATURES)}
Target: Purchased (0 = No, 1 = Yes)

DATA DISCLAIMER:
The supplied educational dataset is synthetic/generated data.
It is not a real customer dataset and should not be used to make
real business decisions.

3. DATA PREPARATION
- Removed duplicate records.
- Removed rows with missing values.
- Used a stratified 80/20 train/test split.
- Applied StandardScaler inside a Scikit-learn Pipeline.
- The scaler was fitted on training data only.

4. MODEL SELECTION
Algorithm: K-Nearest Neighbors (KNN)
Cross-validation: 5-fold StratifiedKFold
Selection metric: Mean validation F1-score
Selected K: {best_k}

K VALUE COMPARISON
{comparison.round(4).to_string(index=False)}

5. FINAL TEST RESULTS
Accuracy: {metrics['Accuracy']:.4f}
Precision: {metrics['Precision']:.4f}
Recall: {metrics['Recall']:.4f}
F1-score: {metrics['F1-score']:.4f}

Confusion matrix (rows = actual, columns = predicted):
{cm}

6. INTERPRETATION
Accuracy measures the overall proportion of correct predictions.
Precision measures how many predicted purchasers actually purchased.
Recall measures how many actual purchasers were identified.
F1-score balances precision and recall.

7. LIMITATIONS
- The dataset is synthetic, so its patterns may not represent real customers.
- KNN is sensitive to feature scaling and the choice of K.
- KNN prediction can become slower as the training dataset grows.
- Customer behavior can change over time.
- Predictions show patterns in the supplied dataset, not certainty about
  an individual's future behavior.
- The example new-customer profiles are illustrative.

8. CONCLUSION
The final KNN model was selected using cross-validation on training data.
The test set was kept separate and used for final evaluation only.
The model can demonstrate a basic purchase prediction workflow, but
real-world deployment would require validated real customer data,
privacy safeguards, monitoring and further testing.
"""

    report_path = OUTPUT_DIR / "project_report.txt"
    report_path.write_text(report.strip(), encoding="utf-8")
    print("\nReport saved to:", report_path)


# ------------------------------------------------------------
# 10. MAIN PROGRAM
# ------------------------------------------------------------

def main():
    df = load_dataset()
    df = clean_dataset(df)
    perform_eda(df)

    X_train, X_test, y_train, y_test = split_data(df)

    best_k, comparison = select_best_k(X_train, y_train)

    model, metrics, cm = train_and_evaluate(
        X_train, X_test, y_train, y_test, best_k
    )

    predict_new_customers(model)

    generate_report(df, best_k, metrics, cm, comparison)

    print("\n" + "=" * 65)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print("Check the outputs folder for charts, CSV files and report.")


if __name__ == "__main__":
    main()
