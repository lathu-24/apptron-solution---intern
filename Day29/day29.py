

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook", palette="deep")

# ---------------------------- CONFIGURATION ----------------------------
DATA_FILE = Path("teleco customer.csv")
OUTPUT_DIR = Path("teleco_eda_output")
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42

# ---------------------------- HELPER FUNCTIONS ----------------------------
def save_plot(filename):
    """Save the current Matplotlib figure in the plot folder."""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def write_text(filename, content):
    (REPORT_DIR / filename).write_text(content, encoding="utf-8")


def safe_rate(numerator, denominator):
    return float(numerator / denominator * 100) if denominator else np.nan


def fmt(value, digits=2):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def top_category_rate(data, category_col, target_col=TARGET, positive="Yes"):
    """Return category churn rates, sorted high to low."""
    temp = data[[category_col, target_col]].dropna()
    rates = temp.groupby(category_col, observed=False)[target_col].apply(
        lambda s: (s.astype(str).str.strip().eq(positive).mean() * 100)
    )
    return rates.sort_values(ascending=False)


# ---------------------------- INITIAL SETUP ----------------------------
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find '{DATA_FILE}'. Put this Python script in the same folder "
        "as the CSV file, or update DATA_FILE at the top of the script."
    )

raw = pd.read_csv(DATA_FILE)
df = raw.copy()
df.columns = df.columns.str.strip()

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found. Columns: {df.columns.tolist()}")

# Normalize whitespace-only strings to missing values.
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

# TotalCharges is numeric in meaning, but can be imported as text due to blanks.
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Normalize target text if present.
df[TARGET] = df[TARGET].astype("string").str.strip()

# ---------------------------- PART 1: DATASET UNDERSTANDING ----------------------------
initial_rows, initial_cols = raw.shape
duplicate_count = int(df.duplicated().sum())
duplicate_id_count = (
    int(df[ID_COLUMN].duplicated().sum())
    if ID_COLUMN in df.columns else 0
)

missing_before = df.isna().sum().sort_values(ascending=False)
missing_before = missing_before[missing_before > 0]

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
identifier_cols = [ID_COLUMN] if ID_COLUMN in df.columns else []

profile_lines = [
    "TELCO CUSTOMER CHURN — DATASET PROFILE",
    "=" * 50,
    f"Rows: {initial_rows:,}",
    f"Columns: {initial_cols:,}",
    f"Exact duplicate rows: {duplicate_count:,}",
    f"Duplicate customer IDs: {duplicate_id_count:,}",
    "",
    "DATA TYPES",
    "-" * 50,
    df.dtypes.astype(str).to_string(),
    "",
    "NUMERICAL COLUMNS",
    "-" * 50,
    ", ".join(numeric_cols) if numeric_cols else "None",
    "",
    "CATEGORICAL / TEXT COLUMNS",
    "-" * 50,
    ", ".join(categorical_cols) if categorical_cols else "None",
    "",
    "MISSING VALUES AFTER TYPE CONVERSION",
    "-" * 50,
    missing_before.to_string() if len(missing_before) else "No missing values detected.",
    "",
    "STATISTICAL SUMMARY — NUMERICAL",
    "-" * 50,
    df[numeric_cols].describe().round(3).to_string() if numeric_cols else "No numerical columns.",
    "",
    "STATISTICAL SUMMARY — CATEGORICAL",
    "-" * 50,
    df[categorical_cols].describe().to_string() if categorical_cols else "No categorical columns.",
]
write_text("dataset_profile.txt", "\n".join(profile_lines))

df.dtypes.astype(str).rename("data_type").to_csv(REPORT_DIR / "data_types.csv")
df.isna().sum().rename("missing_count").to_csv(REPORT_DIR / "missing_values.csv")
df.describe(include="all").transpose().to_csv(REPORT_DIR / "descriptive_statistics.csv")

# ---------------------------- DATA CLEANING FOR ANALYSIS ----------------------------
# Remove exact duplicate rows only. Do not remove duplicate customer IDs automatically:
# report them for investigation instead.
analysis = df.drop_duplicates().copy()

# Fill missing numeric values with the median for plots/ML preparation.
# Keep a separate report of the original missingness before imputation.
for col in analysis.select_dtypes(include=np.number).columns:
    if analysis[col].isna().any():
        analysis[col] = analysis[col].fillna(analysis[col].median())

# Fill missing categorical values with an explicit label, preserving the fact that
# the original value was missing.
for col in analysis.select_dtypes(include=["object", "string", "category"]).columns:
    if analysis[col].isna().any():
        analysis[col] = analysis[col].fillna("Unknown")

# ---------------------------- PART 2: UNIVARIATE ANALYSIS ----------------------------
analysis_numeric = analysis.select_dtypes(include=np.number).columns.tolist()
# ID fields are excluded from charts and feature analysis.
analysis_numeric = [c for c in analysis_numeric if c not in identifier_cols]
analysis_categorical = [
    c for c in analysis.select_dtypes(include=["object", "string", "category"]).columns
    if c not in identifier_cols
]

# Histograms + KDE/distribution plots for every numeric feature.
for col in analysis_numeric:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=analysis, x=col, kde=True, bins=30, color="steelblue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Customer count")
    save_plot(f"univariate_distribution_{col}.png")

# Count plots for categorical columns. Very high-cardinality columns are limited to
# the most common 15 categories to keep the dashboard readable.
for col in analysis_categorical:
    plt.figure(figsize=(9, 5))
    order = analysis[col].value_counts().head(15).index
    sns.countplot(data=analysis, y=col, order=order, color="steelblue")
    plt.title(f"Count Plot: {col}")
    plt.xlabel("Customer count")
    plt.ylabel(col)
    save_plot(f"countplot_{col}.png")

# Box plots for numeric variables.
for col in analysis_numeric:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=analysis, x=col, color="steelblue")
    plt.title(f"Box Plot: {col}")
    save_plot(f"boxplot_{col}.png")

# Target distribution.
plt.figure(figsize=(7, 5))
sns.countplot(data=analysis, x=TARGET, order=analysis[TARGET].value_counts().index, color="steelblue")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Customer count")
save_plot("target_churn_distribution.png")

# ---------------------------- PART 3: BIVARIATE ANALYSIS ----------------------------
# Numeric features against the target: box plots and grouped means.
for col in analysis_numeric:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=analysis, x=TARGET, y=col, color="steelblue")
    plt.title(f"{col} by {TARGET}")
    save_plot(f"target_vs_{col}_boxplot.png")

# Scatter plots for meaningful numeric pairs.
scatter_pairs = []
if {"tenure", "MonthlyCharges"}.issubset(analysis.columns):
    scatter_pairs.append(("tenure", "MonthlyCharges"))
if {"tenure", "TotalCharges"}.issubset(analysis.columns):
    scatter_pairs.append(("tenure", "TotalCharges"))
if {"MonthlyCharges", "TotalCharges"}.issubset(analysis.columns):
    scatter_pairs.append(("MonthlyCharges", "TotalCharges"))

for x_col, y_col in scatter_pairs:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=analysis, x=x_col, y=y_col, hue=TARGET,
        alpha=0.55, s=25
    )
    plt.title(f"{y_col} vs {x_col}, colored by {TARGET}")
    save_plot(f"scatter_{x_col}_vs_{y_col}.png")

# Churn rate by each categorical feature.
categorical_churn_tables = {}
for col in analysis_categorical:
    if col == TARGET:
        continue
    table = pd.crosstab(analysis[col], analysis[TARGET], normalize="index") * 100
    categorical_churn_tables[col] = table
    if "Yes" in table.columns:
        rates = table["Yes"].sort_values(ascending=False)
        plt.figure(figsize=(9, 5))
        sns.barplot(x=rates.values, y=rates.index, color="steelblue")
        plt.title(f"Churn Rate by {col}")
        plt.xlabel("Churn rate (%)")
        plt.ylabel(col)
        save_plot(f"churn_rate_by_{col}.png")
    else:
        counts = analysis[col].value_counts().head(15)
        plt.figure(figsize=(9, 5))
        sns.barplot(x=counts.values, y=counts.index, color="steelblue")
        plt.title(f"Customer Count by {col}")
        plt.xlabel("Customer count")
        plt.ylabel(col)
        save_plot(f"barplot_{col}.png")

for col, table in categorical_churn_tables.items():
    table.to_csv(REPORT_DIR / f"churn_comparison_{col}.csv")

# Numeric grouped analysis by target.
numeric_target_summary = analysis.groupby(TARGET, observed=False)[analysis_numeric].mean().T
numeric_target_summary.to_csv(REPORT_DIR / "numeric_means_by_churn.csv")

# ---------------------------- PART 4: MULTIVARIATE ANALYSIS ----------------------------
# Correlation heatmap.
if len(analysis_numeric) >= 2:
    corr = analysis[analysis_numeric].corr(numeric_only=True)
    corr.to_csv(REPORT_DIR / "numeric_correlation_matrix.csv")
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=False)
    plt.title("Correlation Heatmap — Numerical Features")
    save_plot("correlation_heatmap.png")

# Pair plot on a small, meaningful numeric subset; sample to keep it fast.
preferred_pair_cols = [
    c for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    if c in analysis_numeric
]
if len(preferred_pair_cols) >= 2:
    pair_data = analysis[preferred_pair_cols + [TARGET]].copy()
    if len(pair_data) > 1000:
        pair_data = pair_data.sample(1000, random_state=RANDOM_STATE)
    grid = sns.pairplot(pair_data, vars=preferred_pair_cols, hue=TARGET, corner=True,
                        plot_kws={"alpha": 0.5, "s": 18})
    grid.fig.suptitle("Pair Plot of Selected Features by Churn", y=1.02)
    grid.savefig(PLOT_DIR / "multivariate_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close("all")

# Multiple-feature analysis: churn rate by contract and tenure band.
if {"Contract", "tenure", TARGET}.issubset(analysis.columns):
    tenure_bins = [-1, 12, 24, 48, 72]
    tenure_labels = ["0–12 months", "13–24 months", "25–48 months", "49–72 months"]
    analysis["tenure_band"] = pd.cut(
        analysis["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
    )
    multi = pd.crosstab(
        [analysis["Contract"], analysis["tenure_band"]],
        analysis[TARGET], normalize="index"
    ) * 100
    multi.to_csv(REPORT_DIR / "churn_by_contract_and_tenure_band.csv")
    if "Yes" in multi.columns:
        plot_data = multi["Yes"].reset_index(name="Churn rate (%)")
        plt.figure(figsize=(11, 6))
        sns.barplot(data=plot_data, x="tenure_band", y="Churn rate (%)", hue="Contract", palette="deep")
        plt.title("Churn Rate by Contract Type and Tenure Band")
        save_plot("multiple_feature_contract_tenure_churn.png")

# ---------------------------- PART 5: OUTLIER ANALYSIS ----------------------------
outlier_rows = []
for col in analysis_numeric:
    series = pd.to_numeric(analysis[col], errors="coerce").dropna()
    if series.empty:
        continue
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    n_outliers = int(mask.sum())
    percentage = safe_rate(n_outliers, len(series))

    # Recommendation is deliberately advisory; do not automatically delete valid customers.
    if col in ["SeniorCitizen"]:
        recommendation = "KEEP: binary/indicator variable; IQR flags are not meaningful here."
        explanation = "Treat as a categorical/binary feature rather than removing values."
    elif col in ["tenure"]:
        recommendation = "KEEP after validation."
        explanation = "Extreme tenure values may be valid customers with short or long relationships."
    elif col in ["MonthlyCharges", "TotalCharges"]:
        recommendation = "KEEP after validation; consider transformation only if justified."
        explanation = "Charges may be genuinely high. Check data-entry errors before any transformation."
    else:
        recommendation = "INVESTIGATE; keep valid observations, transform only if needed."
        explanation = "IQR flags potential extremes, not proof of errors."

    outlier_rows.append({
        "column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_bound": lower, "upper_bound": upper,
        "outlier_count": n_outliers, "outlier_percentage": percentage,
        "minimum": series.min(), "maximum": series.max(),
        "recommendation": recommendation, "explanation": explanation
    })

outlier_report = pd.DataFrame(outlier_rows)
outlier_report.to_csv(REPORT_DIR / "outlier_analysis.csv", index=False)

# ---------------------------- PART 6: DATA-DRIVEN INSIGHTS ----------------------------
insights = []
def add_insight(text):
    insights.append(f"Insight {len(insights) + 1}: {text}")

n_customers = len(analysis)
churn_yes = int(analysis[TARGET].astype(str).eq("Yes").sum())
churn_no = int(analysis[TARGET].astype(str).eq("No").sum())
churn_rate = safe_rate(churn_yes, n_customers)

add_insight(f"The analyzed dataset contains {n_customers:,} customer records and {analysis.shape[1]:,} columns after exact duplicate rows are removed.")
add_insight(f"{churn_yes:,} customers churned and {churn_no:,} stayed; the overall churn rate is {fmt(churn_rate)}%.")

# Most common categorical categories.
for col in [c for c in ["Contract", "InternetService", "PaymentMethod", "gender"] if c in analysis.columns]:
    counts = analysis[col].value_counts(dropna=False)
    if len(counts):
        add_insight(f"The most common {col} category is '{counts.index[0]}' ({counts.iloc[0]:,} customers; {safe_rate(counts.iloc[0], n_customers):.1f}% of records).")

# Highest churn rate categories for selected variables.
for col in [c for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "PaperlessBilling", "Partner", "Dependents"] if c in analysis.columns]:
    rates = top_category_rate(analysis, col)
    if len(rates):
        add_insight(f"'{rates.index[0]}' has the highest observed churn rate among {col} categories ({rates.iloc[0]:.1f}%). Compare group sizes before interpreting this difference.")

# Numeric churn comparisons.
for col in [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in analysis.columns]:
    means = analysis.groupby(TARGET, observed=False)[col].mean()
    if "Yes" in means.index and "No" in means.index:
        difference = means["Yes"] - means["No"]
        direction = "higher" if difference > 0 else "lower" if difference < 0 else "about the same"
        add_insight(f"Mean {col} is {direction} for churned customers than retained customers (churned: {fmt(means['Yes'])}; retained: {fmt(means['No'])}). This is an association, not proof of causation.")

# Correlation insight: strongest absolute non-duplicate pair.
if len(analysis_numeric) >= 2:
    corr = analysis[analysis_numeric].corr(numeric_only=True)
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    if not pairs.empty:
        pair = pairs.abs().idxmax()
        value = corr.loc[pair[0], pair[1]]
        direction = "positive" if value > 0 else "negative"
        add_insight(f"The strongest absolute linear correlation among numerical variables is between {pair[0]} and {pair[1]} (r = {value:.2f}, {direction}). Correlation does not establish causation.")

# Outlier insights.
if not outlier_report.empty:
    most_outlier = outlier_report.sort_values("outlier_percentage", ascending=False).iloc[0]
    add_insight(f"By the 1.5×IQR rule, {most_outlier['column']} has the highest flagged outlier percentage ({most_outlier['outlier_percentage']:.1f}%). These values should be validated, not automatically deleted.")

# Missingness insight after numeric conversion.
missing_total = int(df.isna().sum().sum())
if missing_total:
    cols_missing = df.isna().sum().sort_values(ascending=False)
    cols_missing = cols_missing[cols_missing > 0]
    add_insight(f"After converting numeric-looking columns, {missing_total:,} missing values were detected. The most affected column is {cols_missing.index[0]} ({cols_missing.iloc[0]:,} missing values).")
else:
    add_insight("No missing values were detected after whitespace normalization and numeric conversion.")

add_insight(f"The source data has {duplicate_count:,} exact duplicate rows. Exact duplicates were removed for analysis; duplicate customer IDs were reported separately for review.")
add_insight("Potentially useful predictors include tenure, contract, monthly charges, total charges, internet service, payment method, and support/security service choices; validate their usefulness with proper model evaluation later.")
add_insight("customerID is an identifier and should not be used as a predictive feature.")
add_insight("The dataset has a binary target, Churn, so it can support a future binary classification task after encoding categorical variables.")
add_insight("The class distribution should be checked before training because unequal churn/non-churn counts can affect evaluation; accuracy alone may be insufficient.")

# Guarantee at least 15 insight entries, even if an unexpected schema is supplied.
fallbacks = [
    "Histograms and KDE curves should be checked for skewness and multiple peaks before deciding whether transformations are appropriate.",
    "Box plots flag potential extremes; verify records against domain knowledge before removing them.",
    "Categorical churn-rate charts compare proportions and should be interpreted alongside the number of customers in each category."
]
for item in fallbacks:
    if len(insights) >= 18:
        break
    add_insight(item)

write_text("eda_insights.txt", "\n\n".join(insights))
pd.DataFrame({"Insight": insights}).to_csv(REPORT_DIR / "eda_insights.csv", index=False)

# ---------------------------- PART 7: ML PREPARATION ----------------------------
# No model is trained. Save a cleaned table and a model-ready X/y dataset.
# The target remains separate. customerID and temporary tenure_band are excluded.
cleaned = analysis.copy()
cleaned.to_csv(OUTPUT_DIR / "cleaned_telco_customer.csv", index=False)

exclude = [c for c in [TARGET, ID_COLUMN, "tenure_band"] if c in analysis.columns]
X = analysis.drop(columns=exclude, errors="ignore").copy()
y = analysis[TARGET].copy()

# For the ML-ready dataset, impute remaining missing values and one-hot encode categoricals.
# Keep the original target as a separate column for easy use later.
for col in X.select_dtypes(include=np.number).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=["object", "string", "category", "bool"]).columns:
    X[col] = X[col].fillna("Unknown").astype(str)

X_encoded = pd.get_dummies(X, drop_first=False, dtype=int)
ml_ready = X_encoded.copy()
ml_ready[TARGET] = y.astype(str).values

X_encoded.to_csv(OUTPUT_DIR / "X_ml_features_encoded.csv", index=False)
y.rename(TARGET).to_csv(OUTPUT_DIR / "y_ml_target.csv", index=False)
ml_ready.to_csv(OUTPUT_DIR / "ml_ready_telco_dataset.csv", index=False)

feature_notes = [
    "ML PREPARATION NOTES",
    "=" * 40,
    "Target y: Churn",
    "Potential features X: all columns except customerID, Churn, and temporary analysis columns.",
    "Categorical features were one-hot encoded with pandas.get_dummies.",
    "No machine-learning model was trained.",
    "customerID was excluded because it is an identifier.",
    "Inspect class balance and use suitable metrics (precision, recall, F1, ROC-AUC) when training a model.",
    "",
    "Encoded feature columns:",
    ", ".join(X_encoded.columns.tolist())
]
write_text("ml_preparation_notes.txt", "\n".join(feature_notes))

# ---------------------------- FINAL SUMMARY ----------------------------
summary = f"""
TELCO CUSTOMER CHURN EDA — FINAL SUMMARY
========================================
Input file: {DATA_FILE}
Original rows: {initial_rows:,}
Original columns: {initial_cols:,}
Exact duplicate rows: {duplicate_count:,}
Duplicate customer IDs: {duplicate_id_count:,}
Rows used after exact duplicate removal: {len(analysis):,}
Churned customers: {churn_yes:,}
Retained customers: {churn_no:,}
Overall churn rate: {fmt(churn_rate)}%
Numeric columns (excluding ID): {len(analysis_numeric)}
Categorical/text columns: {len(analysis_categorical)}
Insights generated: {len(insights)}
Plots saved: {len(list(PLOT_DIR.glob('*.png')))}
Output folder: {OUTPUT_DIR.resolve()}

MAIN OUTPUTS
------------
reports/dataset_profile.txt
reports/eda_insights.txt
reports/eda_insights.csv
reports/outlier_analysis.csv
reports/numeric_correlation_matrix.csv (if numeric columns are available)
plots/ (histograms/KDE, count plots, box plots, scatter plots, churn comparisons,
        correlation heatmap, pair plot, and multiple-feature analysis)
cleaned_telco_customer.csv
X_ml_features_encoded.csv
y_ml_target.csv
ml_ready_telco_dataset.csv

Important: Outlier flags are recommendations for investigation, not automatic deletion rules.
This analysis is exploratory; relationships observed do not prove causation.
"""
write_text("final_eda_summary.txt", summary)

print(summary)
print("Finished. Open the 'teleco_eda_output' folder to review reports and plots.")
