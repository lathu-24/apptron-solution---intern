import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# 1. Create Patient Dataset
# --------------------------------

np.random.seed(42)

n = 1000

age = np.random.randint(18, 81, n)

gender = np.random.choice(
    ["Male", "Female"],
    n
)

bmi = np.round(
    np.random.normal(27, 5, n),
    1
)

blood_pressure = np.round(
    np.random.normal(125, 15, n),
    0
)

glucose = np.round(
    np.random.normal(105, 25, n),
    0
)

cholesterol = np.round(
    np.random.normal(200, 35, n),
    0
)

# Keep values within realistic ranges
bmi = np.clip(bmi, 15, 45)
blood_pressure = np.clip(blood_pressure, 80, 190)
glucose = np.clip(glucose, 60, 250)
cholesterol = np.clip(cholesterol, 100, 350)


# --------------------------------
# 2. Create Disease Status
# --------------------------------

disease_probability = (
    0.10
    + (age > 50) * 0.15
    + (bmi > 30) * 0.15
    + (glucose > 125) * 0.20
    + (blood_pressure > 140) * 0.15
    + (cholesterol > 240) * 0.10
)

disease_probability = np.clip(
    disease_probability,
    0,
    0.90
)

disease_status = np.where(
    np.random.random(n) < disease_probability,
    "Disease",
    "No Disease"
)


# --------------------------------
# 3. Create DataFrame
# --------------------------------

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Blood Pressure": blood_pressure,
    "Glucose": glucose,
    "Cholesterol": cholesterol,
    "Disease Status": disease_status
})

print("HEALTHCARE DATASET")
print("=" * 70)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nStatistical Summary:")
print(df.describe())

print("\nDisease Distribution:")
print(df["Disease Status"].value_counts())


# --------------------------------
# 4. Age Distribution
# Histogram
# --------------------------------

plt.figure(figsize=(9, 6))

plt.hist(
    df["Age"],
    bins=12,
    edgecolor="black"
)

plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# --------------------------------
# 5. Disease Count
# Count Plot
# --------------------------------

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Disease Status"
)

plt.title("Disease Status Distribution")
plt.xlabel("Disease Status")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# --------------------------------
# 6. BMI by Disease Status
# Box Plot
# --------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Disease Status",
    y="BMI"
)

plt.title("BMI by Disease Status")
plt.xlabel("Disease Status")
plt.ylabel("BMI")

plt.tight_layout()
plt.show()


# --------------------------------
# 7. Age vs Glucose
# Scatter Plot
# --------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Age",
    y="Glucose",
    hue="Disease Status",
    alpha=0.6
)

plt.title("Age vs Glucose")
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend(title="Disease Status")

plt.tight_layout()
plt.show()


# --------------------------------
# 8. Gender vs Disease Status
# Count Plot
# --------------------------------

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="Gender",
    hue="Disease Status"
)

plt.title("Gender vs Disease Status")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.legend(title="Disease Status")

plt.tight_layout()
plt.show()


# --------------------------------
# 9. Age Group Analysis
# --------------------------------

df["Age Group"] = pd.cut(
    df["Age"],
    bins=[17, 30, 40, 50, 60, 80],
    labels=[
        "18-30",
        "31-40",
        "41-50",
        "51-60",
        "61-80"
    ]
)

age_group_counts = (
    df["Age Group"]
    .value_counts()
    .sort_index()
)

largest_age_group = age_group_counts.idxmax()

print("\nAGE GROUP ANALYSIS")
print("=" * 70)

print(age_group_counts)

print(
    f"\nAge group with most patients: "
    f"{largest_age_group}"
)


# --------------------------------
# 10. BMI Group Analysis
# --------------------------------

average_bmi_by_disease = (
    df.groupby("Disease Status")["BMI"]
    .mean()
)

higher_bmi_group = average_bmi_by_disease.idxmax()

print("\nBMI ANALYSIS")
print("=" * 70)

print(average_bmi_by_disease.round(2))

print(
    f"\nGroup with higher average BMI: "
    f"{higher_bmi_group}"
)


# --------------------------------
# 11. Outlier Detection
# --------------------------------

numerical_columns = [
    "Age",
    "BMI",
    "Blood Pressure",
    "Glucose",
    "Cholesterol"
]

print("\nOUTLIER ANALYSIS")
print("=" * 70)

outlier_summary = {}

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    outlier_summary[column] = len(outliers)

    print(
        f"{column}: {len(outliers)} outliers"
    )


# --------------------------------
# 12. Disease Relationship Analysis
# --------------------------------

disease_means = (
    df.groupby("Disease Status")[
        [
            "Age",
            "BMI",
            "Blood Pressure",
            "Glucose",
            "Cholesterol"
        ]
    ]
    .mean()
)

print("\nVARIABLES BY DISEASE STATUS")
print("=" * 70)

print(disease_means.round(2))


# --------------------------------
# 13. Correlation Analysis
# --------------------------------

correlation = df[
    [
        "Age",
        "BMI",
        "Blood Pressure",
        "Glucose",
        "Cholesterol"
    ]
].corr()

print("\nCORRELATION MATRIX")
print("=" * 70)

print(correlation.round(2))


# --------------------------------
# 14. Save Dataset
# --------------------------------

df.to_csv(
    "healthcare_patients.csv",
    index=False
)

age_group_counts.to_csv(
    "age_group_distribution.csv"
)

disease_means.to_csv(
    "disease_variable_analysis.csv"
)

correlation.to_csv(
    "healthcare_correlations.csv"
)

print("\nFILES GENERATED")
print("=" * 70)

print("1. healthcare_patients.csv")
print("2. age_group_distribution.csv")
print("3. disease_variable_analysis.csv")
print("4. healthcare_correlations.csv")

print("\nHealthcare Data Explorer completed!")