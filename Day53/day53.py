# ============================================================
# DAY 53 - CUSTOMER SEGMENTATION ANALYSIS SYSTEM
# Customer Profiles and Business Recommendations
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 1. SETUP
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "customer_data.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Use the K selected in your previous Elbow Method task.
SELECTED_K = 3

RANDOM_STATE = 42
N_INIT = 10


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n========== CUSTOMER DATASET ==========")

print(df.head(10))

print("\nDataset Shape:", df.shape)


# Check required columns

required_columns = [
    "Customer ID",
    "Age",
    "Annual Income",
    "Spending Score"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")


# Remove duplicates and handle missing numerical values

df = df.drop_duplicates().copy()

features = [
    "Age",
    "Annual Income",
    "Spending Score"
]

for column in features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].fillna(
        df[column].median()
    )


# ------------------------------------------------------------
# 3. SELECT CLUSTERING FEATURES
# ------------------------------------------------------------

X = df[
    ["Annual Income", "Spending Score"]
]

print("\nSelected Clustering Features:")

print(X.head())


# ------------------------------------------------------------
# 4. APPLY STANDARD SCALER
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Features:")

print(
    pd.DataFrame(
        X_scaled,
        columns=["Annual Income", "Spending Score"]
    ).head()
)


# ------------------------------------------------------------
# 5. TRAIN K-MEANS MODEL
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=SELECTED_K,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n========== CLUSTER ASSIGNMENTS ==========")

print(df.head(10))


# ------------------------------------------------------------
# 6. CUSTOMER COUNT AND PERCENTAGE
# ------------------------------------------------------------

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_percentages = (
    cluster_counts / len(df) * 100
).round(2)

print("\n========== CUSTOMER DISTRIBUTION ==========")

for cluster in cluster_counts.index:

    print(
        f"Cluster {cluster}: "
        f"{cluster_counts[cluster]} customers "
        f"({cluster_percentages[cluster]}%)"
    )


# ------------------------------------------------------------
# 7. CREATE CLUSTER PROFILE TABLE
# ------------------------------------------------------------

cluster_profile = df.groupby("Cluster").agg(

    Customer_Count=("Customer ID", "count"),

    Average_Age=("Age", "mean"),

    Average_Annual_Income=("Annual Income", "mean"),

    Average_Spending_Score=("Spending Score", "mean")

).round(2)

cluster_profile["Percentage"] = cluster_percentages

print("\n========== CLUSTER PROFILE TABLE ==========")

print(cluster_profile)


# ------------------------------------------------------------
# 8. GENERATE BUSINESS-FRIENDLY CLUSTER NAMES
# ------------------------------------------------------------

overall_income_median = df["Annual Income"].median()

overall_spending_median = df["Spending Score"].median()


def identify_customer_segment(row):

    income = row["Average_Annual_Income"]

    spending = row["Average_Spending_Score"]

    if (
        income >= overall_income_median
        and spending >= overall_spending_median
    ):

        return "Premium Active Spenders"

    elif (
        income >= overall_income_median
        and spending < overall_spending_median
    ):

        return "Affluent Low-Engagement Customers"

    elif (
        income < overall_income_median
        and spending >= overall_spending_median
    ):

        return "Budget-Conscious Active Shoppers"

    else:

        return "Low-Spend Value Customers"


cluster_profile["Segment_Name"] = cluster_profile.apply(
    identify_customer_segment,
    axis=1
)


# ------------------------------------------------------------
# 9. BUSINESS RECOMMENDATIONS
# ------------------------------------------------------------

def recommend_actions(segment):

    if segment == "Premium Active Spenders":

        return (
            "Offer VIP rewards and premium product bundles.",
            "Provide early access to new products and exclusive offers."
        )

    elif segment == "Affluent Low-Engagement Customers":

        return (
            "Send personalized promotions based on customer interests.",
            "Use targeted campaigns to increase customer engagement."
        )

    elif segment == "Budget-Conscious Active Shoppers":

        return (
            "Offer discounts and affordable product bundles.",
            "Introduce loyalty points and limited-time promotions."
        )

    else:

        return (
            "Promote affordable and essential products.",
            "Use low-cost marketing campaigns to encourage purchases."
        )


cluster_profile["Business_Action_1"] = (
    cluster_profile["Segment_Name"].apply(
        lambda name: recommend_actions(name)[0]
    )
)

cluster_profile["Business_Action_2"] = (
    cluster_profile["Segment_Name"].apply(
        lambda name: recommend_actions(name)[1]
    )
)


print("\n========== BUSINESS SEGMENT PROFILES ==========")

print(
    cluster_profile[
        [
            "Customer_Count",
            "Percentage",
            "Average_Age",
            "Average_Annual_Income",
            "Average_Spending_Score",
            "Segment_Name"
        ]
    ]
)


# ------------------------------------------------------------
# 10. SAVE DATASET AND PROFILE TABLE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "customer_data_with_clusters.csv",
    index=False
)

cluster_profile.to_csv(
    OUTPUT_DIR / "cluster_profile_table.csv"
)


# ------------------------------------------------------------
# 11. CHART 1 - INCOME VS SPENDING SCATTER PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    segment_name = cluster_profile.loc[
        cluster, "Segment_Name"
    ]

    plt.scatter(
        cluster_data["Annual Income"],
        cluster_data["Spending Score"],
        label=f"Cluster {cluster}: {segment_name}",
        alpha=0.7
    )


# Convert scaled centroids back to original feature values

centroids = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    edgecolors="black",
    label="Centroids"
)

plt.title("Customer Segmentation: Income vs Spending")

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.legend(fontsize=8)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "income_spending_clusters.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 12. CHART 2 - AVERAGE ANNUAL INCOME
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    cluster_profile.index.astype(str),
    cluster_profile["Average_Annual_Income"]
)

plt.title("Average Annual Income by Cluster")

plt.xlabel("Cluster")

plt.ylabel("Average Annual Income")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "average_income_by_cluster.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 13. CHART 3 - AVERAGE SPENDING SCORE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    cluster_profile.index.astype(str),
    cluster_profile["Average_Spending_Score"]
)

plt.title("Average Spending Score by Cluster")

plt.xlabel("Cluster")

plt.ylabel("Average Spending Score")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "average_spending_by_cluster.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 14. DATA-SUPPORTED BUSINESS FINDINGS
# ------------------------------------------------------------

largest_cluster = cluster_profile[
    "Customer_Count"
].idxmax()

highest_income_cluster = cluster_profile[
    "Average_Annual_Income"
].idxmax()

highest_spending_cluster = cluster_profile[
    "Average_Spending_Score"
].idxmax()


findings = [

    f"Cluster {largest_cluster} is the largest customer group, "
    f"containing {cluster_profile.loc[largest_cluster, 'Customer_Count']} "
    f"customers ({cluster_profile.loc[largest_cluster, 'Percentage']}%).",

    f"Cluster {highest_income_cluster} has the highest average annual "
    f"income of {cluster_profile.loc[highest_income_cluster, 'Average_Annual_Income']}.",

    f"Cluster {highest_spending_cluster} has the highest average spending "
    f"score of {cluster_profile.loc[highest_spending_cluster, 'Average_Spending_Score']}."

]


print("\n========== BUSINESS FINDINGS ==========")

for i, finding in enumerate(findings, 1):

    print(f"{i}. {finding}")


# ------------------------------------------------------------
# 15. BONUS - AGE AND SPENDING SCORE CLUSTERING
# ------------------------------------------------------------

bonus_features = [
    "Age",
    "Spending Score"
]

X_bonus = df[bonus_features]

bonus_scaler = StandardScaler()

X_bonus_scaled = bonus_scaler.fit_transform(
    X_bonus
)


bonus_model = KMeans(
    n_clusters=SELECTED_K,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)

bonus_labels = bonus_model.fit_predict(
    X_bonus_scaled
)

bonus_df = df.copy()

bonus_df["Age_Spending_Cluster"] = bonus_labels


# Create bonus profile

bonus_profile = bonus_df.groupby(
    "Age_Spending_Cluster"
).agg(

    Customer_Count=("Customer ID", "count"),

    Average_Age=("Age", "mean"),

    Average_Annual_Income=("Annual Income", "mean"),

    Average_Spending_Score=("Spending Score", "mean")

).round(2)

bonus_profile["Percentage"] = (
    bonus_profile["Customer_Count"] / len(bonus_df) * 100
).round(2)


print("\n========== BONUS CLUSTER PROFILES ==========")

print(bonus_profile)


# Save bonus results

bonus_df.to_csv(
    OUTPUT_DIR / "age_spending_customer_clusters.csv",
    index=False
)

bonus_profile.to_csv(
    OUTPUT_DIR / "age_spending_cluster_profiles.csv"
)


# ------------------------------------------------------------
# 16. BONUS - AGE VS SPENDING SCATTER PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for cluster in sorted(
    bonus_df["Age_Spending_Cluster"].unique()
):

    subset = bonus_df[
        bonus_df["Age_Spending_Cluster"] == cluster
    ]

    plt.scatter(
        subset["Age"],
        subset["Spending Score"],
        label=f"Cluster {cluster}",
        alpha=0.7
    )


plt.title("Customer Segmentation: Age vs Spending Score")

plt.xlabel("Age")

plt.ylabel("Spending Score")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_spending_clusters.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 17. GENERATE BUSINESS INSIGHTS REPORT
# ------------------------------------------------------------

report = f"""
CUSTOMER SEGMENTATION ANALYSIS REPORT

Dataset Records: {len(df)}

Selected K: {SELECTED_K}

Main Features:
Annual Income and Spending Score

Clustering Algorithm:
K-Means

==================================================
CLUSTER PROFILE TABLE
==================================================

{cluster_profile.to_string()}


==================================================
BUSINESS FINDINGS
==================================================

"""

for i, finding in enumerate(findings, 1):

    report += f"{i}. {finding}\n"


report += """

==================================================
BUSINESS RECOMMENDATIONS
==================================================
"""

for cluster, row in cluster_profile.iterrows():

    report += f"""

Cluster {cluster}: {row['Segment_Name']}

Action 1: {row['Business_Action_1']}

Action 2: {row['Business_Action_2']}

"""


report += """

==================================================
BONUS: AGE AND SPENDING SCORE
==================================================

"""

report += bonus_profile.to_string()


report += """

==================================================
LIMITATION
==================================================

K-Means depends on the selected features and number of clusters.
It can also be sensitive to outliers and may not identify complex
customer behaviour.

The results show patterns in this dataset, but they do not
necessarily explain why customers behave in a particular way.


==================================================
CONCLUSION
==================================================

Customer segmentation helps businesses understand different
customer groups.

The identified segments can support targeted marketing,
personalized offers and customer engagement strategies.
"""


with open(
    OUTPUT_DIR / "business_insights_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ------------------------------------------------------------
# 18. FINAL SUMMARY
# ------------------------------------------------------------

print("\n========== FINAL SUMMARY ==========")

print("Total Customers:", len(df))

print("Selected K:", SELECTED_K)

print("Number of Clusters:", df["Cluster"].nunique())

print("\nGenerated Deliverables:")

print("1. Customer dataset with cluster labels")

print("2. Cluster profile table")

print("3. Income vs Spending scatter plot")

print("4. Average income bar chart")

print("5. Average spending bar chart")

print("6. Business insights report")

print("7. Bonus Age vs Spending analysis")

print("\nDAY 53 COMPLETED SUCCESSFULLY!")