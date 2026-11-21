import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# CUSTOMER CLUSTER PROFILE ANALYZER
# ==========================================

print("=" * 60)
print("       CUSTOMER CLUSTER PROFILE ANALYZER")
print("=" * 60)

os.makedirs("day58_outputs", exist_ok=True)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n[1] Loading dataset...")

df = pd.read_csv("customer_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ==========================================
# 2. DATA CLEANING
# ==========================================

print("\n[2] Cleaning dataset...")

df.drop_duplicates(inplace=True)

required_columns = [
    "Customer ID",
    "Age",
    "Annual Income",
    "Spending Score"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

df.dropna(subset=required_columns, inplace=True)

print("Data cleaning completed.")

# ==========================================
# 3. GENERATE CLUSTER LABELS USING K-MEANS
# ==========================================

print("\n[3] Generating customer clusters...")

features = ["Annual Income", "Spending Score"]

X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Clustering completed successfully!")

print("\nCluster labels:")
print(df["Cluster"].value_counts().sort_index())

# Save clustered dataset
df.to_csv(
    "day58_outputs/customer_data_with_clusters.csv",
    index=False
)

# ==========================================
# 4. CLUSTER SIZE ANALYSIS
# ==========================================

print("\n[4] Analyzing cluster sizes...")

cluster_counts = df["Cluster"].value_counts().sort_index()

cluster_percentages = cluster_counts / len(df) * 100

cluster_size_table = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Customer Count": cluster_counts.values,
    "Percentage": cluster_percentages.values
})

print("\nCluster Size Summary:")
print(cluster_size_table)

cluster_size_table.to_csv(
    "day58_outputs/cluster_size_summary.csv",
    index=False
)

# ==========================================
# 5. AVERAGE CLUSTER PROFILES
# ==========================================

print("\n[5] Calculating average cluster profiles...")

features = ["Age", "Annual Income", "Spending Score"]

cluster_means = df.groupby("Cluster")[features].mean()

print("\nAverage Cluster Profiles:")
print(cluster_means)

cluster_means.to_csv(
    "day58_outputs/cluster_average_profiles.csv"
)

# ==========================================
# 6. MEDIAN CLUSTER PROFILES
# ==========================================

print("\n[6] Calculating median cluster profiles...")

cluster_medians = df.groupby("Cluster")[features].median()

print("\nMedian Cluster Profiles:")
print(cluster_medians)

cluster_medians.to_csv(
    "day58_outputs/cluster_median_profiles.csv"
)

# ==========================================
# 7. COMPLETE CLUSTER PROFILE
# ==========================================

print("\n[7] Creating complete cluster profile...")

complete_profile = cluster_means.copy()

complete_profile["Median Age"] = cluster_medians["Age"]
complete_profile["Median Income"] = cluster_medians["Annual Income"]
complete_profile["Median Spending Score"] = cluster_medians["Spending Score"]

complete_profile["Customer Count"] = cluster_counts
complete_profile["Percentage"] = cluster_percentages

complete_profile.to_csv(
    "day58_outputs/complete_cluster_profile.csv"
)

print(complete_profile)

# ==========================================
# 8. CLUSTER SIZE VISUALIZATION
# ==========================================

print("\n[8] Creating cluster size chart...")

plt.figure(figsize=(10, 6))

sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values
)

plt.title("Customer Count per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "day58_outputs/cluster_size_chart.png",
    dpi=300
)

plt.show()

# ==========================================
# 9. CLUSTER PROFILE HEATMAP
# ==========================================

print("\n[9] Creating cluster profile heatmap...")

plt.figure(figsize=(10, 6))

sns.heatmap(
    cluster_means,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu"
)

plt.title("Average Customer Cluster Profiles")
plt.xlabel("Features")
plt.ylabel("Cluster")

plt.tight_layout()

plt.savefig(
    "day58_outputs/cluster_profile_heatmap.png",
    dpi=300
)

plt.show()

# ==========================================
# 10. INCOME AND SPENDING COMPARISON
# ==========================================

print("\n[10] Creating income and spending comparison...")

comparison = cluster_means[
    ["Annual Income", "Spending Score"]
]

comparison.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Income and Spending Comparison by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Value")

plt.xticks(rotation=0)
plt.legend(title="Features")

plt.tight_layout()

plt.savefig(
    "day58_outputs/income_spending_comparison.png",
    dpi=300
)

plt.show()

# ==========================================
# 11. RADAR CHART
# ==========================================

print("\n[11] Creating radar chart...")

radar_features = [
    "Age",
    "Annual Income",
    "Spending Score"
]

normalized_profiles = cluster_means.copy()

# Normalize each feature
for feature in radar_features:

    minimum = cluster_means[feature].min()
    maximum = cluster_means[feature].max()

    if maximum != minimum:

        normalized_profiles[feature] = (
            (cluster_means[feature] - minimum) /
            (maximum - minimum)
        )

    else:

        normalized_profiles[feature] = 0

labels = radar_features

num_features = len(labels)

angles = np.linspace(
    0,
    2 * np.pi,
    num_features,
    endpoint=False
).tolist()

angles += angles[:1]

plt.figure(figsize=(10, 10))

ax = plt.subplot(111, polar=True)

for cluster in normalized_profiles.index:

    values = normalized_profiles.loc[
        cluster,
        radar_features
    ].tolist()

    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=2,
        label=f"Cluster {cluster}"
    )

    ax.fill(
        angles,
        values,
        alpha=0.1
    )

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title("Customer Cluster Radar Chart")

plt.legend(
    loc="upper right",
    bbox_to_anchor=(1.2, 1.1)
)

plt.tight_layout()

plt.savefig(
    "day58_outputs/cluster_radar_chart.png",
    dpi=300
)

plt.show()

# ==========================================
# 12. LARGEST AND SMALLEST CLUSTERS
# ==========================================

print("\n[12] Identifying largest and smallest clusters...")

largest_cluster = cluster_counts.idxmax()
smallest_cluster = cluster_counts.idxmin()

print(f"Largest Cluster: Cluster {largest_cluster}")
print(f"Customers: {cluster_counts[largest_cluster]}")

print(f"\nSmallest Cluster: Cluster {smallest_cluster}")
print(f"Customers: {cluster_counts[smallest_cluster]}")

# ==========================================
# 13. AUTOMATIC CUSTOMER SEGMENT NAMING
# ==========================================

print("\n[13] Generating customer segment names...")

overall_income_median = df["Annual Income"].median()
overall_spending_median = df["Spending Score"].median()

cluster_names = {}

for cluster in cluster_means.index:

    income = cluster_means.loc[
        cluster, "Annual Income"
    ]

    spending = cluster_means.loc[
        cluster, "Spending Score"
    ]

    if income >= overall_income_median and spending >= overall_spending_median:

        name = "Premium High Spenders"

    elif income >= overall_income_median and spending < overall_spending_median:

        name = "High Income Low Spenders"

    elif income < overall_income_median and spending >= overall_spending_median:

        name = "Budget High Spenders"

    else:

        name = "Budget Low Spenders"

    cluster_names[cluster] = name

print("\nCluster Names:")

for cluster, name in cluster_names.items():

    print(f"Cluster {cluster}: {name}")

# Add cluster names to dataset
df["Cluster Name"] = df["Cluster"].map(cluster_names)

df.to_csv(
    "day58_outputs/customer_data_with_cluster_names.csv",
    index=False
)

# ==========================================
# 14. BUSINESS INSIGHTS
# ==========================================

print("\n[14] Generating business insights...")

insights = []

# Largest cluster
insights.append(
    f"The largest customer segment is Cluster {largest_cluster} "
    f"with {cluster_counts[largest_cluster]} customers "
    f"({cluster_percentages[largest_cluster]:.2f}%)."
)

# Smallest cluster
insights.append(
    f"The smallest customer segment is Cluster {smallest_cluster} "
    f"with {cluster_counts[smallest_cluster]} customers."
)

# Highest income
highest_income_cluster = cluster_means["Annual Income"].idxmax()

insights.append(
    f"Cluster {highest_income_cluster} has the highest average annual income "
    f"({cluster_means.loc[highest_income_cluster, 'Annual Income']:.2f})."
)

# Highest spending
highest_spending_cluster = cluster_means["Spending Score"].idxmax()

insights.append(
    f"Cluster {highest_spending_cluster} has the highest average spending score "
    f"({cluster_means.loc[highest_spending_cluster, 'Spending Score']:.2f})."
)

# Lowest spending
lowest_spending_cluster = cluster_means["Spending Score"].idxmin()

insights.append(
    f"Cluster {lowest_spending_cluster} has the lowest average spending score "
    f"({cluster_means.loc[lowest_spending_cluster, 'Spending Score']:.2f})."
)

# Youngest cluster
youngest_cluster = cluster_means["Age"].idxmin()

insights.append(
    f"Cluster {youngest_cluster} has the lowest average customer age "
    f"({cluster_means.loc[youngest_cluster, 'Age']:.2f})."
)

# Save insights
with open(
    "day58_outputs/business_insights_report.txt",
    "w"
) as file:

    file.write("CUSTOMER CLUSTER BUSINESS INSIGHTS\n")
    file.write("=" * 50 + "\n\n")

    for i, insight in enumerate(insights, 1):

        file.write(f"{i}. {insight}\n\n")

print("\nBUSINESS INSIGHTS:")

for i, insight in enumerate(insights, 1):

    print(f"{i}. {insight}")

# ==========================================
# 15. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("CUSTOMER CLUSTER PROFILE ANALYSIS COMPLETED")
print("=" * 60)

print(f"\nTotal Customers: {len(df)}")
print(f"Total Clusters: {df['Cluster'].nunique()}")

print("\nGenerated files are saved inside the day58_outputs folder.")

print("\nFiles generated:")

print("- customer_data_with_clusters.csv")
print("- cluster_size_summary.csv")
print("- cluster_average_profiles.csv")
print("- cluster_median_profiles.csv")
print("- complete_cluster_profile.csv")
print("- cluster_size_chart.png")
print("- cluster_profile_heatmap.png")
print("- income_spending_comparison.png")
print("- cluster_radar_chart.png")
print("- business_insights_report.txt")
print("- customer_data_with_cluster_names.csv")

print("\nAnalysis completed successfully!")