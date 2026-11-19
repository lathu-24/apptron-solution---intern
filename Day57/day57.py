import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CREATE OUTPUT FOLDER
# ==========================================

os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 2. LOAD CUSTOMER DATASET
# ==========================================

df = pd.read_csv("customer_data.csv")

print("\n========== DATASET INFORMATION ==========")

print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ==========================================
# 3. DATA CLEANING
# ==========================================

features = [
    "Age",
    "Annual Income",
    "Spending Score"
]

for column in features:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.drop_duplicates()
df = df.dropna(subset=features)

print("\nCleaned Dataset Shape:", df.shape)

# ==========================================
# 4. GENERATE K-MEANS CLUSTER LABELS
# ==========================================

print("\n========== K-MEANS CLUSTERING ==========")

# Select clustering features
clustering_features = [
    "Annual Income",
    "Spending Score"
]

# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[clustering_features]
)

# Train K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("K-Means clustering completed!")

# Save dataset with cluster labels
df.to_csv(
    "outputs/customer_data_with_clusters.csv",
    index=False
)

# ==========================================
# 5. DISPLAY CLUSTER COUNTS
# ==========================================

print("\n========== CUSTOMER COUNT PER CLUSTER ==========")

cluster_counts = df["Cluster"].value_counts().sort_index()

print(cluster_counts)

# ==========================================
# 6. SEABORN PAIR PLOT
# ==========================================

print("\nCreating Pair Plot...")

pair_plot = sns.pairplot(
    df,
    vars=features,
    hue="Cluster",
    diag_kind="hist",
    plot_kws={"alpha": 0.7}
)

pair_plot.fig.suptitle(
    "Multi-Feature Customer Cluster Pair Plot",
    y=1.02
)

pair_plot.savefig(
    "outputs/customer_pairplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 7. CUSTOMER BUBBLE CHART
# ==========================================

print("\nCreating Bubble Chart...")

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="Annual Income",
    y="Spending Score",
    hue="Cluster",
    size=np.full(len(df), 80),
    sizes=(80, 80),
    alpha=0.7,
    legend="full"
)

plt.title("Customer Segmentation Bubble Chart")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")
plt.grid(alpha=0.3)

plt.savefig(
    "outputs/customer_bubble_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 8. HISTOGRAM FOR EACH FEATURE
# ==========================================

print("\nCreating Feature Histograms...")

for feature in features:

    plt.figure(figsize=(9, 6))

    sns.histplot(
        data=df,
        x=feature,
        hue="Cluster",
        multiple="layer",
        kde=True,
        alpha=0.4
    )

    plt.title(f"{feature} Distribution by Cluster")
    plt.xlabel(feature)
    plt.ylabel("Number of Customers")

    filename = feature.lower().replace(" ", "_")

    plt.savefig(
        f"outputs/{filename}_histogram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# ==========================================
# 9. COMPARE FEATURE DISTRIBUTIONS
# ==========================================

print("\n========== CLUSTER FEATURE COMPARISON ==========")

profile = df.groupby("Cluster")[features].mean().round(2)

print(profile)

profile.to_csv(
    "outputs/cluster_feature_comparison.csv"
)

# ==========================================
# 10. IDENTIFY FEATURES THAT SEPARATE CLUSTERS
# ==========================================

print("\n========== FEATURE SEPARATION ANALYSIS ==========")

overall_std = df[features].std().replace(0, np.nan)

separation = (
    profile.max() - profile.min()
) / overall_std

separation = separation.sort_values(ascending=False)

print(separation.round(2))

# ==========================================
# 11. GENERATE FIVE OBSERVATIONS
# ==========================================

print("\n========== FIVE CUSTOMER OBSERVATIONS ==========")

observations = []

# Observation 1-3: Feature comparisons
for feature in features:

    highest_cluster = profile[feature].idxmax()
    lowest_cluster = profile[feature].idxmin()

    observation = (
        f"{feature}: Cluster {highest_cluster} has the highest "
        f"average ({profile.loc[highest_cluster, feature]}), "
        f"while Cluster {lowest_cluster} has the lowest "
        f"average ({profile.loc[lowest_cluster, feature]})."
    )

    observations.append(observation)

# Observation 4: Largest cluster
largest_cluster = cluster_counts.idxmax()

observations.append(
    f"Cluster {largest_cluster} is the largest group "
    f"with {cluster_counts[largest_cluster]} customers."
)

# Observation 5: Smallest cluster
smallest_cluster = cluster_counts.idxmin()

observations.append(
    f"Cluster {smallest_cluster} is the smallest group "
    f"with {cluster_counts[smallest_cluster]} customers."
)

for i, observation in enumerate(observations, start=1):
    print(f"{i}. {observation}")

# ==========================================
# 12. SAVE ANALYSIS REPORT
# ==========================================

with open(
    "outputs/multi_feature_analysis_report.txt",
    "w"
) as file:

    file.write("MULTI-FEATURE CUSTOMER CLUSTER ANALYSIS\n\n")

    file.write("DATASET INFORMATION\n")
    file.write(f"Total Customers: {len(df)}\n")
    file.write(f"Number of Clusters: 4\n\n")

    file.write("CLUSTER COUNTS\n")
    file.write(cluster_counts.to_string())

    file.write("\n\nCLUSTER FEATURE COMPARISON\n")
    file.write(profile.to_string())

    file.write("\n\nFEATURE SEPARATION ANALYSIS\n")
    file.write(separation.round(2).to_string())

    file.write("\n\nFIVE OBSERVATIONS\n")

    for i, observation in enumerate(observations, start=1):
        file.write(f"{i}. {observation}\n")

# ==========================================
# 13. BONUS: CORRELATION HEATMAP
# ==========================================

print("\nCreating Correlation Heatmap...")

plt.figure(figsize=(9, 7))

correlation = df[features].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Customer Feature Correlation Heatmap")

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 14. COMPLETION MESSAGE
# ==========================================

print("\n==========================================")
print("CUSTOMER CLUSTER EXPLORER COMPLETED!")
print("==========================================")

print("\nAll charts and reports are saved in outputs/")