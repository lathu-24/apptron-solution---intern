import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# ADVANCED CLUSTER EXPLORER
# ==========================================

print("=" * 60)
print("          ADVANCED CLUSTER EXPLORER")
print("=" * 60)

os.makedirs("day59_outputs", exist_ok=True)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n[1] Loading customer dataset...")

df = pd.read_csv("customer_data.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

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
# 3. GENERATE CLUSTER LABELS
# ==========================================

print("\n[3] Generating cluster labels...")

cluster_features = ["Annual Income", "Spending Score"]

scaler_cluster = StandardScaler()

X_cluster_scaled = scaler_cluster.fit_transform(
    df[cluster_features]
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_cluster_scaled)

print("Clustering completed!")

print("\nCluster distribution:")
print(df["Cluster"].value_counts().sort_index())

# Save dataset with cluster labels
df.to_csv(
    "day59_outputs/customer_data_with_clusters.csv",
    index=False
)

# ==========================================
# 4. SELECT NUMERICAL FEATURES
# ==========================================

print("\n[4] Selecting numerical features...")

numerical_features = [
    "Age",
    "Annual Income",
    "Spending Score"
]

X = df[numerical_features]

print("Selected features:")
print(numerical_features)

print("Number of features:", len(numerical_features))

# ==========================================
# 5. STANDARDIZE FEATURES
# ==========================================

print("\n[5] Standardizing features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Standardization completed.")

# ==========================================
# 6. APPLY PCA
# ==========================================

print("\n[6] Applying PCA...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

print("PCA transformation completed.")

# ==========================================
# 7. EXPLAINED VARIANCE ANALYSIS
# ==========================================

print("\n[7] Explained variance analysis...")

explained_variance = pca.explained_variance_ratio_

pc1_variance = explained_variance[0] * 100
pc2_variance = explained_variance[1] * 100

total_variance = explained_variance.sum() * 100

print(f"\nPC1 Explained Variance: {pc1_variance:.2f}%")
print(f"PC2 Explained Variance: {pc2_variance:.2f}%")
print(f"Total Explained Variance: {total_variance:.2f}%")

variance_report = pd.DataFrame({
    "Principal Component": ["PC1", "PC2"],
    "Explained Variance Ratio": explained_variance,
    "Explained Variance Percentage": explained_variance * 100
})

variance_report.to_csv(
    "day59_outputs/explained_variance_report.csv",
    index=False
)

# ==========================================
# 8. PCA VISUALIZATION
# ==========================================

print("\n[8] Creating PCA visualization...")

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=80
)

plt.title(
    f"PCA Customer Cluster Visualization\n"
    f"Explained Variance: {total_variance:.2f}%"
)

plt.xlabel(f"PC1 ({pc1_variance:.2f}% variance)")
plt.ylabel(f"PC2 ({pc2_variance:.2f}% variance)")

plt.legend(title="Cluster")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "day59_outputs/pca_cluster_visualization.png",
    dpi=300
)

plt.show()

# ==========================================
# 9. INTERACTIVE PLOTLY SCATTER PLOT
# ==========================================

print("\n[9] Creating interactive customer scatter plot...")

fig = px.scatter(
    df,
    x="Annual Income",
    y="Spending Score",
    color="Cluster",
    hover_data=[
        "Customer ID",
        "Age",
        "Annual Income",
        "Spending Score",
        "Cluster"
    ],
    title="Interactive Customer Cluster Explorer",
    labels={
        "Annual Income": "Annual Income",
        "Spending Score": "Spending Score",
        "Cluster": "Customer Cluster"
    }
)

fig.update_layout(
    width=1000,
    height=700,
    template="plotly_white"
)

fig.write_html(
    "day59_outputs/interactive_customer_scatter.html"
)

fig.show()

# ==========================================
# 10. ORIGINAL FEATURE SCATTER PLOT
# ==========================================

print("\n[10] Creating original feature scatter plot...")

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df,
    x="Annual Income",
    y="Spending Score",
    hue="Cluster",
    palette="tab10",
    s=80
)

plt.title("Customer Clusters - Original Features")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.legend(title="Cluster")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "day59_outputs/original_feature_scatter.png",
    dpi=300
)

plt.show()

# ==========================================
# 11. PCA VS ORIGINAL FEATURE COMPARISON
# ==========================================

print("\n[11] Comparing original features with PCA...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Original features
sns.scatterplot(
    data=df,
    x="Annual Income",
    y="Spending Score",
    hue="Cluster",
    palette="tab10",
    s=60,
    ax=axes[0]
)

axes[0].set_title("Original Features")
axes[0].set_xlabel("Annual Income")
axes[0].set_ylabel("Spending Score")

# PCA features
sns.scatterplot(
    data=df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=60,
    ax=axes[1],
    legend=False
)

axes[1].set_title("PCA Visualization")
axes[1].set_xlabel(f"PC1 ({pc1_variance:.2f}%)")
axes[1].set_ylabel(f"PC2 ({pc2_variance:.2f}%)")

plt.suptitle("Original Features vs PCA Visualization")

plt.tight_layout()

plt.savefig(
    "day59_outputs/original_vs_pca_comparison.png",
    dpi=300
)

plt.show()

# ==========================================
# 12. FIVE OBSERVATIONS
# ==========================================

print("\n[12] Generating observations...")

observations = []

# Observation 1
observations.append(
    f"PCA reduces the original {len(numerical_features)} numerical features "
    f"into two principal components."
)

# Observation 2
observations.append(
    f"PC1 explains {pc1_variance:.2f}% of the total data variance."
)

# Observation 3
observations.append(
    f"PC2 explains {pc2_variance:.2f}% of the total data variance."
)

# Observation 4
observations.append(
    f"The two PCA components together explain {total_variance:.2f}% "
    f"of the total variance."
)

# Observation 5
observations.append(
    "The PCA plot may show cluster separation or overlap, helping "
    "visualize relationships between customer groups."
)

print("\nOBSERVATIONS:")

for i, observation in enumerate(observations, 1):
    print(f"{i}. {observation}")

# ==========================================
# 13. PCA LIMITATION
# ==========================================

pca_limitation = (
    "PCA reduces multiple dimensions into two components, which makes "
    "visualization easier. However, some information is lost during "
    "dimensionality reduction. Customers that are well separated in the "
    "original feature space may appear closer together or overlap in the "
    "2D PCA plot. Therefore, the PCA visualization may not completely "
    "represent the original customer cluster structure."
)

print("\nPCA LIMITATION:")
print(pca_limitation)

# ==========================================
# 14. BONUS - 3D PCA
# ==========================================

print("\n[14] Creating bonus 3D PCA visualization...")

pca_3d = PCA(n_components=3)

X_pca_3d = pca_3d.fit_transform(X_scaled)

df["PC3"] = X_pca_3d[:, 2]

variance_3d = (
    pca_3d.explained_variance_ratio_.sum() * 100
)

fig_3d = px.scatter_3d(
    df,
    x="PC1",
    y="PC2",
    z="PC3",
    color="Cluster",
    hover_data=[
        "Customer ID",
        "Age",
        "Annual Income",
        "Spending Score",
        "Cluster"
    ],
    title=f"3D PCA Customer Cluster Explorer - "
          f"Explained Variance: {variance_3d:.2f}%"
)

fig_3d.write_html(
    "day59_outputs/3d_pca_cluster_visualization.html"
)

fig_3d.show()

# ==========================================
# 15. ANALYSIS REPORT
# ==========================================

print("\n[15] Creating analysis report...")

with open(
    "day59_outputs/analysis_report.txt",
    "w"
) as file:

    file.write("ADVANCED CLUSTER EXPLORER - ANALYSIS REPORT\n")
    file.write("=" * 60 + "\n\n")

    file.write("DATASET INFORMATION\n")
    file.write("-" * 30 + "\n")
    file.write(f"Total customers: {len(df)}\n")
    file.write(f"Features used: {numerical_features}\n")
    file.write(f"Number of clusters: {df['Cluster'].nunique()}\n\n")

    file.write("PCA EXPLAINED VARIANCE\n")
    file.write("-" * 30 + "\n")
    file.write(f"PC1: {pc1_variance:.2f}%\n")
    file.write(f"PC2: {pc2_variance:.2f}%\n")
    file.write(f"Total: {total_variance:.2f}%\n\n")

    file.write("OBSERVATIONS\n")
    file.write("-" * 30 + "\n")

    for i, observation in enumerate(observations, 1):
        file.write(f"{i}. {observation}\n\n")

    file.write("PCA LIMITATION\n")
    file.write("-" * 30 + "\n")
    file.write(pca_limitation + "\n\n")

    file.write("BONUS 3D PCA\n")
    file.write("-" * 30 + "\n")
    file.write(
        f"First three components explain {variance_3d:.2f}% "
        f"of total variance.\n"
    )

print("Analysis report generated.")

# ==========================================
# 16. SAVE FINAL DATA
# ==========================================

df.to_csv(
    "day59_outputs/customer_data_with_pca.csv",
    index=False
)

# ==========================================
# 17. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("ADVANCED CLUSTER EXPLORER COMPLETED")
print("=" * 60)

print(f"\nTotal customers: {len(df)}")
print(f"Total clusters: {df['Cluster'].nunique()}")

print(f"\nPC1 Variance: {pc1_variance:.2f}%")
print(f"PC2 Variance: {pc2_variance:.2f}%")
print(f"Total Variance: {total_variance:.2f}%")

print("\nGenerated files:")
print("- pca_cluster_visualization.png")
print("- explained_variance_report.csv")
print("- interactive_customer_scatter.html")
print("- original_feature_scatter.png")
print("- original_vs_pca_comparison.png")
print("- 3d_pca_cluster_visualization.html")
print("- analysis_report.txt")
print("- customer_data_with_clusters.csv")
print("- customer_data_with_pca.csv")

print("\nAll tasks completed successfully!")