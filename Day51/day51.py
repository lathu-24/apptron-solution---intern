# ============================================================
# DAY 51 - BUILD YOUR FIRST K-MEANS MODEL
# Customer Segmentation using K-Means Clustering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ------------------------------------------------------------
# 1. SETUP
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== CUSTOMER DATASET ==========")
print(df.head(10))

print("\nDataset Shape:", df.shape)


# ------------------------------------------------------------
# 2. SELECT FEATURES
# ------------------------------------------------------------

features = ["Annual Income", "Spending Score"]

X = df[features].copy()

print("\nSelected Features:")
print(X.head())


# ------------------------------------------------------------
# 3. CHECK FEATURE SCALES
# ------------------------------------------------------------

print("\n========== FEATURE SCALE ANALYSIS ==========")

print(X.describe())

print("\nFeature Ranges:")
for column in features:
    feature_range = X[column].max() - X[column].min()
    print(f"{column}: {feature_range:.2f}")


# ------------------------------------------------------------
# 4. APPLY STANDARD SCALER
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

print("\n========== SCALED DATA ==========")
print(scaled_df.head())

print("\nScaled Feature Means:")
print(scaled_df.mean().round(3))

print("\nScaled Feature Standard Deviations:")
print(scaled_df.std().round(3))


# ------------------------------------------------------------
# 5. BUILD K-MEANS MODEL (K = 3)
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Fit model and predict clusters
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n========== K-MEANS RESULTS (K = 3) ==========")

print("\nFirst 10 Customers with Cluster Labels:")
print(df.head(10).to_string(index=False))


# ------------------------------------------------------------
# 6. DISPLAY CLUSTER CENTROIDS
# ------------------------------------------------------------

centroids = kmeans.cluster_centers_

centroid_df = pd.DataFrame(
    centroids,
    columns=features
)

print("\n========== CLUSTER CENTROIDS (SCALED SPACE) ==========")
print(centroid_df.round(3))


# ------------------------------------------------------------
# 7. COUNT CUSTOMERS IN EACH CLUSTER
# ------------------------------------------------------------

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\n========== CUSTOMERS PER CLUSTER ==========")
print(cluster_counts)

cluster_counts.to_csv(
    OUTPUT_DIR / "cluster_counts.csv",
    header=["Customer Count"]
)


# ------------------------------------------------------------
# 8. VISUALIZE K = 3 CLUSTERS
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = X_scaled[df["Cluster"] == cluster]

    plt.scatter(
        cluster_data[:, 0],
        cluster_data[:, 1],
        label=f"Cluster {cluster}",
        alpha=0.7
    )

# Display centroids in the same scaled feature space
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    edgecolors="black",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means (K = 3)")
plt.xlabel("Annual Income (Standardized)")
plt.ylabel("Spending Score (Standardized)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_k3_clusters.png",
    dpi=150
)

plt.show()


# ------------------------------------------------------------
# 9. SAVE CUSTOMER DATA WITH CLUSTER LABELS
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "customer_data_with_clusters.csv",
    index=False
)

print("\nCluster-labelled dataset saved successfully!")


# ------------------------------------------------------------
# 10. ANALYZE CLUSTER CHARACTERISTICS
# ------------------------------------------------------------

print("\n========== CLUSTER CHARACTERISTICS ==========")

cluster_analysis = df.groupby("Cluster")[features].mean().round(2)

print(cluster_analysis)

cluster_analysis.to_csv(
    OUTPUT_DIR / "cluster_characteristics.csv"
)


print("\n========== CLUSTER OBSERVATIONS ==========")

for cluster in sorted(df["Cluster"].unique()):

    income_avg = df[df["Cluster"] == cluster]["Annual Income"].mean()

    spending_avg = df[df["Cluster"] == cluster]["Spending Score"].mean()

    print(f"\nCluster {cluster}:")
    print(f"Average Annual Income: {income_avg:.2f}")
    print(f"Average Spending Score: {spending_avg:.2f}")

    if income_avg >= df["Annual Income"].median() and spending_avg >= df["Spending Score"].median():
        print("Group: Higher-income customers with higher spending.")

    elif income_avg >= df["Annual Income"].median() and spending_avg < df["Spending Score"].median():
        print("Group: Higher-income customers with lower spending.")

    elif income_avg < df["Annual Income"].median() and spending_avg >= df["Spending Score"].median():
        print("Group: Lower-income customers with higher spending.")

    else:
        print("Group: Lower-income customers with lower spending.")


# ------------------------------------------------------------
# 11. BONUS - EXPERIMENT WITH K = 2 AND K = 4
# ------------------------------------------------------------

print("\n========== BONUS: K-MEANS COMPARISON ==========")

comparison_results = []

for k in [2, 3, 4]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    counts = pd.Series(labels).value_counts().sort_index()

    print(f"\nK = {k}")
    print("Cluster Counts:")
    print(counts)

    comparison_results.append({
        "K": k,
        "Inertia": model.inertia_,
        "Number of Clusters": k
    })

    # Plot each experiment
    plt.figure(figsize=(9, 6))

    for cluster in range(k):

        cluster_points = X_scaled[labels == cluster]

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {cluster}",
            alpha=0.7
        )

    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=250,
        edgecolors="black",
        label="Centroids"
    )

    plt.title(f"Customer Segmentation (K = {k})")
    plt.xlabel("Annual Income (Standardized)")
    plt.ylabel("Spending Score (Standardized)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"kmeans_k{k}_clusters.png",
        dpi=150
    )

    plt.close()


# ------------------------------------------------------------
# 12. COMPARE K VALUES
# ------------------------------------------------------------

comparison_df = pd.DataFrame(comparison_results)

print("\n========== K VALUE COMPARISON ==========")
print(comparison_df)

comparison_df.to_csv(
    OUTPUT_DIR / "kmeans_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print("\n========== FINAL SUMMARY ==========")

print("Dataset Records:", len(df))
print("Selected Features:", features)
print("Main Model: K-Means")
print("Main Number of Clusters: 3")
print("Random State: 42")
print("N Initializations: 10")

print("\nAll tasks completed successfully!")

print("\nCheck the outputs folder for:")
print("1. Customer data with cluster labels")
print("2. Cluster counts")
print("3. Cluster characteristics")
print("4. K-Means scatter plots")
print("5. K-value comparison")