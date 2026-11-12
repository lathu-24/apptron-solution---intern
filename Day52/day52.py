# ==========================================================
# DAY 52 - AUTOMATIC ELBOW ANALYSIS TOOL
# Customer Segmentation using K-Means
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ----------------------------------------------------------
# 1. SETUP
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_INIT = 10

FEATURES = ["Annual Income", "Spending Score"]


# ----------------------------------------------------------
# 2. LOAD DATASET
# ----------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n========== CUSTOMER DATA ==========")
print(df.head(10))

print("\nDataset Shape:", df.shape)


# ----------------------------------------------------------
# 3. SELECT FEATURES
# ----------------------------------------------------------

X = df[FEATURES].copy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

print("\nSelected Features:")
print(X.head())


# ----------------------------------------------------------
# 4. APPLY STANDARD SCALER
# ----------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n========== SCALED DATA ==========")

print(pd.DataFrame(
    X_scaled,
    columns=FEATURES
).head())


# ----------------------------------------------------------
# 5. REUSABLE ELBOW ANALYSIS FUNCTION
# ----------------------------------------------------------

def elbow_analysis(data, k_range=range(1, 11)):

    inertia_values = []

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )

        model.fit(data)

        inertia_values.append({
            "K": k,
            "Inertia": model.inertia_
        })

    results = pd.DataFrame(inertia_values)

    return results


# ----------------------------------------------------------
# 6. TEST K VALUES FROM 1 TO 10
# ----------------------------------------------------------

elbow_results = elbow_analysis(
    X_scaled,
    range(1, 11)
)

print("\n========== INERTIA TABLE ==========")

print(elbow_results.round(3))

elbow_results.to_csv(
    OUTPUT_DIR / "inertia_table.csv",
    index=False
)


# ----------------------------------------------------------
# 7. IDENTIFY CANDIDATE ELBOW
# ----------------------------------------------------------

def find_elbow(results):

    x = results["K"].to_numpy(dtype=float)
    y = results["Inertia"].to_numpy(dtype=float)

    # Normalize values
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    # Calculate distance from the straight line
    # joining the first and last points

    distances = np.abs(
        (y[-1] - y[0]) * x
        - (x[-1] - x[0]) * y
        + x[-1] * y[0]
        - y[-1] * x[0]
    ) / np.sqrt(
        (y[-1] - y[0]) ** 2
        + (x[-1] - x[0]) ** 2
    )

    # Exclude first and last points
    elbow_index = np.argmax(distances[1:-1]) + 1

    return int(results.iloc[elbow_index]["K"])


selected_k = find_elbow(elbow_results)

print("\nCandidate Elbow K:", selected_k)


# ----------------------------------------------------------
# 8. PLOT ELBOW METHOD
# ----------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    elbow_results["K"],
    elbow_results["Inertia"],
    marker="o",
    linewidth=2
)

plt.axvline(
    selected_k,
    linestyle="--",
    label=f"Selected K = {selected_k}"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "elbow_chart.png",
    dpi=150
)

plt.show()


# ----------------------------------------------------------
# 9. BONUS - SILHOUETTE SCORE COMPARISON
# ----------------------------------------------------------

silhouette_results = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_results.append({
        "K": k,
        "Silhouette Score": score
    })


silhouette_df = pd.DataFrame(silhouette_results)

print("\n========== SILHOUETTE SCORES ==========")

print(silhouette_df.round(4))

silhouette_df.to_csv(
    OUTPUT_DIR / "silhouette_scores.csv",
    index=False
)


# Find the K with the highest silhouette score

best_silhouette_k = int(
    silhouette_df.loc[
        silhouette_df["Silhouette Score"].idxmax(),
        "K"
    ]
)

print("\nBest Silhouette K:", best_silhouette_k)


# Silhouette chart

plt.figure(figsize=(9, 6))

plt.plot(
    silhouette_df["K"],
    silhouette_df["Silhouette Score"],
    marker="o",
    linewidth=2
)

plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.xticks(range(2, 11))
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "silhouette_chart.png",
    dpi=150
)

plt.show()


# ----------------------------------------------------------
# 10. TRAIN FINAL K-MEANS MODEL
# ----------------------------------------------------------

final_model = KMeans(
    n_clusters=selected_k,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)

df["Cluster"] = final_model.fit_predict(X_scaled)

print("\n========== FINAL CLUSTER RESULTS ==========")

print(df.head(10))


# ----------------------------------------------------------
# 11. DISPLAY CLUSTER COUNTS
# ----------------------------------------------------------

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\n========== CUSTOMER COUNTS ==========")

print(cluster_counts)


# ----------------------------------------------------------
# 12. SAVE CUSTOMER DATA WITH CLUSTER LABELS
# ----------------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "customer_clusters.csv",
    index=False
)


# ----------------------------------------------------------
# 13. PLOT FINAL CLUSTERS
# ----------------------------------------------------------

centroids = final_model.cluster_centers_

plt.figure(figsize=(9, 6))

for cluster in range(selected_k):

    cluster_data = X_scaled[
        df["Cluster"].values == cluster
    ]

    plt.scatter(
        cluster_data[:, 0],
        cluster_data[:, 1],
        alpha=0.7,
        label=f"Cluster {cluster}"
    )


# Display centroids in scaled feature space

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    edgecolors="black",
    label="Centroids"
)

plt.title(f"Final Customer Clusters (K={selected_k})")

plt.xlabel("Annual Income (Standardized)")
plt.ylabel("Spending Score (Standardized)")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_clusters.png",
    dpi=150
)

plt.show()


# ----------------------------------------------------------
# 14. GENERATE WRITTEN ANALYSIS REPORT
# ----------------------------------------------------------

if selected_k == best_silhouette_k:

    comparison = "Both methods suggest the same K value."

else:

    comparison = (
        f"Elbow suggests K={selected_k}, while "
        f"Silhouette suggests K={best_silhouette_k}."
    )


report = f"""
CUSTOMER CLUSTERING - ANALYSIS REPORT

Dataset Records: {len(df)}

Selected Features:
- Annual Income
- Spending Score

Scaling Method:
StandardScaler

K Values Tested:
1 to 10

Selected K:
{selected_k}

Silhouette Best K:
{best_silhouette_k}

METHOD COMPARISON:
{comparison}

WHY THIS K WAS SELECTED:

The selected K represents the estimated elbow point,
where the reduction in inertia starts slowing down.

WHY THE ELBOW IS NOT ALWAYS DEFINITIVE:

Some datasets do not have a clear elbow point.
The inertia curve may decrease gradually, making
the best K difficult to identify.

Therefore, the Elbow Method should be supported by
Silhouette Scores, cluster visualizations and
business knowledge.

CLUSTER COUNTS:

{cluster_counts.to_string()}

CONCLUSION:

K-Means successfully grouped customers according to
their Annual Income and Spending Score.

These clusters can help identify different customer
behaviour patterns for marketing and business decisions.
"""

with open(
    OUTPUT_DIR / "analysis_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


print("\n========== FINAL SUMMARY ==========")

print("Selected K:", selected_k)
print("Silhouette Best K:", best_silhouette_k)

print("\nAll tasks completed successfully!")

print("\nCheck the outputs folder for:")
print("1. Inertia table")
print("2. Elbow chart")
print("3. Silhouette scores")
print("4. Silhouette chart")
print("5. Customer dataset with cluster labels")
print("6. Final cluster visualization")
print("7. Analysis report")