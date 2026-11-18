import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("customer_data.csv")

# 2. Select required features
# Support the standard Mall Customers column names
df = df.rename(columns={
    "Annual Income (k$)": "Annual Income",
    "Spending Score (1-100)": "Spending Score"
})

features = ["Annual Income", "Spending Score"]

# Clean data
df = df.drop_duplicates()
df[features] = df[features].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=features)

X = df[features]

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train K-Means model
K = 5

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# 5. Convert centroids back to original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# 6. Display customer counts
cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nCUSTOMER COUNT PER CLUSTER")
print(cluster_counts)

# 7. Create scatter plot
plt.figure(figsize=(10, 7))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    plt.scatter(
        cluster_data["Annual Income"],
        cluster_data["Spending Score"],
        label=f"Cluster {cluster}",
        alpha=0.7,
        s=60
    )

# Display centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    color="black",
    edgecolors="white",
    linewidths=1.5,
    label="Centroids"
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(alpha=0.3)

# Save chart
plt.savefig("outputs/customer_clusters.png", dpi=300)
plt.show()

# 8. Save clustered dataset
df.to_csv("outputs/customer_clusters.csv", index=False)

# 9. Generate cluster profiles
profile = df.groupby("Cluster")[features].mean().round(2)

print("\nCLUSTER PROFILES")
print(profile)

# 10. Generate five observations
overall_income = df["Annual Income"].median()
overall_spending = df["Spending Score"].median()

observations = []

for cluster, row in profile.iterrows():

    income = row["Annual Income"]
    spending = row["Spending Score"]
    count = cluster_counts[cluster]

    if income >= overall_income and spending >= overall_spending:
        description = "High-income customers with high spending."

    elif income >= overall_income and spending < overall_spending:
        description = "High-income customers with relatively low spending."

    elif income < overall_income and spending >= overall_spending:
        description = "Lower-income customers with high spending."

    else:
        description = "Lower-income customers with relatively low spending."

    observations.append(
        f"Cluster {cluster}: {description} "
        f"Number of customers: {count}. "
        f"Average income: {income}. "
        f"Average spending score: {spending}."
    )

# Save observation report
with open("outputs/observation_report.txt", "w") as file:

    file.write("CUSTOMER CLUSTER OBSERVATION REPORT\n\n")

    for observation in observations:
        file.write(observation + "\n\n")

print("\nFIVE CUSTOMER OBSERVATIONS")

for observation in observations:
    print(observation)

print("\nMain visualization and reports saved successfully.")


# BONUS: Compare K = 3, 4 and 5

for k in [3, 4, 5]:

    bonus_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    bonus_labels = bonus_model.fit_predict(X_scaled)

    bonus_centroids = scaler.inverse_transform(
        bonus_model.cluster_centers_
    )

    plt.figure(figsize=(9, 6))

    for cluster in range(k):

        cluster_data = df[bonus_labels == cluster]

        plt.scatter(
            cluster_data["Annual Income"],
            cluster_data["Spending Score"],
            label=f"Cluster {cluster}",
            alpha=0.7
        )

    plt.scatter(
        bonus_centroids[:, 0],
        bonus_centroids[:, 1],
        marker="X",
        color="black",
        s=250,
        label="Centroids"
    )

    plt.title(f"Customer Segmentation - K = {k}")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(f"outputs/customer_clusters_k{k}.png", dpi=300)
    plt.close()

print("\nBonus visualizations completed!")