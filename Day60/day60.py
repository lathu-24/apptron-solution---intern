# ============================================================
# CUSTOMER SEGMENTATION ANALYTICS DASHBOARD
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Segmentation Analytics Dashboard")

st.markdown("""
Explore customer behavior, discover meaningful customer segments,
and generate data-driven business insights using K-Means clustering.
""")

st.divider()


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/Mall_Customers.csv"

OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):

        return None

    data = pd.read_csv(DATA_PATH)

    return data


df_original = load_data()


# ============================================================
# DATASET UPLOAD OPTION
# ============================================================

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df_original = pd.read_csv(uploaded_file)


if df_original is None:

    st.error(
        "Dataset not found. Please place Mall_Customers.csv "
        "inside the data folder or upload a CSV file."
    )

    st.info("""
    Expected dataset columns:

    - CustomerID
    - Gender
    - Age
    - Annual Income (k$)
    - Spending Score (1-100)
    """)

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df = df_original.copy()


# Remove duplicate rows
df = df.drop_duplicates()


# Identify numerical columns
numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


# Fill missing numerical values
for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# Fill missing categorical values
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()


for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()

        if not mode_value.empty:

            df[column] = df[column].fillna(
                mode_value[0]
            )


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(columns, possible_names):

    for name in possible_names:

        if name in columns:

            return name

    return None


customer_id_column = find_column(
    df.columns,
    ["CustomerID", "Customer Id", "customer_id", "ID"]
)

age_column = find_column(
    df.columns,
    ["Age", "age"]
)

income_column = find_column(
    df.columns,
    [
        "Annual Income (k$)",
        "Annual Income",
        "Income",
        "annual_income"
    ]
)

spending_column = find_column(
    df.columns,
    [
        "Spending Score (1-100)",
        "Spending Score",
        "Spending Score (1-100)",
        "spending_score"
    ]
)

gender_column = find_column(
    df.columns,
    ["Gender", "gender"]
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("Dashboard Settings")


# ============================================================
# CLUSTERING FEATURES
# ============================================================

available_numeric_features = df.select_dtypes(
    include=np.number
).columns.tolist()


excluded_columns = []

if customer_id_column:

    excluded_columns.append(
        customer_id_column
    )


clustering_features = [
    column
    for column in available_numeric_features
    if column not in excluded_columns
]


if len(clustering_features) < 2:

    st.error(
        "At least two numerical features are required "
        "for clustering."
    )

    st.stop()


selected_features = st.sidebar.multiselect(
    "Select clustering features",
    clustering_features,
    default=clustering_features
)


if len(selected_features) < 2:

    st.warning(
        "Please select at least two clustering features."
    )

    st.stop()


# ============================================================
# K-MEANS SETTINGS
# ============================================================

cluster_options = [3, 4, 5, 6, 7, 8]

selected_k = st.sidebar.selectbox(
    "Number of clusters (K)",
    cluster_options,
    index=2
)


# ============================================================
# CLUSTERING
# ============================================================

X = df[selected_features].copy()


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


kmeans = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10
)


cluster_labels = kmeans.fit_predict(
    X_scaled
)


df["Cluster"] = cluster_labels


# ============================================================
# CENTROIDS
# ============================================================

centroids_scaled = kmeans.cluster_centers_


centroids_original = scaler.inverse_transform(
    centroids_scaled
)


centroid_df = pd.DataFrame(
    centroids_original,
    columns=selected_features
)


centroid_df["Cluster"] = range(
    selected_k
)


# ============================================================
# PCA
# ============================================================

pca = PCA(
    n_components=2
)


pca_result = pca.fit_transform(
    X_scaled
)


df["PCA1"] = pca_result[:, 0]

df["PCA2"] = pca_result[:, 1]


explained_variance = (
    pca.explained_variance_ratio_.sum() * 100
)


# ============================================================
# CLUSTER PROFILE
# ============================================================

cluster_profile = df.groupby(
    "Cluster"
)[selected_features].mean()


cluster_sizes = df.groupby(
    "Cluster"
).size()


cluster_profile["Customer Count"] = (
    cluster_sizes
)


cluster_profile["Customer Percentage"] = (
    cluster_sizes / len(df) * 100
)


cluster_profile = cluster_profile.round(
    2
)


# ============================================================
# FILTER
# ============================================================

cluster_selection = st.sidebar.selectbox(
    "Select customer cluster",
    ["All Clusters"] + list(range(selected_k))
)


if cluster_selection == "All Clusters":

    filtered_df = df.copy()

else:

    filtered_df = df[
        df["Cluster"] == cluster_selection
    ].copy()


# ============================================================
# DATA EXPORT
# ============================================================

cleaned_data = df.copy()


clustered_csv = cleaned_data.to_csv(
    index=False
).encode("utf-8")


# ============================================================
# SIDEBAR DOWNLOAD
# ============================================================

st.sidebar.download_button(
    label="⬇ Download Clustered Dataset",
    data=clustered_csv,
    file_name="customer_dataset_clustered.csv",
    mime="text/csv"
)


filtered_csv = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.sidebar.download_button(
    label="⬇ Download Filtered Customers",
    data=filtered_csv,
    file_name="filtered_customers.csv",
    mime="text/csv"
)


# ============================================================
# SAVE DATASETS
# ============================================================

cleaned_data.to_csv(
    "outputs/customer_dataset_clustered.csv",
    index=False
)


cluster_profile.to_csv(
    "outputs/cluster_profiles.csv"
)


centroid_df.to_csv(
    "outputs/cluster_centroids.csv",
    index=False
)


# ============================================================
# KPI SECTION
# ============================================================

st.header("Business Overview")


total_customers = len(df)


number_of_clusters = df["Cluster"].nunique()


average_spending = (
    df[spending_column].mean()
    if spending_column
    else 0
)


average_income = (
    df[income_column].mean()
    if income_column
    else 0
)


kpi1, kpi2, kpi3, kpi4 = st.columns(4)


kpi1.metric(
    "Total Customers",
    f"{total_customers:,}"
)


kpi2.metric(
    "Number of Clusters",
    number_of_clusters
)


kpi3.metric(
    "Average Spending Score",
    f"{average_spending:.2f}"
    if spending_column
    else "N/A"
)


kpi4.metric(
    "Average Annual Income",
    f"{average_income:.2f}"
    if income_column
    else "N/A"
)


st.divider()


# ============================================================
# CLUSTER SIZE CHART
# ============================================================

st.header("Customer Cluster Distribution")


cluster_counts = df["Cluster"].value_counts().sort_index()


fig_cluster_size = px.bar(
    x=cluster_counts.index.astype(str),
    y=cluster_counts.values,
    labels={
        "x": "Cluster",
        "y": "Number of Customers"
    },
    title="Customer Distribution by Cluster",
    text=cluster_counts.values
)


fig_cluster_size.update_layout(
    template="plotly_white",
    height=450
)


st.plotly_chart(
    fig_cluster_size,
    use_container_width=True
)


# ============================================================
# INCOME VS SPENDING
# ============================================================

if income_column and spending_column:

    st.header("Annual Income vs Spending Score")


    fig_income_spending = px.scatter(
        df,
        x=income_column,
        y=spending_column,
        color="Cluster",
        title="Customer Segmentation",
        hover_data=df.columns.tolist(),
        template="plotly_white"
    )


    # Add centroids if these features were selected for clustering
    if (
        income_column in selected_features
        and spending_column in selected_features
    ):

        income_index = selected_features.index(
            income_column
        )

        spending_index = selected_features.index(
            spending_column
        )


        fig_income_spending.add_trace(
            go.Scatter(
                x=centroid_df[income_column],
                y=centroid_df[spending_column],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=18,
                    color="black",
                    line=dict(width=3)
                ),
                name="Cluster Centroids"
            )
        )


    fig_income_spending.update_layout(
        height=600
    )


    st.plotly_chart(
        fig_income_spending,
        use_container_width=True
    )


# ============================================================
# CLUSTER HEATMAP
# ============================================================

st.header("Cluster Profile Heatmap")


heatmap_data = df.groupby(
    "Cluster"
)[selected_features].mean()


heatmap_normalized = (
    heatmap_data - heatmap_data.mean()
) / heatmap_data.std().replace(0, 1)


fig_heatmap, ax = plt.subplots(
    figsize=(12, 6)
)


sns.heatmap(
    heatmap_normalized,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax
)


ax.set_title(
    "Normalized Customer Cluster Profiles"
)


st.pyplot(
    fig_heatmap,
    use_container_width=True
)


fig_heatmap.savefig(
    "outputs/cluster_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig_heatmap)


# ============================================================
# PCA VISUALIZATION
# ============================================================

st.header("PCA Cluster Visualization")


fig_pca = px.scatter(
    df,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    hover_data=df.columns.tolist(),
    title="Customer Clusters Using PCA",
    template="plotly_white"
)


fig_pca.update_layout(
    height=600
)


st.plotly_chart(
    fig_pca,
    use_container_width=True
)


st.caption(
    f"Total explained variance by the two principal components: "
    f"{explained_variance:.2f}%"
)


# Save PCA chart

fig_pca.write_html(
    "outputs/pca_visualization.html"
)


# ============================================================
# INTERACTIVE FEATURE SCATTER PLOT
# ============================================================

st.header("Interactive Feature Exploration")


feature_options = selected_features


x_axis = st.selectbox(
    "Select X-axis",
    feature_options,
    index=0
)


y_axis = st.selectbox(
    "Select Y-axis",
    feature_options,
    index=1
)


fig_interactive = px.scatter(
    filtered_df,
    x=x_axis,
    y=y_axis,
    color="Cluster",
    hover_data=df.columns.tolist(),
    title=f"{x_axis} vs {y_axis}",
    template="plotly_white"
)


fig_interactive.update_layout(
    height=600
)


st.plotly_chart(
    fig_interactive,
    use_container_width=True
)


fig_interactive.write_html(
    "outputs/interactive_scatter.html"
)


# ============================================================
# CLUSTER PROFILE COMPARISON
# ============================================================

st.header("Cluster Profile Comparison")


st.dataframe(
    cluster_profile,
    use_container_width=True
)


# ============================================================
# CENTROID TABLE
# ============================================================

st.header("Cluster Centroid Information")


st.dataframe(
    centroid_df.round(2),
    use_container_width=True
)


# ============================================================
# CLUSTER BUSINESS INSIGHTS
# ============================================================

st.header("Business Insights")


insights = []


# Cluster size insight

largest_cluster = cluster_sizes.idxmax()

largest_cluster_size = cluster_sizes.max()


largest_cluster_percentage = (
    largest_cluster_size / len(df) * 100
)


insights.append(
    f"Cluster {largest_cluster} is the largest customer segment, "
    f"containing {largest_cluster_size} customers "
    f"({largest_cluster_percentage:.2f}% of the dataset). "
    f"Marketing strategies targeting this segment may influence "
    f"a relatively large portion of customers."
)


smallest_cluster = cluster_sizes.idxmin()


smallest_cluster_size = cluster_sizes.min()


smallest_cluster_percentage = (
    smallest_cluster_size / len(df) * 100
)


insights.append(
    f"Cluster {smallest_cluster} is the smallest customer segment, "
    f"containing {smallest_cluster_size} customers "
    f"({smallest_cluster_percentage:.2f}% of the dataset). "
    f"This segment may require a specialized strategy."
)


# Feature-based insights

for feature in selected_features:

    highest_cluster = cluster_profile[
        feature
    ].idxmax()


    lowest_cluster = cluster_profile[
        feature
    ].idxmin()


    highest_value = cluster_profile.loc[
        highest_cluster,
        feature
    ]


    lowest_value = cluster_profile.loc[
        lowest_cluster,
        feature
    ]


    insights.append(
        f"Cluster {highest_cluster} has the highest average "
        f"{feature} ({highest_value:.2f}), while Cluster "
        f"{lowest_cluster} has the lowest average "
        f"{feature} ({lowest_value:.2f}). "
        f"This indicates a meaningful difference in customer "
        f"behavior across these segments."
    )


# Display insights

for index, insight in enumerate(
    insights,
    start=1
):

    st.markdown(
        f"""
        **Insight {index}**

        {insight}
        """
    )


# ============================================================
# CUSTOMER RECORDS
# ============================================================

st.header("Customer Records")


st.write(
    f"Showing {len(filtered_df)} customer records."
)


st.dataframe(
    filtered_df,
    use_container_width=True
)


# ============================================================
# DOWNLOAD FILTERED DATA
# ============================================================

st.download_button(
    label="Download Selected Customer Records",
    data=filtered_csv,
    file_name="selected_cluster_customers.csv",
    mime="text/csv"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("Dataset Information")


col1, col2 = st.columns(2)


with col1:

    st.subheader("Dataset Shape")

    st.write(
        f"Rows: {df.shape[0]}"
    )

    st.write(
        f"Columns: {df.shape[1]}"
    )


with col2:

    st.subheader("Data Quality")

    st.write(
        f"Duplicate rows: {df_original.duplicated().sum()}"
    )

    st.write(
        f"Missing values: {df_original.isnull().sum().sum()}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.markdown("""
<div style="text-align:center">

### Customer Segmentation Analytics Dashboard

Developed using Python, Scikit-learn, Pandas, Plotly, and Streamlit.

</div>
""", unsafe_allow_html=True)