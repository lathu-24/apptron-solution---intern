
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Customer Segmentation",layout="wide")
st.title("Customer Segmentation Dashboard")
upload=st.file_uploader("Upload customer CSV (optional)",type=["csv"])
df=pd.read_csv(upload) if upload else pd.read_csv(ROOT/"data"/"customer_data.csv")
options=[c for c in ["Age","Annual Income","Spending Score"] if c in df.columns]
features=st.multiselect("Choose clustering features",options,default=["Annual Income","Spending Score"])
k=st.slider("Choose K",2,10,3)
if len(features)<2:
    st.warning("Select at least two features."); st.stop()
x=df[features].apply(pd.to_numeric,errors="coerce").fillna(df[features].median())
scaler=StandardScaler(); z=scaler.fit_transform(x)
model=KMeans(n_clusters=k,random_state=42,n_init=10)
df["Cluster"]=model.fit_predict(z)
a,b=st.columns(2); a.metric("Customers",len(df)); b.metric("Clusters",k)
st.subheader("Cluster profiles")
profile=df.groupby("Cluster")[features].mean().round(2)
profile["Customer Count"]=df.Cluster.value_counts().sort_index()
st.dataframe(profile,use_container_width=True)
st.subheader("Cluster plot")
xcol=st.selectbox("X-axis",features,index=0); ycol=st.selectbox("Y-axis",features,index=1)
fig,ax=plt.subplots()
for c in sorted(df.Cluster.unique()):
    part=df[df.Cluster==c]; ax.scatter(part[xcol],part[ycol],alpha=.65,label=f"Cluster {c}")
ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.legend()
st.pyplot(fig)
st.download_button("Download labelled CSV",df.to_csv(index=False),"customer_clusters.csv","text/csv")
