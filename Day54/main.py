
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customer_data.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
RANDOM_STATE, N_INIT = 42, 10
FEATURES = ["Annual Income", "Spending Score"]

def load_clean(path=DATA):
    df = pd.read_csv(path)
    required = ["Customer ID", "Age", "Annual Income", "Spending Score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    before = len(df)
    df = df.drop_duplicates().copy()
    for c in ["Age", "Annual Income", "Spending Score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[~df["Age"].between(15, 100), "Age"] = np.nan
    df.loc[df["Annual Income"] < 0, "Annual Income"] = np.nan
    df.loc[~df["Spending Score"].between(1, 100), "Spending Score"] = np.nan
    for c in ["Age", "Annual Income", "Spending Score"]:
        df[c] = df[c].fillna(df[c].median())
    df.to_csv(OUT / "clean_customer_data.csv", index=False)
    quality = {"rows_before": before, "rows_after": len(df),
               "duplicates_removed": before-len(df),
               "missing_after": int(df.isna().sum().sum())}
    return df, quality

def create_eda(df):
    numeric = ["Age", "Annual Income", "Spending Score"]
    stats = df[numeric].agg(["mean","median","min","max"]).T
    stats.columns = ["Mean","Median","Minimum","Maximum"]
    stats.to_csv(OUT / "summary_statistics.csv")
    for c in numeric:
        plt.figure(figsize=(7,4))
        plt.hist(df[c], bins=20, edgecolor="black")
        plt.title(f"Distribution of {c}")
        plt.xlabel(c); plt.ylabel("Customers"); plt.tight_layout()
        plt.savefig(OUT / f"eda_{c.lower().replace(' ','_')}.png", dpi=150); plt.close()
    plt.figure(figsize=(7,5))
    plt.scatter(df["Annual Income"],df["Spending Score"],alpha=.6)
    plt.title("Annual Income vs Spending Score")
    plt.xlabel("Annual Income"); plt.ylabel("Spending Score")
    plt.tight_layout(); plt.savefig(OUT/"eda_income_spending.png",dpi=150); plt.close()
    return stats

def evaluate_k(X):
    rows=[]
    for k in range(1,11):
        model=KMeans(n_clusters=k,random_state=RANDOM_STATE,n_init=N_INIT)
        labels=model.fit_predict(X)
        sil=silhouette_score(X,labels) if k>=2 else np.nan
        rows.append({"K":k,"Inertia":model.inertia_,"Silhouette Score":sil})
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"k_evaluation_table.csv",index=False)
    plt.figure(figsize=(8,5)); plt.plot(result.K,result.Inertia,marker="o")
    plt.title("Elbow Method"); plt.xlabel("K"); plt.ylabel("Inertia")
    plt.xticks(range(1,11)); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT/"elbow_plot.png",dpi=150); plt.close()
    sil=result.dropna()
    plt.figure(figsize=(8,5)); plt.plot(sil.K,sil["Silhouette Score"],marker="o")
    plt.title("Silhouette Scores"); plt.xlabel("K"); plt.ylabel("Silhouette Score")
    plt.xticks(range(2,11)); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT/"silhouette_scores.png",dpi=150); plt.close()
    return result

def name_segment(row, med_income, med_spend):
    hi=row["Average_Annual_Income"] >= med_income
    hs=row["Average_Spending_Score"] >= med_spend
    if hi and hs: return "Premium High-Value Customers"
    if hi and not hs: return "Affluent Low-Engagement Customers"
    if not hi and hs: return "Value-Seeking Active Spenders"
    return "Low-Spend Budget Customers"

def actions(name):
    return {
      "Premium High-Value Customers":("Offer VIP rewards and early access.","Use personalized cross-selling and retention campaigns."),
      "Affluent Low-Engagement Customers":("Send tailored re-engagement offers.","Collect feedback and test premium experiences."),
      "Value-Seeking Active Spenders":("Offer affordable bundles and coupons.","Use loyalty points to encourage repeat purchases."),
      "Low-Spend Budget Customers":("Highlight essential products and value pricing.","Use low-cost promotions and monitor response.")
    }[name]

def profile(df, labels, prefix="main"):
    result=df.copy(); result["Cluster"]=labels
    p=result.groupby("Cluster").agg(
      Customer_Count=("Customer ID","count"),
      Average_Age=("Age","mean"),
      Average_Annual_Income=("Annual Income","mean"),
      Average_Spending_Score=("Spending Score","mean")).round(2)
    p["Percentage"]=(p.Customer_Count/len(result)*100).round(2)
    mi,ms=result["Annual Income"].median(),result["Spending Score"].median()
    p["Segment_Name"]=p.apply(lambda r:name_segment(r,mi,ms),axis=1)
    rec=p.Segment_Name.apply(actions)
    p["Action_1"]=rec.apply(lambda x:x[0]); p["Action_2"]=rec.apply(lambda x:x[1])
    result.to_csv(OUT/f"{prefix}_clustered_customers.csv",index=False)
    p.to_csv(OUT/f"{prefix}_cluster_profiles.csv")
    return result,p

def assign_new_customers(new_df):
    saved=joblib.load(MODELS/"customer_kmeans_pipeline.joblib")
    return saved["model"].predict(saved["scaler"].transform(new_df[saved["features"]]))

def main():
    df,quality=load_clean()
    stats=create_eda(df)
    scaler=StandardScaler()
    X=scaler.fit_transform(df[FEATURES])
    evaluation=evaluate_k(X)
    # Highest silhouette is the automated candidate; inspect elbow and business fit too.
    valid=evaluation.dropna(subset=["Silhouette Score"])
    selected_k=int(valid.loc[valid["Silhouette Score"].idxmax(),"K"])
    model=KMeans(n_clusters=selected_k,random_state=RANDOM_STATE,n_init=N_INIT)
    labels=model.fit_predict(X)
    clustered,p=profile(df,labels)
    joblib.dump({"scaler":scaler,"model":model,"features":FEATURES,"selected_k":selected_k},
                MODELS/"customer_kmeans_pipeline.joblib")
    # Main cluster scatter with centroids in original units.
    plt.figure(figsize=(9,6))
    for c in sorted(clustered.Cluster.unique()):
        part=clustered[clustered.Cluster==c]
        plt.scatter(part["Annual Income"],part["Spending Score"],alpha=.65,
                    label=f"Cluster {c}: {p.loc[c,'Segment_Name']}")
    centers=scaler.inverse_transform(model.cluster_centers_)
    plt.scatter(centers[:,0],centers[:,1],marker="X",s=220,edgecolors="black",label="Centroids")
    plt.title("Customer Segments: Income vs Spending")
    plt.xlabel("Annual Income"); plt.ylabel("Spending Score"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT/"cluster_scatter.png",dpi=150); plt.close()
    for col,title,file in [
      ("Average_Annual_Income","Average Income by Cluster","average_income_by_cluster.png"),
      ("Average_Spending_Score","Average Spending by Cluster","average_spending_by_cluster.png"),
      ("Customer_Count","Customer Count by Cluster","customer_count_by_cluster.png")]:
        plt.figure(figsize=(8,5)); plt.bar(p.index.astype(str),p[col])
        plt.title(title); plt.xlabel("Cluster"); plt.ylabel(title.replace(" by Cluster",""))
        plt.tight_layout(); plt.savefig(OUT/file,dpi=150); plt.close()
    # Alternative feature combinations
    comparisons=[]
    for name,features in {
      "income_spending":["Annual Income","Spending Score"],
      "age_spending":["Age","Spending Score"],
      "all_three":["Age","Annual Income","Spending Score"]}.items():
        sc=StandardScaler(); z=sc.fit_transform(df[features])
        km=KMeans(n_clusters=selected_k,random_state=RANDOM_STATE,n_init=N_INIT)
        lab=km.fit_predict(z)
        comparisons.append({"Feature_Set":name,"Features":", ".join(features),
                            "Silhouette_Score":silhouette_score(z,lab),"Inertia":km.inertia_})
        if name!="income_spending":
            alt,ap=profile(df,lab,name)
            if len(features)==2:
                plt.figure(figsize=(8,5))
                for c in sorted(alt.Cluster.unique()):
                    q=alt[alt.Cluster==c]; plt.scatter(q[features[0]],q[features[1]],alpha=.65,label=f"Cluster {c}")
                plt.title(f"Alternative Feature Set: {name}"); plt.xlabel(features[0]); plt.ylabel(features[1])
                plt.legend(); plt.tight_layout(); plt.savefig(OUT/f"{name}_scatter.png",dpi=150); plt.close()
    pd.DataFrame(comparisons).to_csv(OUT/"feature_set_comparison.csv",index=False)
    # Compare K-Means with Agglomerative clustering.
    ag=AgglomerativeClustering(n_clusters=selected_k).fit_predict(X)
    pd.DataFrame([{"Method":"K-Means","Silhouette":silhouette_score(X,labels)},
                  {"Method":"Agglomerative","Silhouette":silhouette_score(X,ag)}]).to_csv(
                      OUT/"clustering_method_comparison.csv",index=False)
    # Five actual findings from computed profile.
    largest=int(p.Customer_Count.idxmax()); hi=int(p.Average_Annual_Income.idxmax())
    hs=int(p.Average_Spending_Score.idxmax()); ls=int(p.Average_Spending_Score.idxmin())
    findings=[
      f"Cluster {largest} ({p.loc[largest,'Segment_Name']}) is largest: {p.loc[largest,'Customer_Count']} customers ({p.loc[largest,'Percentage']}%).",
      f"Cluster {hi} has the highest average annual income: {p.loc[hi,'Average_Annual_Income']:.2f}.",
      f"Cluster {hs} has the highest average spending score: {p.loc[hs,'Average_Spending_Score']:.2f}.",
      f"Cluster {ls} has the lowest average spending score: {p.loc[ls,'Average_Spending_Score']:.2f}.",
      f"Average cluster ages range from {p.Average_Age.min():.1f} to {p.Average_Age.max():.1f} years."
    ]
    report=f"""COMPLETE CUSTOMER SEGMENTATION REPORT

DATA QUALITY
{json.dumps(quality,indent=2)}

SUMMARY STATISTICS
{stats.round(2).to_string()}

FEATURE SELECTION
Main features: {FEATURES}. They represent customer purchasing capacity and spending
behaviour. Customer ID is excluded because it is an arbitrary identifier. Age is used
for profiling and alternative feature comparisons.

MODEL SELECTION
K=1..10 evaluated using inertia. Silhouette scores evaluated for K=2..10.
Selected K={selected_k}, chosen by the highest silhouette score, with the elbow curve
and business usefulness to be reviewed before final adoption. This is a transparent
heuristic rather than a universal optimum.

CLUSTER PROFILE
{p.to_string()}

FIVE DATA-SUPPORTED FINDINGS
"""+"\n".join(f"{i}. {v}" for i,v in enumerate(findings,1))+"""

BUSINESS ACTIONS
"""+ "\n".join(f"\nCluster {i}: {r.Segment_Name}\n1. {r.Action_1}\n2. {r.Action_2}" for i,r in p.iterrows())+"""

LIMITATIONS
K-Means is sensitive to outliers, scaling, feature selection and K, and works best
when groups are reasonably compact around centroids. Synthetic data may not reflect
real customer behaviour. Cluster labels describe patterns; they do not prove causes
or predict future behaviour. Validate recommendations with real campaign results.

BONUS
See feature_set_comparison.csv and clustering_method_comparison.csv.
The fitted scaler and model are saved in models/customer_kmeans_pipeline.joblib.
Use assign_new_customers(new_dataframe) in main.py to assign new customers with the
same saved preprocessing and model.
"""
    (OUT/"business_insights_report.txt").write_text(report,encoding="utf-8")
    (OUT/"data_quality_report.json").write_text(json.dumps(quality,indent=2),encoding="utf-8")
    print("Data quality:",quality)
    print("\nK evaluation:\n",evaluation.round(4).to_string(index=False))
    print("\nSelected K:",selected_k)
    print("\nCluster profile:\n",p.to_string())
    print("\nInsights:\n"+"\n".join(findings))
    print("\nOutputs saved to:",OUT)

if __name__=="__main__":
    main()
