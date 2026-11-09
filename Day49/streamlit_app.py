from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
st.set_page_config(page_title="Student ML Dashboard", layout="wide")
st.title("Student Pass/Fail – ML Model Comparison")
st.caption("Fail is the positive class. Fail recall indicates the proportion of failing students identified.")

if not (OUT / "dashboard_summary.json").exists():
    st.warning("Run `python main.py` first.")
    st.stop()

summary = json.loads((OUT / "dashboard_summary.json").read_text())
st.metric("Recommended model", summary["recommended_model"])
st.write(f"Training records: {summary['train_records']} | Test records: {summary['test_records']}")
st.subheader("Held-out test metrics")
st.dataframe(pd.read_csv(OUT / "test_model_comparison.csv"), use_container_width=True)

for title, filename in [
    ("Test accuracy", "dashboard_accuracy.png"),
    ("Precision, recall and F1", "dashboard_metrics.png"),
    ("Training vs testing", "dashboard_train_test_gap.png"),
    ("Prediction time", "dashboard_prediction_time.png"),
    ("Logistic Regression confusion matrix", "logistic_regression_confusion_matrix.png"),
    ("Decision Tree confusion matrix", "decision_tree_confusion_matrix.png"),
    ("KNN confusion matrix", "knn_confusion_matrix.png")
]:
    p = OUT / filename
    if p.exists():
        st.subheader(title)
        st.image(str(p), use_container_width=True)

report = OUT / "final_results_analysis_report.txt"
if report.exists():
    st.subheader("Final report")
    st.text(report.read_text(encoding="utf-8"))
