"""
BITS Pilani M.Tech AIML - Machine Learning Assignment 2
Streamlit App: Heart Disease Classification — 5 ML Models Comparison
Author: satyabrata.pani@gmail.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease ML Classifier",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ Heart Disease Classification — ML Model Comparison")
st.markdown(
    "**BITS Pilani M.Tech AIML | Machine Learning Assignment 2**  \n"
    "Upload test data, select a model, and view evaluation metrics."
)
st.divider()

# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Logistic Regression":    "model/logistic_regression.joblib",
    "Decision Tree":          "model/decision_tree.joblib",
    "KNN":                    "model/knn.joblib",
    "Naive Bayes":            "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "target"

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Upload + Model Selector
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Test CSV",
    type=["csv"],
    help="CSV must contain 13 feature columns + 'target' column"
)

model_choice = st.sidebar.selectbox(
    "🤖 Select ML Model",
    list(MODEL_FILES.keys())
)

run_all = st.sidebar.checkbox("📊 Compare ALL Models", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Features required:**\n"
    + ", ".join(FEATURE_COLS)
)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def load_model(model_name):
    path = MODEL_FILES[model_name]
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None
    return joblib.load(path)


def compute_metrics(model, X, y_true):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1 Score":  round(f1_score(y_true, y_pred), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }, y_pred


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig


def metrics_bar_chart(metrics_dict, model_name):
    fig, ax = plt.subplots(figsize=(6, 3))
    names = list(metrics_dict.keys())
    vals  = list(metrics_dict.values())
    bars  = ax.barh(names, vals, color="steelblue")
    ax.set_xlim(0, 1.05)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_title(f"Metrics — {model_name}")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN APP LOGIC
# ─────────────────────────────────────────────────────────────

if uploaded_file is None:
    st.info(
        "👆 Upload your **test_data.csv** in the sidebar to get started.  \n"
        "The CSV must contain the 13 feature columns and a `target` column (0 or 1)."
    )
    st.markdown("### 📋 Expected columns")
    st.code(", ".join(FEATURE_COLS + [TARGET_COL]))

    st.markdown("### 🏥 Dataset Overview")
    st.markdown(
        """
| Property | Value |
|---|---|
| **Dataset** | Heart Disease Classification (Synthetic based on UCI Cleveland) |
| **Instances** | 600 |
| **Features** | 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) |
| **Target** | Binary — 0 = No Disease, 1 = Disease |
| **Train/Test Split** | 80% / 20% |
        """
    )

else:
    # ── Load & validate CSV ──
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        st.error(f"Missing columns in uploaded CSV: {missing}")
        st.stop()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    st.success(f"✅ Loaded test data — **{len(df)} rows**, {df[TARGET_COL].sum()} positive cases")

    with st.expander("👁️ Preview uploaded data (first 5 rows)"):
        st.dataframe(df.head())

    st.divider()

    # ── COMPARE ALL MODELS ──
    if run_all:
        st.subheader("📊 All Models — Performance Comparison")

        rows = []
        for name in MODEL_FILES.keys():
            m = load_model(name)
            if m:
                met, _ = compute_metrics(m, X, y)
                rows.append({"Model": name, **met})

        compare_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(compare_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(10, 4))
        compare_df.T.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.1)
        ax.set_title("All Models — Metric Comparison")
        ax.legend(loc="lower right", fontsize=7)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        best_model = compare_df["F1 Score"].idxmax()
        st.success(f"🏆 **Best Model by F1 Score: {best_model}**  (F1 = {compare_df.loc[best_model, 'F1 Score']})")

        st.divider()

    # ── SINGLE MODEL DETAIL ──
    st.subheader(f"🔍 Detailed Results — {model_choice}")

    model = load_model(model_choice)
    if model:
        metrics, y_pred = compute_metrics(model, X, y)

        # Metric cards
        cols = st.columns(6)
        icons = ["🎯", "📈", "🎖️", "🔁", "⚖️", "📐"]
        for i, (k, v) in enumerate(metrics.items()):
            cols[i].metric(label=f"{icons[i]} {k}", value=f"{v:.4f}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_confusion_matrix(y, y_pred, model_choice))
        with col2:
            st.pyplot(metrics_bar_chart(metrics, model_choice))

        st.divider()
        st.subheader("📋 Classification Report")
        report = classification_report(y, y_pred, target_names=["No Disease", "Disease"])
        st.code(report)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "BITS Pilani WILP | M.Tech AIML | Machine Learning Assignment 2 | 2026"
    "</div>",
    unsafe_allow_html=True
)
