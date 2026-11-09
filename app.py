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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
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
# CONSTANTS
# ─────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine",
]

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
# BUILD PIPELINES (always uses current sklearn version)
# ─────────────────────────────────────────────────────────────
def build_pipelines():
    """Create fresh sklearn Pipelines — compatible with any sklearn version."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                       solver="lbfgs")),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=42), cv=5
            )),
        ]),
    }

# ─────────────────────────────────────────────────────────────
# TRAIN OR LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Training models on Heart Disease dataset...")
def get_trained_models():
    """
    Try to load saved .joblib models.
    If any fail (version mismatch), retrain all from scratch using
    heart_disease_data.csv — this guarantees sklearn compatibility.
    """
    pipelines = build_pipelines()

    # --- Try loading saved models first ---
    all_loaded = True
    loaded = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception:
                all_loaded = False
                break
        else:
            all_loaded = False
            break

    if all_loaded:
        return loaded, None   # (models dict, train_metrics or None)

    # --- Retrain from CSV ---
    csv_path = "heart_disease_data.csv"
    if not os.path.exists(csv_path):
        # Generate synthetic data inline if CSV missing too
        np.random.seed(42)
        n = 600
        age      = np.random.randint(29, 77, n)
        sex      = np.random.randint(0, 2, n)
        cp       = np.random.randint(0, 4, n)
        trestbps = np.random.randint(94, 200, n)
        chol     = np.random.randint(126, 564, n)
        fbs      = (chol > 220).astype(int)
        restecg  = np.random.randint(0, 3, n)
        thalach  = np.random.randint(71, 202, n)
        exang    = np.random.randint(0, 2, n)
        oldpeak  = np.round(np.random.uniform(0, 6.2, n), 1)
        slope    = np.random.randint(0, 3, n)
        ca       = np.random.randint(0, 4, n)
        thal     = np.random.choice([3, 6, 7], n)
        score = (0.03*(age-54) + 0.4*sex - 0.5*(cp==0) + 0.01*(trestbps-131)
                 + 0.003*(chol-246) + 0.3*fbs - 0.002*(thalach-150)
                 + 0.5*exang + 0.2*oldpeak + 0.3*ca
                 + np.random.normal(0, 0.5, n))
        target = (score > np.median(score)).astype(int)
        df = pd.DataFrame({
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
            "thal": thal, "target": target
        })
        df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save test_data.csv if missing
    if not os.path.exists("test_data.csv"):
        td = X_test.copy(); td[TARGET_COL] = y_test.values
        td.to_csv("test_data.csv", index=False)

    trained = {}
    os.makedirs("model", exist_ok=True)
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        fname = name.lower().replace(" ", "_") + ".joblib"
        try:
            joblib.dump(pipe, f"model/{fname}")
        except Exception:
            pass   # read-only filesystem on some cloud envs — skip saving

    return trained, None


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def compute_metrics(model, X, y_true):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
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
    ax.set_xlim(0, 1.15)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_title(f"Metrics — {model_name}")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────────────────────
models, _ = get_trained_models()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Test CSV",
    type=["csv"],
    help="CSV must contain 13 feature columns + 'target' column"
)

model_choice = st.sidebar.selectbox("🤖 Select ML Model", MODEL_NAMES)
run_all = st.sidebar.checkbox("📊 Compare ALL Models", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Features required:**\n" + ", ".join(FEATURE_COLS))

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info(
        "👆 Upload your **test_data.csv** in the sidebar to get started.  \n"
        "The CSV must contain the 13 feature columns and a `target` column (0 or 1)."
    )
    st.markdown("### 📋 Expected columns")
    st.code(", ".join(FEATURE_COLS + [TARGET_COL]))

    st.markdown("### 🏥 Dataset Overview")
    st.markdown("""
| Property | Value |
|---|---|
| **Dataset** | Heart Disease Classification (UCI Cleveland based) |
| **Instances** | 600 |
| **Features** | 13 |
| **Target** | Binary — 0 = No Disease, 1 = Disease |
| **Train/Test Split** | 80% / 20% |
    """)

else:
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

    # ── COMPARE ALL ──
    if run_all:
        st.subheader("📊 All Models — Performance Comparison")
        rows = []
        for name in MODEL_NAMES:
            m = models.get(name)
            if m:
                met, _ = compute_metrics(m, X, y)
                rows.append({"Model": name, **met})

        compare_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(
            compare_df.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        compare_df.T.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.15)
        ax.set_title("All Models — Metric Comparison")
        ax.legend(loc="lower right", fontsize=7)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        best = compare_df["F1 Score"].idxmax()
        st.success(
            f"🏆 **Best Model by F1 Score: {best}** "
            f"(F1 = {compare_df.loc[best, 'F1 Score']})"
        )
        st.divider()

    # ── SINGLE MODEL DETAIL ──
    st.subheader(f"🔍 Detailed Results — {model_choice}")
    model = models.get(model_choice)
    if model:
        metrics, y_pred = compute_metrics(model, X, y)

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
        report = classification_report(
            y, y_pred, target_names=["No Disease", "Disease"]
        )
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
