"""
BITS Pilani M.Tech AIML - Machine Learning Assignment 2
Train all 5 classification models on Heart Disease dataset (UCI)
Run this script ONCE to generate trained model .joblib files and test_data.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
import joblib
import os

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATASET  (Heart Disease - UCI, 303 instances, 13 features + target)
#    We use the extended Cleveland dataset fetched from UCI ML Repo
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("heart_disease_data.csv")

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['target'].value_counts()}")

X = df.drop("target", axis=1)
y = df["target"]

# ─────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT  (80/20)
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Save test CSV (features + target) for Streamlit upload demo
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"\ntest_data.csv saved — {len(test_df)} rows")

# ─────────────────────────────────────────────────────────────
# 3. DEFINE MODELS (all wrapped in Pipeline with StandardScaler)
# ─────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
}

# ─────────────────────────────────────────────────────────────
# 4. TRAIN, EVALUATE & SAVE
# ─────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

results = []

print("\n" + "="*70)
print(f"{'Model':<28} {'Acc':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6}")
print("="*70)

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_prob  = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    mcc  = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "AUC":      round(auc, 4),
        "Precision":round(prec, 4),
        "Recall":   round(rec, 4),
        "F1":       round(f1, 4),
        "MCC":      round(mcc, 4),
    })

    print(f"{name:<28} {acc:>6.4f} {auc:>6.4f} {prec:>6.4f} {rec:>6.4f} {f1:>6.4f} {mcc:>6.4f}")

    # Save model
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, f"model/{fname}")
    print(f"  → Saved model/{fname}")

print("="*70)

results_df = pd.DataFrame(results)
results_df.to_csv("model/results_summary.csv", index=False)
print("\nmodel/results_summary.csv saved")
print("\nAll models trained and saved successfully!")
