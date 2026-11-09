# Heart Disease Classification — ML Assignment 2

**BITS Pilani | Work Integrated Learning Programmes**  
**M.Tech (AIML) | Machine Learning | AIML ZG565**  
**Student: satyabrata.pani@gmail.com**

---

## a. Problem Statement

Predict whether a patient has heart disease (binary classification: 0 = No Disease, 1 = Disease) based on 13 clinical and demographic features. The goal is to compare five classical machine learning classification algorithms on the same dataset using standard evaluation metrics.

---

## b. Dataset Description

| Property | Value |
|---|---|
| **Dataset Name** | Heart Disease Classification Dataset |
| **Source** | Based on UCI Cleveland Heart Disease Dataset (synthetic version) |
| **Total Instances** | 600 |
| **Number of Features** | 13 |
| **Target Variable** | `target` — Binary (0: No Disease, 1: Disease) |
| **Class Distribution** | Balanced — 300 each class |
| **Train/Test Split** | 80% train (480 rows) / 20% test (120 rows) |

### Feature Description

| Feature | Description | Type |
|---|---|---|
| age | Age of patient (years) | Numeric |
| sex | Gender (1=Male, 0=Female) | Binary |
| cp | Chest pain type (0–3) | Nominal |
| trestbps | Resting blood pressure (mmHg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl (1=True) | Binary |
| restecg | Resting ECG results (0–2) | Nominal |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise induced angina (1=Yes) | Binary |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment (0–2) | Ordinal |
| ca | Number of major vessels (0–3) | Numeric |
| thal | Thalassemia type (3, 6, 7) | Nominal |

---

## c. GitHub Repository Link

> **[https://github.com/YOUR_USERNAME/ml-assignment-2-bits](https://github.com/YOUR_USERNAME/ml-assignment-2-bits)**
>
> *(Replace YOUR_USERNAME with your actual GitHub username after uploading)*

### Repository Structure

```
ml-assignment-2-bits/
├── app.py                      # Streamlit web application
├── train_models.py             # Script to train all 5 models
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── heart_disease_data.csv      # Full dataset (600 rows)
├── test_data.csv               # Test split (120 rows) — upload to Streamlit
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

---

## d. Models Used — Evaluation Metrics

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.8167 | 0.9072 | 0.8065 | 0.8333 | 0.8197 | 0.6337 |
| **Decision Tree** | 0.7000 | 0.7653 | 0.6935 | 0.7167 | 0.7049 | 0.4002 |
| **KNN** | 0.7667 | 0.8207 | 0.7500 | 0.8000 | 0.7742 | 0.5345 |
| **Naive Bayes** | 0.7583 | 0.8867 | 0.7123 | 0.8667 | 0.7820 | 0.5292 |
| **Support Vector Machine** | 0.7833 | 0.8858 | 0.7656 | 0.8167 | 0.7903 | 0.5679 |

### Model Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Best overall performer with highest Accuracy (81.67%), AUC (0.9072), F1 (0.8197) and MCC (0.6337). The dataset's features have near-linear separability which suits Logistic Regression well. The high AUC indicates excellent discriminative ability between disease and non-disease cases. |
| **Decision Tree** | Lowest performance across all metrics (Accuracy: 70%, AUC: 0.7653). Despite max_depth=5 to prevent overfitting, the model struggles to generalize. Decision Trees are sensitive to the specific splits chosen and may not capture subtle patterns as well as probabilistic or distance-based models. |
| **KNN** | Moderate performance (Accuracy: 76.67%, AUC: 0.8207). KNN benefits from StandardScaler normalization. It performs reasonably well on this balanced dataset. The main limitation is prediction-time complexity O(N×D) but with 480 training instances this is acceptable. |
| **Naive Bayes** | Good AUC (0.8867) despite lower accuracy (75.83%). The high AUC suggests Naive Bayes produces well-calibrated probability estimates even though the conditional independence assumption doesn't perfectly hold. Good for ranking instances by disease risk. |
| **Support Vector Machine** | Second best performer (Accuracy: 78.33%, AUC: 0.8858, F1: 0.7903). RBF kernel effectively captures non-linear boundaries. Its strong MCC (0.5679) confirms balanced performance across both classes. Competitive with Logistic Regression especially in Precision. |
| **Overall Winner** | **Logistic Regression** — achieves the best scores across Accuracy, AUC, F1, and MCC. For this clinical dataset where features are largely continuous and exhibit near-linear relationships with the target, Logistic Regression is the most reliable and interpretable model. |

---

## Streamlit App Link

> **[https://YOUR-APP-NAME.streamlit.app](https://YOUR-APP-NAME.streamlit.app)**
>
> *(Replace with your actual deployed app URL)*

### App Features
- CSV test data upload
- Model selection dropdown (5 models)
- Compare ALL models simultaneously
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix visualization
- Classification report

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-assignment-2-bits.git
cd ml-assignment-2-bits

# Install dependencies
pip install -r requirements.txt

# Train models (generates .joblib files)
python train_models.py

# Run Streamlit app
streamlit run app.py
```
<img width="936" height="521" alt="image" src="https://github.com/user-attachments/assets/f5e99bce-1e92-4ddd-b197-7aac3714fa91" />

---

## Academic Integrity

This submission is original work completed as part of BITS Pilani M.Tech AIML coursework.  
AI tools were used for learning support only.
