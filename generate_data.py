"""
Generate synthetic Heart Disease-like dataset with 14 features (13 + target)
600 instances — meets assignment min requirements (12 features, 500 instances)
This is used as the training dataset.
"""
import numpy as np
import pandas as pd

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

# Generate a realistic target correlated with features
score = (
    0.03*(age - 54)
    + 0.4*(sex)
    - 0.5*(cp == 0)
    + 0.01*(trestbps - 131)
    + 0.003*(chol - 246)
    + 0.3*(fbs)
    - 0.002*(thalach - 150)
    + 0.5*(exang)
    + 0.2*(oldpeak)
    + 0.3*(ca)
    + np.random.normal(0, 0.5, n)
)
# Balanced target ~50/50
target = (score > np.median(score)).astype(int)

df = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
    "thal": thal, "target": target
})

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['target'].value_counts()}")
df.to_csv("heart_disease_data.csv", index=False)
print("heart_disease_data.csv saved!")
