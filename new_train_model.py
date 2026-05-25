"""
╔══════════════════════════════════════════════════════════════════╗
║         DIABETES MODEL TRAINER (SVM)                            ║
║         Run this ONCE to create final_diabetes_model.pkl         ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    python new_train_model.py

It will create:
    final_diabetes_model.pkl
    final_diabetes_features.json
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.combine import SMOTEENN
import joblib

# ── 1. Load data ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(BASE, "diabetes_data_upload.csv")

if not os.path.exists(CSV):
    raise FileNotFoundError(
        f"Cannot find: {CSV}\n"
        f"Make sure diabetes_data_upload.csv is in the same folder."
    )

df = pd.read_csv(CSV)
df.columns = df.columns.str.lower()
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Class distribution:\n{df['class'].value_counts().to_string()}\n")

# ── 2. Drop Obesity ───────────────────────────────────────────────
if "obesity" in df.columns:
    df = df.drop(columns=["obesity"])

# ── 3. Manual encoding (same as app will use) ─────────────────────
# Gender: Male=1, Female=0
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# Yes/No columns: Yes=1, No=0
yes_no_cols = [c for c in df.columns if c not in ["age", "gender", "class"]]
for col in yes_no_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Target: Positive=1, Negative=0
df["class"] = df["class"].map({"Positive": 1, "Negative": 0})

# ── 4. Features & Target ─────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c != "class"]
X = df[FEATURE_COLS]
Y = df["class"]

print(f"✅ Features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")

# ── 5. Train/test split ───────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, stratify=Y
)

# ── 6. Pipeline: MinMaxScaler + SMOTEENN + SVM ───────────────────
final_pipeline = imbpipeline(steps=[
    ("scaler",    MinMaxScaler()),
    ("smote_enn", SMOTEENN(random_state=42)),
    ("SVM_model", SVC(random_state=42, probability=True))
])

param_grid = [{
    "SVM_model__C":      [0.2, 0.4, 0.6, 0.8, 1.0],
    "SVM_model__kernel": ["linear", "rbf"],
    "SVM_model__gamma":  [1, 0.1, 0.01, 0.001, 0.0001]
}]

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("⏳ Running GridSearchCV (this may take a few minutes)...")
grid_search = GridSearchCV(
    final_pipeline,
    param_grid,
    cv=kf,
    scoring="recall_macro",
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, Y_train)

model      = grid_search.best_estimator_
cv_score   = grid_search.best_score_
test_score = grid_search.score(X_test, Y_test)

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
print(f"📊 Cross-validation score : {round(cv_score*100, 2)}%")
print(f"📊 Test score             : {round(test_score*100, 2)}%")

# ── 7. Classification report ─────────────────────────────────────
Y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, target_names=["Non-Diabetic", "Diabetic"]))

# ── 8. Sanity check ───────────────────────────────────────────────
# All No patient: age=30, gender=Male(1), all symptoms=No(0)
test_no  = pd.DataFrame([[30, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                         columns=FEATURE_COLS)
# All Yes patient: age=55, gender=Male(1), all symptoms=Yes(1)
test_yes = pd.DataFrame([[55, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                         columns=FEATURE_COLS)

prob_no  = model.predict_proba(test_no)[0][1]
prob_yes = model.predict_proba(test_yes)[0][1]

print(f"\n🔬 Sanity Check:")
print(f"   All-No  patient → Diabetes probability: {prob_no*100:.1f}%  (should be LOW)")
print(f"   All-Yes patient → Diabetes probability: {prob_yes*100:.1f}% (should be HIGH)")

# ── 9. Save ───────────────────────────────────────────────────────
PKL  = os.path.join(BASE, "final_diabetes_model.pkl")
FEAT = os.path.join(BASE, "final_diabetes_features.json")

joblib.dump(model, PKL)
with open(FEAT, "w") as f:
    json.dump(FEATURE_COLS, f)

print(f"\n✅ Saved: final_diabetes_model.pkl")
print(f"✅ Saved: final_diabetes_features.json")
print("\n🎉 Training complete! Now run: python new_diabetes_app.py")
