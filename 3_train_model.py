"""
PITCHMIND — IPL PREDICTION MODEL
Step 3: Model Training
======================
Trains Random Forest + XGBoost + Ensemble model.
Saves model artifacts to models/ directory.

Run:
  python 3_train_model.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/master_features.csv"
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Columns that are identifiers / leakage — NOT used as features
EXCLUDE_COLS = {"match_id", "date", "team1", "team2", "venue", "season", "target"}


# ── 1. LOAD DATASET ────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"❌ '{DATA_PATH}' not found.\n"
            "   Run 2_feature_engineering.py first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset  →  shape: {df.shape}")
    return df


# ── 2. DEFINE FEATURES & TARGET ────────────────────────────────────────────────
def prepare_data(df):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"   Feature columns  →  {len(feature_cols)} features")
    print(f"   Features: {feature_cols}\n")

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["target"]
    return X, y, feature_cols


# ── 3. TRAIN / TEST SPLIT ──────────────────────────────────────────────────────
def split_data(X, y):
    # TIME-AWARE SPLIT — train on older matches, test on recent ones.
    # Never trains on future data to validate past. Gives honest accuracy.
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"   Train samples    →  {len(X_train)}  (2008 – 2023 seasons)")
    print(f"   Test  samples    →  {len(X_test)}   (2024 – 2025 seasons, no leakage)\n")
    return X_train, X_test, y_train, y_test


# ── 4. RANDOM FOREST ───────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test):
    print("── Random Forest ────────────────────────────────────────")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,          # shallower than before to reduce overfit
        min_samples_leaf=5,   # each leaf needs 5+ samples
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"   RF Accuracy      →  {acc:.4f}  ({acc:.2%})\n")
    return rf, acc


# ── 5. XGBOOST ─────────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test):
    print("── XGBoost ──────────────────────────────────────────────")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,          # shallower = less overfit on small dataset
        learning_rate=0.03,   # slower learning = better generalization
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,   # prevents fitting noise
        gamma=0.1,            # minimum loss reduction to split
        reg_alpha=0.1,        # L1 regularization
        reg_lambda=1.5,       # L2 regularization
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=42
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"   XGB Accuracy     →  {acc:.4f}  ({acc:.2%})\n")
    return xgb, acc


# ── 6. ENSEMBLE (averaged probabilities) ───────────────────────────────────────
def train_ensemble(rf, xgb, X_test, y_test):
    print("── Ensemble (RF + XGBoost, avg proba) ───────────────────")
    rf_prob  = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    avg_prob   = (rf_prob + xgb_prob) / 2
    final_pred = (avg_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, final_pred)
    print(f"   Ensemble Accuracy →  {acc:.4f}  ({acc:.2%})\n")
    return acc, final_pred


# ── 7. FEATURE IMPORTANCE ──────────────────────────────────────────────────────
def print_feature_importance(rf, feature_cols):
    print("── Top 10 Feature Importances (Random Forest) ───────────")
    imp = pd.Series(rf.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(10)
    for feat, val in imp.items():
        bar = "█" * int(val * 50)
        print(f"   {feat:<30}  {bar}  {val:.4f}")
    print()


# ── 8. CLASSIFICATION REPORT ───────────────────────────────────────────────────
def print_report(y_test, final_pred):
    print("── Classification Report (Ensemble) ─────────────────────")
    print(classification_report(
        y_test, final_pred,
        target_names=["Team2 Wins", "Team1 Wins"]
    ))


# ── 9. SAVE ARTIFACTS ──────────────────────────────────────────────────────────
def save_artifacts(rf, xgb, feature_cols, rf_acc, xgb_acc, ens_acc):
    # Save the XGBoost model as primary (typically best single model)
    model_path = os.path.join(MODELS_DIR, "ipl_model.pkl")
    feat_path  = os.path.join(MODELS_DIR, "feature_cols.pkl")

    joblib.dump(xgb, model_path)
    joblib.dump(feature_cols, feat_path)

    # Also save RF separately (needed for ensemble in dashboard)
    rf_path = os.path.join(MODELS_DIR, "ipl_model_rf.pkl")
    joblib.dump(rf, rf_path)

    # Save results summary
    results_path = os.path.join(MODELS_DIR, "training_results.txt")
    with open(results_path, "w") as f:
        f.write("PITCHMIND — IPL Model Training Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Random Forest Accuracy : {rf_acc:.2%}\n")
        f.write(f"XGBoost Accuracy       : {xgb_acc:.2%}\n")
        f.write(f"Ensemble Accuracy      : {ens_acc:.2%}\n")
        f.write(f"Features Used          : {len(feature_cols)}\n")

    print(f"✅ XGBoost model saved  →  {model_path}")
    print(f"✅ RF model saved       →  {rf_path}")
    print(f"✅ Feature list saved   →  {feat_path}")
    print(f"✅ Results saved        →  {results_path}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PITCHMIND — IPL MODEL TRAINING")
    print("=" * 60 + "\n")

    # Load & prepare
    df = load_data()
    X, y, feature_cols = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    rf_model,  rf_acc  = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_acc = train_xgboost(X_train, y_train, X_test, y_test)
    ens_acc, final_pred = train_ensemble(rf_model, xgb_model, X_test, y_test)

    # Reports
    print_feature_importance(rf_model, feature_cols)
    print_report(y_test, final_pred)

    # Save
    save_artifacts(rf_model, xgb_model, feature_cols, rf_acc, xgb_acc, ens_acc)

    # Final summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Random Forest Accuracy  :  {rf_acc:.2%}")
    print(f"  XGBoost Accuracy        :  {xgb_acc:.2%}")
    print(f"  Ensemble Accuracy       :  {ens_acc:.2%}")
    print("  Model saved successfully")
    print("=" * 60)
    print("\n✅ Next step: streamlit run 4_dashboard.py\n")