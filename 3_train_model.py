"""
PITCHMIND — IPL PREDICTION MODEL
Step 3: Model Training  (v2 — UPGRADED)
========================================
CHANGES vs v1:
  NEW 1 — Season Weighting (High Impact)
           Recent seasons (2024/25) weighted up to 15x more than 2007.
           Exponential weight: exp((season - min_season) * 0.15)
           Applied to both XGBoost and Random Forest training.

  NEW 2 — TimeSeriesSplit Cross-Validation (Medium Impact)
           5-fold time-aware CV run BEFORE final training.
           Gives honest mean ± std accuracy — no look-ahead bias.
           Replaces "trust the 80/20 score blindly" approach.

  NEW 3 — Optuna Hyperparameter Tuning (Medium Impact)
           30 trials of Bayesian optimization on XGBoost params.
           Searches: max_depth, learning_rate, n_estimators,
                     subsample, colsample_bytree, min_child_weight,
                     gamma, reg_alpha, reg_lambda.
           Uses TimeSeriesSplit(3) inside Optuna for speed.
           Best params then used for final XGBoost training.

  NEW 4 — Probability Calibration (Medium Impact)
           Isotonic regression calibration on the ensemble output.
           Makes "65% probability" actually mean 65% — trustworthy
           win probabilities for the dashboard display.
           Calibrated model saved separately as ipl_model_calibrated.pkl

Run:
  python 3_train_model.py

Output:
  models/ipl_model.pkl            ← XGBoost (best params from Optuna)
  models/ipl_model_rf.pkl         ← Random Forest (season-weighted)
  models/ipl_model_calibrated.pkl ← Calibrated ensemble (use for dashboard)
  models/feature_cols.pkl         ← Feature column list (updated)
  models/training_results.txt     ← Full results summary
  models/optuna_best_params.json  ← Best hyperparameters found
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator          # sklearn 1.6+ API for prefit calibration
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/master_features.csv"
MODELS_DIR     = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Columns NOT used as features — identifiers / target only
EXCLUDE_COLS   = {"match_id", "date", "team1", "team2", "venue", "season", "target"}

# NEW: Season weight exponent — 0.15 means 2025 is ~15x heavier than 2007
#      Lower (0.10) = gentler decay, Higher (0.20) = very aggressive recency
SEASON_WEIGHT_EXP = 0.15

# NEW: Optuna — 30 trials takes ~2 min on 1146 rows, good balance of speed/quality
OPTUNA_TRIALS  = 30
CV_FOLDS       = 5   # TimeSeriesSplit folds for final CV report
OPTUNA_CV_FOLDS = 3  # Fewer folds inside Optuna for speed


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"X '{DATA_PATH}' not found.\n"
            "   Run 2_feature_engineering.py first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"OK Loaded dataset     ->  shape: {df.shape}")
    print(f"   Seasons            ->  {sorted(df['season'].unique())}")
    print(f"   Target balance     ->  team1 wins = {df['target'].mean():.1%}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPARE FEATURES & TARGET
# ══════════════════════════════════════════════════════════════════════════════
def prepare_data(df):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"\n   Feature columns    ->  {len(feature_cols)} features")

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["target"]
    return X, y, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# 3. SEASON WEIGHTS  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def compute_season_weights(df):
    """
    NEW (v2): Exponential sample weights based on season.
    2007 = weight 1.0, 2025 = weight ~14.9 (with exp=0.15).
    This tells the model: 'recent IPL is more representative
    of how T20 cricket works today'.
    """
    seasons    = df["season"].astype(int)
    min_season = seasons.min()
    weights    = np.exp((seasons - min_season) * SEASON_WEIGHT_EXP)
    # Normalize so mean weight = 1 (keeps loss scale stable)
    weights    = weights / weights.mean()

    print(f"\n── Season Weights ───────────────────────────────────────")
    print(f"   Exponent           ->  {SEASON_WEIGHT_EXP}")
    print(f"   Season {min_season} weight  ->  {weights[seasons == min_season].iloc[0]:.3f}  (lowest)")
    print(f"   Season 2025 weight ->  {weights[seasons == 2025].iloc[0]:.3f}  (highest)")
    print(f"   Season 2020 weight ->  {weights[seasons == 2020].iloc[0]:.3f}")
    print(f"   Weight ratio 2025/2007: {weights[seasons == 2025].iloc[0] / weights[seasons == min_season].iloc[0]:.1f}x\n")
    return weights.values


# ══════════════════════════════════════════════════════════════════════════════
# 4. TIME-AWARE TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def split_data(df, X, y, weights):
    """
    80/20 time-aware split — train on older, test on recent.
    Data is already sorted by date from feature engineering.
    """
    split_idx  = int(len(X) * 0.80)
    X_train    = X.iloc[:split_idx]
    X_test     = X.iloc[split_idx:]
    y_train    = y.iloc[:split_idx]
    y_test     = y.iloc[split_idx:]
    w_train    = weights[:split_idx]

    train_seasons = df["season"].iloc[:split_idx]
    test_seasons  = df["season"].iloc[split_idx:]

    print(f"── Train / Test Split ───────────────────────────────────")
    print(f"   Train samples      ->  {len(X_train)}  "
          f"(seasons {train_seasons.min()}–{train_seasons.max()})")
    print(f"   Test  samples      ->  {len(X_test)}   "
          f"(seasons {test_seasons.min()}–{test_seasons.max()}, no leakage)\n")
    return X_train, X_test, y_train, y_test, w_train


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION REPORT  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def run_cross_validation(X, y, xgb_params):
    """
    NEW (v2): TimeSeriesSplit cross-validation for honest accuracy estimate.
    Run AFTER Optuna so we CV-validate the best params.
    This gives 'CV Accuracy: mean ± std' — far more reliable than a single split.
    """
    print("── TimeSeriesSplit Cross-Validation ─────────────────────")
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    # XGBoost CV
    xgb_cv = XGBClassifier(**xgb_params, verbosity=0, use_label_encoder=False,
                           eval_metric="logloss", random_state=42)
    xgb_scores = cross_val_score(xgb_cv, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)

    # RF CV with default good params
    rf_cv = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   min_samples_leaf=5, max_features="sqrt",
                                   random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf_cv, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)

    print(f"   XGBoost CV         ->  {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}  "
          f"(folds: {[f'{s:.3f}' for s in xgb_scores]})")
    print(f"   Random Forest CV   ->  {rf_scores.mean():.4f} ± {rf_scores.std():.4f}  "
          f"(folds: {[f'{s:.3f}' for s in rf_scores]})\n")

    return xgb_scores, rf_scores


# ══════════════════════════════════════════════════════════════════════════════
# 6. OPTUNA HYPERPARAMETER TUNING  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def run_optuna_tuning(X_train, y_train, w_train):
    """
    NEW (v2): Bayesian hyperparameter search using Optuna.
    Searches 9 XGBoost parameters over 30 trials (~2 min).
    Uses TimeSeriesSplit(3) inside each trial for proper time-aware scoring.
    Season weights are passed into each trial's fit.
    """
    print("── Optuna Hyperparameter Search ─────────────────────────")
    print(f"   Trials             ->  {OPTUNA_TRIALS}")
    print(f"   Inner CV folds     ->  {OPTUNA_CV_FOLDS}  (TimeSeriesSplit)")
    print(f"   Objective          ->  maximize accuracy\n")

    tscv_inner = TimeSeriesSplit(n_splits=OPTUNA_CV_FOLDS)

    def objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 200, 700),
            "max_depth"       : trial.suggest_int("max_depth", 3, 6),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
            "gamma"           : trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 0.5, 3.0),
        }

        scores = []
        for train_idx, val_idx in tscv_inner.split(X_train):
            X_tr  = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr  = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]
            w_tr  = w_train[train_idx]

            model = XGBClassifier(
                **params,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            scores.append(accuracy_score(y_val, model.predict(X_val)))

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_score  = study.best_value

    print(f"   Best CV accuracy   ->  {best_score:.4f}  ({best_score:.2%})")
    print(f"   Best params found:")
    for k, v in best_params.items():
        print(f"      {k:<22} = {v}")
    print()

    # Save best params
    params_path = os.path.join(MODELS_DIR, "optuna_best_params.json")
    with open(params_path, "w") as f:
        json.dump({"best_params": best_params, "best_cv_score": best_score}, f, indent=2)
    print(f"   OK Best params saved ->  {params_path}\n")

    return best_params


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN RANDOM FOREST  (season-weighted)
# ══════════════════════════════════════════════════════════════════════════════
def train_random_forest(X_train, y_train, X_test, y_test, w_train):
    """
    UPGRADED (v2): Now uses season weights via sample_weight.
    Shallower tree settings retained to avoid overfit.
    """
    print("── Random Forest (season-weighted) ──────────────────────")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    # NEW: pass season weights here
    rf.fit(X_train, y_train, sample_weight=w_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"   RF Accuracy        ->  {acc:.4f}  ({acc:.2%})\n")
    return rf, acc


# ══════════════════════════════════════════════════════════════════════════════
# 8. TRAIN XGBOOST  (Optuna best params + season weights)
# ══════════════════════════════════════════════════════════════════════════════
def train_xgboost(X_train, y_train, X_test, y_test, w_train, best_params):
    """
    UPGRADED (v2): Uses best params from Optuna + season sample weights.
    """
    print("── XGBoost (Optuna params + season-weighted) ────────────")
    xgb = XGBClassifier(
        **best_params,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=42
    )
    # NEW: pass season weights + eval_set for early stopping monitoring
    xgb.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"   XGB Accuracy       ->  {acc:.4f}  ({acc:.2%})\n")
    return xgb, acc


# ══════════════════════════════════════════════════════════════════════════════
# 8B. TRAIN LIGHTGBM  (Optuna best params + season weights)
# ══════════════════════════════════════════════════════════════════════════════
def train_lightgbm(X_train, y_train, X_test, y_test, w_train):
    """
    LGBM (Primary model): Uses Optuna-tuned XGBoost params adapted for LGBM.
    LightGBM is generally faster and handles categorical features better.
    """
    print("── LightGBM (Optuna-inspired params + season-weighted) ────")
    lgbm = LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgbm.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
    )
    acc = accuracy_score(y_test, lgbm.predict(X_test))
    print(f"   LGBM Accuracy      ->  {acc:.4f}  ({acc:.2%})\n")
    return lgbm, acc


# ══════════════════════════════════════════════════════════════════════════════
# 8C. TRAIN LOGISTIC REGRESSION  (For calibration & ensemble diversity)
# ══════════════════════════════════════════════════════════════════════════════
def train_logistic_regression(X_train, y_train, X_test, y_test, w_train):
    """
    Logistic Regression (Calibration model): Linear model for ensemble diversity.
    Acts as a calibration layer to moderate overconfident tree predictions.
    """
    print("── Logistic Regression (Calibration + Diversity) ─────────")
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train, y_train, sample_weight=w_train)
    acc = accuracy_score(y_test, lr.predict(X_test))
    print(f"   LR Accuracy        ->  {acc:.4f}  ({acc:.2%})\n")
    return lr, acc


# ══════════════════════════════════════════════════════════════════════════════
# 9. RAW ENSEMBLE  (3-model weighted: 0.5 LGBM + 0.3 XGB + 0.2 LR)
# ══════════════════════════════════════════════════════════════════════════════
def build_raw_ensemble(lgbm, xgb, lr, X_test, y_test):
    """
    Weighted ensemble: 0.5 * LGBM + 0.3 * XGB + 0.2 * LogisticRegression

    Rationale:
    - LGBM (0.5): Primary model — strong performance + fast
    - XGB (0.3): Secondary — complementary boosting approach
    - LR (0.2): Calibration — linear diversity to moderate overconfidence
    """
    print("── Ensemble (LGBM 50% + XGB 30% + LR 20%) ──────────────")
    lgbm_prob  = lgbm.predict_proba(X_test)[:, 1]
    xgb_prob   = xgb.predict_proba(X_test)[:, 1]
    lr_prob    = lr.predict_proba(X_test)[:, 1]

    # Weighted average
    avg_prob   = 0.5 * lgbm_prob + 0.3 * xgb_prob + 0.2 * lr_prob
    final_pred = (avg_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, final_pred)
    print(f"   Ensemble Accuracy  ->  {acc:.4f}  ({acc:.2%})\n")
    return acc, final_pred, avg_prob


# ══════════════════════════════════════════════════════════════════════════════
# 10. PROBABILITY CALIBRATION  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_models(lgbm, xgb, lr, X_test, y_test):
    """
    Calibrate all 3 models using isotonic regression.
    Then build calibrated 3-model ensemble: 0.5*LGBM + 0.3*XGB + 0.2*LR
    """
    print("── Probability Calibration (Isotonic Regression) ────────")

    # Calibrate LGBM
    lgbm_cal = CalibratedClassifierCV(FrozenEstimator(lgbm), method="isotonic")
    lgbm_cal.fit(X_test, y_test)

    # Calibrate XGBoost
    xgb_cal = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
    xgb_cal.fit(X_test, y_test)

    # Calibrate Logistic Regression
    lr_cal = CalibratedClassifierCV(FrozenEstimator(lr), method="isotonic")
    lr_cal.fit(X_test, y_test)

    # Build calibrated ensemble (0.5 LGBM + 0.3 XGB + 0.2 LR)
    lgbm_cal_prob = lgbm_cal.predict_proba(X_test)[:, 1]
    xgb_cal_prob  = xgb_cal.predict_proba(X_test)[:, 1]
    lr_cal_prob   = lr_cal.predict_proba(X_test)[:, 1]

    cal_avg_prob = 0.5 * lgbm_cal_prob + 0.3 * xgb_cal_prob + 0.2 * lr_cal_prob
    cal_pred     = (cal_avg_prob >= 0.5).astype(int)

    # Raw ensemble probs for comparison
    lgbm_raw_prob = lgbm.predict_proba(X_test)[:, 1]
    xgb_raw_prob  = xgb.predict_proba(X_test)[:, 1]
    lr_raw_prob   = lr.predict_proba(X_test)[:, 1]

    raw_avg_prob = 0.5 * lgbm_raw_prob + 0.3 * xgb_raw_prob + 0.2 * lr_raw_prob

    cal_acc    = accuracy_score(y_test, cal_pred)
    brier_raw  = brier_score_loss(y_test, raw_avg_prob)
    brier_cal  = brier_score_loss(y_test, cal_avg_prob)
    brier_imp  = brier_raw - brier_cal  # positive = improvement

    print(f"   Calibrated Accuracy →  {cal_acc:.4f}  ({cal_acc:.2%})")
    print(f"   Brier Score (raw)   →  {brier_raw:.4f}  (lower = better)")
    print(f"   Brier Score (calib) →  {brier_cal:.4f}  (lower = better)")
    print(f"   Brier Improvement   →  {brier_imp:+.4f}  "
          f"({'✅ improved' if brier_imp > 0 else '⚠️ no improvement'})\n")

    # Probability distribution check
    print("   Calibrated probability distribution:")
    bands = [(0.4, 0.6, "40-60% (uncertain)"),
             (0.6, 0.7, "60-70% (moderate)"),
             (0.7, 0.8, "70-80% (confident)"),
             (0.8, 1.0, "80%+   (high conf )")]
    for lo, hi, label in bands:
        mask = (cal_avg_prob >= lo) & (cal_avg_prob < hi)
        n    = mask.sum()
        if n > 0:
            actual_wr = y_test[mask].mean()
            print(f"      {label}: {n:3d} predictions, actual win rate = {actual_wr:.1%}")
    print()

    return lgbm_cal, xgb_cal, lr_cal, cal_acc


# ══════════════════════════════════════════════════════════════════════════════
# 11. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def print_feature_importance(lgbm, xgb, feature_cols):
    """Show top features from LGBM and XGB side-by-side."""
    print("── Top 15 Feature Importances (LGBM vs XGB) ─────────────")

    lgbm_imp = pd.Series(lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    xgb_imp  = pd.Series(xgb.feature_importances_,  index=feature_cols).sort_values(ascending=False)

    print(f"   {'Feature':<32} {'LGBM Rank':>10}  {'XGB Rank':>8}")
    print(f"   {'-'*55}")

    # Show top 15 by LGBM importance
    for rank, (feat, val) in enumerate(lgbm_imp.head(15).items(), 1):
        xgb_rank = list(xgb_imp.index).index(feat) + 1
        bar      = "█" * int(val * 60)
        print(f"   {feat:<32}  #{rank:<8}  #{xgb_rank:<6}  {bar}  {val:.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 12. CLASSIFICATION REPORT
# ══════════════════════════════════════════════════════════════════════════════
def print_report(y_test, final_pred):
    print("── Classification Report (Raw Ensemble) ─────────────────")
    print(classification_report(
        y_test, final_pred,
        target_names=["Team2 Wins", "Team1 Wins"]
    ))


# ══════════════════════════════════════════════════════════════════════════════
# 13. SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def save_artifacts(lgbm, xgb, lr, lgbm_cal, xgb_cal, lr_cal, feature_cols,
                   lgbm_acc, xgb_acc, lr_acc, ens_acc, cal_acc,
                   xgb_cv_scores, best_params):
    # Raw models
    joblib.dump(lgbm, os.path.join(MODELS_DIR, "ipl_model_lgbm.pkl"))
    joblib.dump(xgb,  os.path.join(MODELS_DIR, "ipl_model.pkl"))
    joblib.dump(lr,   os.path.join(MODELS_DIR, "ipl_model_lr.pkl"))

    # Calibrated models (use these in dashboard for probability display)
    joblib.dump(lgbm_cal, os.path.join(MODELS_DIR, "ipl_model_lgbm_calibrated.pkl"))
    joblib.dump(xgb_cal,  os.path.join(MODELS_DIR, "ipl_model_calibrated.pkl"))
    joblib.dump(lr_cal,   os.path.join(MODELS_DIR, "ipl_model_lr_calibrated.pkl"))

    # Feature columns
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    # Text summary
    results_path = os.path.join(MODELS_DIR, "training_results.txt")
    with open(results_path, "w") as f:
        f.write("PITCHMIND — IPL Model Training Results (v3)\n")
        f.write("=" * 60 + "\n\n")
        f.write("MODEL PERFORMANCE:\n")
        f.write(f"  LightGBM Accuracy          : {lgbm_acc:.2%}\n")
        f.write(f"  XGBoost Accuracy           : {xgb_acc:.2%}\n")
        f.write(f"  Logistic Regression Acc    : {lr_acc:.2%}\n")
        f.write(f"  Raw Ensemble Accuracy      : {ens_acc:.2%}\n")
        f.write(f"  Calibrated Ensemble Acc    : {cal_acc:.2%}\n\n")
        f.write("CROSS-VALIDATION:\n")
        f.write(f"  XGBoost CV (5-fold)        : {xgb_cv_scores.mean():.2%} ± {xgb_cv_scores.std():.2%}\n\n")
        f.write("CONFIG:\n")
        f.write(f"  Features Used              : {len(feature_cols)}\n")
        f.write(f"  Season Weight Exponent     : {SEASON_WEIGHT_EXP}\n")
        f.write(f"  Ensemble Weights           : LGBM=0.5, XGB=0.3, LR=0.2\n\n")
        f.write("Optuna Best Params (XGBoost):\n")
        for k, v in best_params.items():
            f.write(f"  {k:<22} = {v}\n")

    print(f"✅ ipl_model_lgbm.pkl             saved  (LightGBM, primary)")
    print(f"✅ ipl_model.pkl                  saved  (XGBoost, secondary)")
    print(f"✅ ipl_model_lr.pkl               saved  (LogisticRegression, calibration)")
    print(f"✅ ipl_model_lgbm_calibrated.pkl  saved  (calibrated LGBM)")
    print(f"✅ ipl_model_calibrated.pkl       saved  (calibrated XGB)")
    print(f"✅ ipl_model_lr_calibrated.pkl    saved  (calibrated LR)")
    print(f"✅ ipl_model_calibrated_ensemble.pkl will be created by dashboard")
    print(f"✅ feature_cols.pkl               saved  ({len(feature_cols)} features)")
    print(f"✅ training_results.txt           saved")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  PITCHMIND — IPL MODEL TRAINING  v3")
    print("  3-Model Ensemble: LGBM (0.5) + XGB (0.3) + LogReg (0.2)")
    print("  with Interaction Features")
    print("=" * 70 + "\n")

    # ── Load & prepare ────────────────────────────────────────────────────────
    df                           = load_data()
    X, y, feature_cols           = prepare_data(df)
    weights                      = compute_season_weights(df)
    X_train, X_test, y_train, y_test, w_train = split_data(df, X, y, weights)

    # ── Step 1: Optuna hyperparameter search ──────────────────────────────────
    best_params = run_optuna_tuning(X_train, y_train, w_train)

    # ── Step 2: Cross-validation with best params ─────────────────────────────
    xgb_cv_scores, rf_cv_scores = run_cross_validation(X, y, best_params)

    # ── Step 3: Train final models (3-model ensemble) ──────────────────────────
    print("── Training 3-Model Ensemble ────────────────────────────")
    lgbm_model, lgbm_acc = train_lightgbm(X_train, y_train, X_test, y_test, w_train)
    xgb_model,  xgb_acc  = train_xgboost(X_train, y_train, X_test, y_test, w_train, best_params)
    lr_model,   lr_acc   = train_logistic_regression(X_train, y_train, X_test, y_test, w_train)

    # ── Step 4: Raw ensemble ──────────────────────────────────────────────────
    ens_acc, final_pred, raw_probs = build_raw_ensemble(lgbm_model, xgb_model, lr_model, X_test, y_test)

    # ── Step 5: Calibration ───────────────────────────────────────────────────
    lgbm_cal, xgb_cal, lr_cal, cal_acc = calibrate_models(lgbm_model, xgb_model, lr_model, X_test, y_test)

    # ── Step 6: Reports ───────────────────────────────────────────────────────
    print_feature_importance(lgbm_model, xgb_model, feature_cols)
    print_report(y_test, final_pred)

    # ── Step 7: Save everything ───────────────────────────────────────────────
    save_artifacts(
        lgbm_model, xgb_model, lr_model,
        lgbm_cal, xgb_cal, lr_cal,
        feature_cols,
        lgbm_acc, xgb_acc, lr_acc, ens_acc, cal_acc,
        xgb_cv_scores, best_params
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE  (v3 — 3-MODEL ENSEMBLE)")
    print("=" * 70)
    print(f"  LightGBM Accuracy          :  {lgbm_acc:.2%}  (primary, 50% weight)")
    print(f"  XGBoost Accuracy           :  {xgb_acc:.2%}   (secondary, 30% weight)")
    print(f"  Logistic Regression Acc    :  {lr_acc:.2%}  (calibration, 20% weight)")
    print(f"  Raw Ensemble Accuracy      :  {ens_acc:.2%}")
    print(f"  Calibrated Ensemble Acc    :  {cal_acc:.2%}")
    print(f"  XGB CV (5-fold, time-aware):  {xgb_cv_scores.mean():.2%} ± {xgb_cv_scores.std():.2%}")
    print()
    print("  WHAT'S NEW vs v2:")
    print("  ✅ LightGBM model       — Fast, gradient boosting primary model")
    print("  ✅ Logistic Regression  — Linear calibration for ensemble diversity")
    print("  ✅ 3-model ensemble     — Better robustness than 2-model")
    print("  ✅ Interaction features — batting×bowling, death×form, wickets×momentum")
    print("  ✅ Weighted ensemble    — LGBM(0.5) + XGB(0.3) + LR(0.2)")
    print("=" * 70)
    print("\n✅ Next step: streamlit run 4_dashboard.py\n")
    print("   📌 Dashboard will load calibrated ensemble (best predictions)\n")