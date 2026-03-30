"""
TEST FEATURE ALIGNMENT
======================
Verify all features are consistent across training, dashboard, and phase prediction.
"""

import os
import joblib
import pandas as pd
import numpy as np

def test_main_model_features():
    """Check main prediction model features."""
    print("\n" + "="*70)
    print("TEST 1: MAIN MODEL FEATURES")
    print("="*70)

    # Load feature_cols from training
    feat_path = os.path.join("models", "feature_cols.pkl")
    if not os.path.exists(feat_path):
        print("[FAIL] feature_cols.pkl not found")
        return False

    feature_cols = joblib.load(feat_path)
    print("[OK] Loaded {} features from training".format(len(feature_cols)))

    # Check for critical new features
    required_features = [
        "season_recency", "season_avg_runs",
        "team1_form_velocity", "team2_form_velocity", "diff_form_velocity",
        "team1_relative_score", "team2_relative_score", "diff_relative_score",
        "team1_relative_rr", "team2_relative_rr"
    ]

    missing = [f for f in required_features if f not in feature_cols]
    if missing:
        print("[FAIL] Missing features in model: {}".format(missing))
        return False
    print("[OK] All {} new v5 features present".format(len(required_features)))

    print("\nFeature list ({} total):".format(len(feature_cols)))
    for i, feat in enumerate(feature_cols, 1):
        print("   {:2d}. {}".format(i, feat))

    return True


def test_dashboard_build_vector():
    """Check dashboard build_feature_vector creates all features."""
    print("\n" + "="*70)
    print("TEST 2: DASHBOARD build_feature_vector")
    print("="*70)

    # Check the code for critical feature names in build_feature_vector
    critical_names = [
        "season_recency",
        "season_avg_runs",
        "team1_form_velocity",
        "team2_form_velocity",
        "team1_relative_score",
        "team2_relative_score",
        "team1_relative_rr",
        "team2_relative_rr",
    ]

    with open("4_dashboard.py", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    missing = [name for name in critical_names if name not in code]
    if missing:
        print("[FAIL] Missing feature references in dashboard: {}".format(missing))
        return False

    print("[OK] All {} new features referenced in dashboard".format(len(critical_names)))
    return True


def test_phase_model_features():
    """Check phase model features."""
    print("\n" + "="*70)
    print("TEST 3: PHASE MODEL FEATURES")
    print("="*70)

    feat_path = os.path.join("models", "phase_feature_cols.pkl")
    if not os.path.exists(feat_path):
        print("[INFO] phase_feature_cols.pkl not found (phase models may not be trained)")
        return True  # Not critical

    phase_features = joblib.load(feat_path)
    print("[OK] Loaded {} phase features".format(len(phase_features)))

    # Check for season_recency
    if "season_recency" not in phase_features:
        print("[FAIL] season_recency missing from phase features: {}".format(phase_features))
        return False

    print("[OK] season_recency present in phase features")
    print("\nPhase feature list ({} total):".format(len(phase_features)))
    for i, feat in enumerate(phase_features, 1):
        print("   {:2d}. {}".format(i, feat))

    return True


def test_master_features_csv():
    """Check master_features.csv has all expected columns."""
    print("\n" + "="*70)
    print("TEST 4: MASTER_FEATURES.CSV COLUMNS")
    print("="*70)

    csv_path = os.path.join("data", "master_features.csv")
    if not os.path.exists(csv_path):
        print("[INFO] master_features.csv not found (not critical)")
        return True

    df = pd.read_csv(csv_path, nrows=1)
    print("[OK] Loaded master_features.csv with {} columns".format(len(df.columns)))

    # Check for critical columns
    required_cols = [
        "season_recency", "season_avg_runs",
        "team1_form_velocity", "team2_form_velocity", "diff_form_velocity",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("[FAIL] Missing columns in CSV: {}".format(missing))
        print("   Available: {}".format(list(df.columns)))
        return False

    print("[OK] All {} v5 feature columns present in CSV".format(len(required_cols)))
    return True


def test_training_results():
    """Check training results file."""
    print("\n" + "="*70)
    print("TEST 5: TRAINING RESULTS")
    print("="*70)

    results_path = os.path.join("models", "training_results.txt")
    if not os.path.exists(results_path):
        print("[INFO] training_results.txt not found")
        return True

    with open(results_path) as f:
        content = f.read()

    print(content)
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   PITCHMIND FEATURE ALIGNMENT VALIDATION")
    print("=" * 70)

    tests = [
        ("Main Model Features", test_main_model_features),
        ("Dashboard Build Vector", test_dashboard_build_vector),
        ("Phase Model Features", test_phase_model_features),
        ("Master Features CSV", test_master_features_csv),
        ("Training Results", test_training_results),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print("\n[ERROR] in {}: {}".format(name, e))
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print("{:6s}  {}".format(status, name))

    all_pass = all(r for _, r in results)
    print("\n" + "="*70)
    if all_pass:
        print("[OK] ALL TESTS PASSED -- System is fully aligned!")
    else:
        print("[FAILED] SOME TESTS FAILED -- See above for issues")
    print("=" * 70 + "\n")

    exit(0 if all_pass else 1)
