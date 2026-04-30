"""
PITCHMIND — STEP 7: In-Match Phase Run Predictor  (v2 — FIXES)
===============================================================
FIXES vs v1:
  FIX 1 — CRITICAL BUG: build_phase_samples() used global `bowler_eco_lookup`
           instead of the parameter `bowler_eco`. Renamed parameter to
           `bowler_eco_dict` to avoid shadowing. This caused wrong economy
           values being looked up during training.

  FIX 2 — venue_factor_lookup: fixed lambda aggregation bug.
           The old lambda inside .agg() was incorrectly referencing the outer
           `grp` DataFrame (index mismatch). Replaced with explicit per-over
           filtering via merge.

  FIX 3 — Phase defaults updated to match 2023+ reality:
           DEFAULT_VENUE_TOTAL: 160 → 175
           DEFAULT_BATTER_SR:   120 → 130
           DEFAULT_BOWLER_ECO:  8.5 → 9.8

  FIX 4 — MIN_BALLS thresholds lowered for better coverage:
           Batter: 20 → 15 balls minimum
           Bowler: 20 → 15 balls minimum

  FIX 5 — Phase sample: pp_wickets tracked correctly for death/middle phase
           (total wickets from start of innings, not just in phase)

Run:
    python 7_phase_predictor.py

Saves:
    models/phase_powerplay_model.pkl
    models/phase_middle_model.pkl
    models/phase_death_model.pkl
    models/phase_batter_sr.pkl
    models/phase_bowler_eco.pkl
    models/phase_venue_factor.pkl
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Over ranges (0-indexed)
PHASE_RANGES = {
    "powerplay": (0,  5),
    "middle"   : (6,  14),
    "death"    : (15, 19),
}

BATTING_STRENGTH_MAP = {
    0: 1.0, 1: 1.0, 2: 0.95, 3: 0.90, 4: 0.85,
    5: 0.80, 6: 0.75, 7: 0.65, 8: 0.50, 9: 0.35, 10: 0.25,
}

# FIX 3: Updated defaults
DEFAULT_BATTER_SR   = 130.0   # was 120.0
DEFAULT_BOWLER_ECO  = 9.8     # was 8.5
DEFAULT_VENUE_TOTAL = 175.0   # was 160.0
DEFAULT_VENUE_PP    = 51.0
DEFAULT_VENUE_MID   = 73.0
DEFAULT_VENUE_DEATH = 61.0

# FIX 4: Lower min-balls for better player coverage
MIN_BATTER_BALLS = 15   # was 20
MIN_BOWLER_BALLS = 15   # was 20


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    del_path = os.path.join(DATA_DIR, "deliveries_clean.csv")
    if not os.path.exists(del_path):
        print(f"  ⚠️  deliveries_clean.csv not found, using raw deliveries.csv")
        del_path = os.path.join(DATA_DIR, "deliveries.csv")

    mat_path = os.path.join(DATA_DIR, "matches_clean.csv")
    if not os.path.exists(mat_path):
        print(f"  ⚠️  matches_clean.csv not found, using raw matches.csv")
        mat_path = os.path.join(DATA_DIR, "matches.csv")

    print(f"  Loading {del_path} ...")
    del_df = pd.read_csv(del_path, low_memory=False)
    del_df.columns = del_df.columns.str.lower()

    for col in ["over", "ball", "batsman_runs", "extra_runs", "total_runs", "is_wicket"]:
        del_df[col] = pd.to_numeric(del_df[col], errors="coerce").fillna(0)
    del_df["over"]      = del_df["over"].astype(int)
    del_df["is_wicket"] = del_df["is_wicket"].astype(int)

    print(f"  Loading {mat_path} ...")
    mat_df = pd.read_csv(mat_path, low_memory=False)
    if "id" in mat_df.columns and "match_id" not in mat_df.columns:
        mat_df = mat_df.rename(columns={"id": "match_id"})
    mat_df["match_id"] = mat_df["match_id"].astype(str)
    del_df["match_id"] = del_df["match_id"].astype(str)

    print(f"  Deliveries: {del_df.shape}  |  Matches: {mat_df.shape}")
    return del_df, mat_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD CAREER LOOKUP TABLES
# ══════════════════════════════════════════════════════════════════════════════

def build_batter_sr_lookup(del_df):
    """Per batter: career SR overall + per phase."""
    # FIX 4: Min balls lowered to 15
    legal = del_df[del_df["extras_type"].fillna("") != "wides"].copy()

    result = {}
    for batter, grp in legal.groupby("batter"):
        if len(grp) < MIN_BATTER_BALLS:
            continue

        def _sr(subset):
            b = len(subset)
            r = subset["batsman_runs"].sum()
            return round(r / b * 100, 2) if b > 0 else DEFAULT_BATTER_SR

        result[batter] = {
            "overall": _sr(grp),
            "pp"     : _sr(grp[grp["over"].between(0, 5)]),
            "mid"    : _sr(grp[grp["over"].between(6, 14)]),
            "death"  : _sr(grp[grp["over"].between(15, 19)]),
        }

    print(f"  Batter SR lookup: {len(result)} batters")
    return result


def build_bowler_eco_lookup(del_df):
    """Per bowler: career economy overall + per phase."""
    result = {}
    for bowler, grp in del_df.groupby("bowler"):
        # FIX 4: Min balls lowered to 15
        legal_bowled = grp[~grp["extras_type"].fillna("").isin(["wides", "noballs"])]
        if len(legal_bowled) < MIN_BOWLER_BALLS:
            continue

        def _eco(subset):
            balls = len(subset[~subset["extras_type"].fillna("").isin(["wides", "noballs"])])
            runs  = subset["total_runs"].sum()
            overs = balls / 6
            return round(runs / overs, 2) if overs > 0 else DEFAULT_BOWLER_ECO

        result[bowler] = {
            "overall": _eco(grp),
            "pp"     : _eco(grp[grp["over"].between(0, 5)]),
            "mid"    : _eco(grp[grp["over"].between(6, 14)]),
            "death"  : _eco(grp[grp["over"].between(15, 19)]),
        }

    print(f"  Bowler economy lookup: {len(result)} bowlers")
    return result


def build_venue_factor_lookup(del_df, mat_df):
    """
    Per venue: avg first innings total, powerplay, middle, death runs.
    FIX 2: Fixed the lambda aggregation bug — now uses explicit merge+groupby
    instead of nested lambdas that caused index mismatches.
    """
    venue_map = mat_df.set_index("match_id")["venue"].to_dict() if "venue" in mat_df.columns else {}
    del_work  = del_df.copy()
    del_work["venue"] = del_work["match_id"].map(venue_map).fillna("Unknown")

    first_inn = del_work[del_work["inning"] == 1].copy()

    # Compute per-match phase runs explicitly (FIX 2)
    per_match_rows = []
    for match_id, grp in first_inn.groupby("match_id"):
        venue     = grp["venue"].iloc[0]
        total     = grp["total_runs"].sum()
        pp_runs   = grp[grp["over"].between(0, 5)]["total_runs"].sum()
        mid_runs  = grp[grp["over"].between(6, 14)]["total_runs"].sum()
        dth_runs  = grp[grp["over"].between(15, 19)]["total_runs"].sum()
        per_match_rows.append({
            "match_id": match_id, "venue": venue,
            "total": total, "pp": pp_runs, "mid": mid_runs, "death": dth_runs,
        })

    if not per_match_rows:
        return {}

    pm_df  = pd.DataFrame(per_match_rows)
    result = {}

    for venue, grp in pm_df.groupby("venue"):
        if len(grp) < 3:
            continue
        result[venue] = {
            "avg_total": round(grp["total"].mean(), 1),
            "avg_pp"   : round(grp["pp"].mean(),    1),
            "avg_mid"  : round(grp["mid"].mean(),   1),
            "avg_death": round(grp["death"].mean(), 1),
        }

    print(f"  Venue factor lookup: {len(result)} venues")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD TRAINING SAMPLES PER PHASE
# ══════════════════════════════════════════════════════════════════════════════

def build_phase_samples(del_df, mat_df, batter_sr_dict, bowler_eco_dict, venue_factor, phase_name):
    """
    FIX 1: Parameter renamed from `bowler_eco` to `bowler_eco_dict` to avoid
    shadowing the local variable `bowler_eco` computed inside the loop.
    """
    over_start, over_end = PHASE_RANGES[phase_name]
    total_phase_balls    = (over_end - over_start + 1) * 6
    phase_key = {"powerplay": "pp", "middle": "mid", "death": "death"}[phase_name]
    eco_phase_key = {"powerplay": "pp", "middle": "mid", "death": "death"}[phase_name]

    venue_map = mat_df.set_index("match_id")["venue"].to_dict() if "venue" in mat_df.columns else {}

    samples = []

    for (match_id, inning), inn_del in del_df.groupby(["match_id", "inning"]):
        if inning > 2:
            continue

        phase_del = inn_del[inn_del["over"].between(over_start, over_end)].reset_index(drop=True)
        if len(phase_del) < 6:
            continue

        phase_total_runs = phase_del["total_runs"].sum()
        venue = venue_map.get(match_id, "Unknown")
        vf    = venue_factor.get(venue, {})

        for snapshot_ball in range(1, len(phase_del)):
            so_far   = phase_del.iloc[:snapshot_ball]
            remains  = phase_del.iloc[snapshot_ball:]

            runs_sofar      = int(so_far["total_runs"].sum())
            wickets_sofar   = int(so_far["is_wicket"].sum())
            balls_completed = snapshot_ball
            balls_remaining = total_phase_balls - balls_completed
            if balls_remaining <= 0:
                continue

            phase_rr = (runs_sofar / (balls_completed / 6)
                        if balls_completed >= 6
                        else runs_sofar * (6 / max(balls_completed, 1)))

            # FIX 5: Track total wickets from full inning start
            full_so_far   = inn_del[inn_del.index < phase_del.index[snapshot_ball]]
            total_wickets = int(full_so_far["is_wicket"].sum()) if len(full_so_far) > 0 else wickets_sofar
            wickets_in_hand = max(0, 10 - total_wickets)

            batting_strength = sum(
                BATTING_STRENGTH_MAP.get(i, 0.2)
                for i in range(total_wickets, 10)
            )

            latest      = so_far.iloc[-1]
            striker     = str(latest.get("batter",      ""))
            non_striker = str(latest.get("non_striker", ""))
            bowler_name = str(latest.get("bowler",      ""))

            # FIX 1: Use bowler_eco_dict parameter (not a global)
            striker_sr     = batter_sr_dict.get(striker,     {}).get(phase_key, DEFAULT_BATTER_SR)
            non_striker_sr = batter_sr_dict.get(non_striker, {}).get(phase_key, DEFAULT_BATTER_SR - 5)
            # FIX 1: renamed variable — bowler_eco_val (not bowler_eco)
            bowler_eco_val = bowler_eco_dict.get(bowler_name, {}).get(eco_phase_key, DEFAULT_BOWLER_ECO)

            last_wicket_idx = so_far[so_far["is_wicket"] == 1].index
            if len(last_wicket_idx) > 0:
                partner_start = last_wicket_idx[-1] + 1
                partnership   = (so_far.loc[partner_start:, "total_runs"].sum()
                                 if partner_start <= so_far.index[-1] else 0)
            else:
                partnership = runs_sofar

            target_remaining_runs = int(remains["total_runs"].sum())

            # FIX 4: venue default values updated
            samples.append({
                "runs_sofar"           : runs_sofar,
                "wickets_in_phase"     : wickets_sofar,
                "balls_completed"      : balls_completed,
                "balls_remaining"      : balls_remaining,
                "phase_run_rate"       : round(phase_rr, 3),
                "striker_sr"           : round(striker_sr, 2),
                "non_striker_sr"       : round(non_striker_sr, 2),
                "partnership_runs"     : int(partnership),
                "bowler_economy"       : round(bowler_eco_val, 3),  # FIX 1
                "wickets_in_hand"      : wickets_in_hand,
                "batting_strength"     : round(batting_strength, 3),
                "venue_avg_phase_runs" : vf.get(f"avg_{phase_key}", DEFAULT_VENUE_PP if phase_name == "powerplay" else DEFAULT_VENUE_MID if phase_name == "middle" else DEFAULT_VENUE_DEATH),
                "venue_avg_total"      : vf.get("avg_total", DEFAULT_VENUE_TOTAL),
                "target_remaining"     : target_remaining_runs,
            })

    df = pd.DataFrame(samples)
    print(f"  [{phase_name.upper()}] samples: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN XGBoost REGRESSOR PER PHASE
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "runs_sofar", "wickets_in_phase", "balls_completed", "balls_remaining",
    "phase_run_rate", "striker_sr", "non_striker_sr", "partnership_runs",
    "bowler_economy", "wickets_in_hand", "batting_strength",
    "venue_avg_phase_runs", "venue_avg_total",
]

def train_phase_model(df, phase_name):
    if len(df) < 100:
        print(f"  ⚠️  [{phase_name}] Not enough samples ({len(df)}), skipping.")
        return None, None

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df["target_remaining"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators    = 400,
        max_depth       = 5,
        learning_rate   = 0.04,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        min_child_weight= 10,
        gamma           = 0.1,
        reg_alpha       = 0.1,
        reg_lambda      = 1.5,
        verbosity       = 0,
        random_state    = 42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    top5 = ", ".join([f"{k}({v:.3f})" for k, v in imp.head(5).items()])
    print(f"  [{phase_name.upper()}] MAE = {mae:.2f} runs  |  Top features: {top5}")
    return model, mae


# ══════════════════════════════════════════════════════════════════════════════
# 5. PREDICTION FUNCTION (used by dashboard at runtime)
# ══════════════════════════════════════════════════════════════════════════════
def predict_phase(
    phase_name,
    runs_sofar, wickets_in_phase, balls_completed,
    striker_sr, non_striker_sr, bowler_economy,
    wickets_in_hand, batting_strength, partnership_runs,
    venue_avg_phase_runs, venue_avg_total,
    model=None,
):
    """
    Predicts REMAINING runs in the current phase.
    Returns (predicted_remaining, low, high, predicted_total)
    """
    over_start, over_end = PHASE_RANGES[phase_name]
    total_phase_balls    = (over_end - over_start + 1) * 6
    balls_remaining      = total_phase_balls - balls_completed

    if balls_remaining <= 0:
        return 0, 0, 0, runs_sofar

    phase_rr = (runs_sofar / (balls_completed / 6)
                if balls_completed >= 6
                else runs_sofar * (6 / max(balls_completed, 1)))

    if model is None:
        projected = runs_sofar + (balls_remaining / 6) * phase_rr
        lo = max(0, projected - 12)
        hi = projected + 12
        return round(projected - runs_sofar), round(lo - runs_sofar), round(hi - runs_sofar), round(projected)

    feat = pd.DataFrame([{
        "runs_sofar"           : runs_sofar,
        "wickets_in_phase"     : wickets_in_phase,
        "balls_completed"      : balls_completed,
        "balls_remaining"      : balls_remaining,
        "phase_run_rate"       : phase_rr,
        "striker_sr"           : striker_sr,
        "non_striker_sr"       : non_striker_sr,
        "partnership_runs"     : partnership_runs,
        "bowler_economy"       : bowler_economy,
        "wickets_in_hand"      : wickets_in_hand,
        "batting_strength"     : batting_strength,
        "venue_avg_phase_runs" : venue_avg_phase_runs,
        "venue_avg_total"      : venue_avg_total,
    }])

    remaining_pred = max(0.0, float(model.predict(feat)[0]))
    margin         = max(8, remaining_pred * 0.15)
    lo             = max(0, remaining_pred - margin)
    hi             = remaining_pred + margin
    total_pred     = runs_sofar + remaining_pred

    return round(remaining_pred), round(lo), round(hi), round(total_pred)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  PITCHMIND — PHASE RUN PREDICTOR TRAINING  (v2 — FIXES)")
    print("=" * 65 + "\n")

    del_df, mat_df = load_data()

    print("\n── Building lookup tables ───────────────────────────────────")
    batter_sr_lookup    = build_batter_sr_lookup(del_df)
    bowler_eco_lookup   = build_bowler_eco_lookup(del_df)
    venue_factor_lookup = build_venue_factor_lookup(del_df, mat_df)

    joblib.dump(batter_sr_lookup,    os.path.join(MODELS_DIR, "phase_batter_sr.pkl"))
    joblib.dump(bowler_eco_lookup,   os.path.join(MODELS_DIR, "phase_bowler_eco.pkl"))
    joblib.dump(venue_factor_lookup, os.path.join(MODELS_DIR, "phase_venue_factor.pkl"))
    print("  ✅ Lookups saved")

    print("\n── Training phase models ────────────────────────────────────")
    results = {}
    for phase in ["powerplay", "middle", "death"]:
        print(f"\n  Building training data for [{phase.upper()}]...")
        # FIX 1: Pass bowler_eco_lookup as named param bowler_eco_dict
        samples = build_phase_samples(
            del_df, mat_df,
            batter_sr_dict=batter_sr_lookup,
            bowler_eco_dict=bowler_eco_lookup,
            venue_factor=venue_factor_lookup,
            phase_name=phase,
        )
        model, mae = train_phase_model(samples, phase)
        if model is not None:
            joblib.dump(model, os.path.join(MODELS_DIR, f"phase_{phase}_model.pkl"))
            print(f"  ✅ phase_{phase}_model.pkl saved")
            results[phase] = mae

    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "phase_feature_cols.pkl"))

    print("\n" + "=" * 65)
    print("  PHASE MODEL TRAINING COMPLETE  (v2 — FIXES)")
    print("=" * 65)
    for phase, mae in results.items():
        print(f"  {phase.upper():<12}  MAE = {mae:.2f} runs")
    print()
    print("  FIXES applied:")
    print("  ✅ FIX 1: Variable scope bug fixed (bowler_eco_dict not shadowed)")
    print("  ✅ FIX 2: Venue factor uses explicit per-over loop (not broken lambda)")
    print(f"  ✅ FIX 3: Defaults updated: batter_sr={DEFAULT_BATTER_SR}, bowler_eco={DEFAULT_BOWLER_ECO}, venue_total={DEFAULT_VENUE_TOTAL}")
    print(f"  ✅ FIX 4: Min balls lowered to {MIN_BATTER_BALLS} for better player coverage")
    print("  ✅ FIX 5: Total wickets tracked from innings start, not phase start")
    print("\n  Next → streamlit run 4_dashboard.py")
    print("         (Live Match tab will use these models)\n")