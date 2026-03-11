"""
PITCHMIND — STEP 7: In-Match Phase Run Predictor
=================================================
Trains XGBoost regressors to predict remaining runs in each phase
(Powerplay / Middle / Death) from current match state mid-phase.

Key Features:
  - Batsman strike rate (live striker + non-striker)
  - Bowling strength (current bowler economy)
  - Wickets in hand + batting strength remaining
  - Venue run factor
  - All 3 phases modeled separately for accuracy

Run:
    python 7_phase_predictor.py

Saves:
    models/phase_pp_model.pkl
    models/phase_mid_model.pkl
    models/phase_death_model.pkl
    models/phase_batter_sr.pkl      ← per-batter career SR lookup
    models/phase_bowler_eco.pkl     ← per-bowler career economy lookup
    models/phase_venue_factor.pkl   ← venue avg run factor lookup
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

# Over ranges per phase (0-indexed as in deliveries.csv)
PHASE_RANGES = {
    "powerplay": (0,  5),   # overs 1-6
    "middle"   : (6,  14),  # overs 7-15
    "death"    : (15, 19),  # overs 16-20
}

# Batting order strength weights — top order worth more
# We approximate using historical SR by position
BATTING_STRENGTH_MAP = {
    0: 1.0, 1: 1.0, 2: 0.95, 3: 0.90, 4: 0.85,
    5: 0.80, 6: 0.75, 7: 0.65, 8: 0.50, 9: 0.35, 10: 0.25,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    del_path = os.path.join(DATA_DIR, "deliveries_clean.csv")
    if not os.path.exists(del_path):
        del_path = os.path.join(DATA_DIR, "deliveries.csv")

    mat_path = os.path.join(DATA_DIR, "matches_clean.csv")
    if not os.path.exists(mat_path):
        mat_path = os.path.join(DATA_DIR, "matches.csv")

    print(f"  Loading {del_path} ...")
    del_df = pd.read_csv(del_path, low_memory=False)
    del_df.columns = del_df.columns.str.lower()

    # Ensure numeric
    for col in ["over", "ball", "batsman_runs", "extra_runs", "total_runs", "is_wicket"]:
        del_df[col] = pd.to_numeric(del_df[col], errors="coerce").fillna(0)
    del_df["over"]  = del_df["over"].astype(int)
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
# 2. BUILD CAREER LOOKUP TABLES  (batter SR, bowler economy, venue factor)
# ══════════════════════════════════════════════════════════════════════════════
def build_batter_sr_lookup(del_df):
    """
    Per batter: career strike rate overall + per phase.
    Returns dict: batter_name → {"overall": sr, "pp": sr, "mid": sr, "death": sr}
    """
    legal = del_df[del_df["extras_type"].fillna("") != "wides"].copy()

    result = {}
    for batter, grp in legal.groupby("batter"):
        if len(grp) < 20:
            continue

        def _sr(subset):
            b = len(subset)
            r = subset["batsman_runs"].sum()
            return round(r / b * 100, 2) if b > 0 else 120.0

        result[batter] = {
            "overall": _sr(grp),
            "pp"     : _sr(grp[grp["over"].between(0, 5)]),
            "mid"    : _sr(grp[grp["over"].between(6, 14)]),
            "death"  : _sr(grp[grp["over"].between(15, 19)]),
        }

    print(f"  Batter SR lookup: {len(result)} batters")
    return result


def build_bowler_eco_lookup(del_df):
    """
    Per bowler: career economy overall + per phase.
    Returns dict: bowler_name → {"overall": eco, "pp": eco, "death": eco}
    """
    result = {}
    for bowler, grp in del_df.groupby("bowler"):
        if len(grp) < 20:
            continue

        def _eco(subset):
            balls = len(subset)
            runs  = subset["total_runs"].sum()
            overs = balls / 6
            return round(runs / overs, 2) if overs > 0 else 8.5

        result[bowler] = {
            "overall": _eco(grp),
            "pp"     : _eco(grp[grp["over"].between(0, 5)]),
            "death"  : _eco(grp[grp["over"].between(15, 19)]),
        }

    print(f"  Bowler economy lookup: {len(result)} bowlers")
    return result


def build_venue_factor_lookup(del_df, mat_df):
    """
    Per venue: avg first innings total, powerplay, middle, death runs.
    Returns dict: venue → {"avg_total": x, "avg_pp": x, "avg_mid": x, "avg_death": x}
    """
    # Merge venue into deliveries
    venue_map = mat_df.set_index("match_id")["venue"].to_dict() if "venue" in mat_df.columns else {}
    del_df["venue"] = del_df["match_id"].map(venue_map).fillna("Unknown")

    first_inn = del_df[del_df["inning"] == 1]

    result = {}
    for venue, grp in first_inn.groupby("venue"):
        if grp["match_id"].nunique() < 3:
            continue

        per_match = grp.groupby("match_id").agg(
            total=("total_runs", "sum"),
            pp=("total_runs",    lambda x: x[grp.loc[x.index, "over"].between(0,5)].sum()),
            mid=("total_runs",   lambda x: x[grp.loc[x.index, "over"].between(6,14)].sum()),
            death=("total_runs", lambda x: x[grp.loc[x.index, "over"].between(15,19)].sum()),
        )

        result[venue] = {
            "avg_total": round(per_match["total"].mean(), 1),
            "avg_pp"   : round(per_match["pp"].mean(), 1),
            "avg_mid"  : round(per_match["mid"].mean(), 1),
            "avg_death": round(per_match["death"].mean(), 1),
        }

    print(f"  Venue factor lookup: {len(result)} venues")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD TRAINING SAMPLES PER PHASE
#    For each phase, simulate "mid-phase snapshots" at every ball.
#    Target = total runs scored in the REMAINING balls of that phase.
# ══════════════════════════════════════════════════════════════════════════════
def build_phase_samples(del_df, mat_df, batter_sr, bowler_eco, venue_factor, phase_name):
    over_start, over_end = PHASE_RANGES[phase_name]
    total_phase_balls    = (over_end - over_start + 1) * 6

    venue_map = mat_df.set_index("match_id")["venue"].to_dict() if "venue" in mat_df.columns else {}

    samples = []

    for (match_id, inning), inn_del in del_df.groupby(["match_id", "inning"]):
        if inning > 2:
            continue

        # Phase deliveries only
        phase_del = inn_del[inn_del["over"].between(over_start, over_end)].reset_index(drop=True)
        if len(phase_del) < 6:   # need at least 1 over of phase data
            continue

        # Total runs actually scored in this phase (ground truth)
        phase_total_runs = phase_del["total_runs"].sum()
        venue = venue_map.get(match_id, "Unknown")
        vf    = venue_factor.get(venue, {})

        # Create a snapshot at each ball mid-phase (from ball 1 onwards)
        for snapshot_ball in range(1, len(phase_del)):
            so_far   = phase_del.iloc[:snapshot_ball]
            remains  = phase_del.iloc[snapshot_ball:]

            runs_sofar      = int(so_far["total_runs"].sum())
            wickets_sofar   = int(so_far["is_wicket"].sum())
            balls_completed = snapshot_ball
            balls_remaining = total_phase_balls - balls_completed
            if balls_remaining <= 0:
                continue

            # Current run rate in phase
            phase_rr = runs_sofar / (balls_completed / 6) if balls_completed >= 6 else runs_sofar * (6 / max(balls_completed, 1))

            # Wickets in hand (full inning wickets up to now)
            full_so_far     = inn_del[inn_del.index < phase_del.index[snapshot_ball]]
            total_wickets   = int(full_so_far["is_wicket"].sum()) if len(full_so_far) > 0 else wickets_sofar
            wickets_in_hand = 10 - total_wickets

            # Batting strength remaining — proxy via wickets in hand weighted
            batting_strength = sum(
                BATTING_STRENGTH_MAP.get(i, 0.2)
                for i in range(total_wickets, 10)
            )

            # Current striker & non-striker SR
            # Most recent delivery tells us who's batting
            latest = so_far.iloc[-1]
            striker     = latest.get("batter",       "")
            non_striker = latest.get("non_striker",  "")
            bowler      = latest.get("bowler",       "")

            # SR lookup per phase
            phase_key = {"powerplay": "pp", "middle": "mid", "death": "death"}[phase_name]
            striker_sr     = batter_sr.get(striker,     {}).get(phase_key, 120.0)
            non_striker_sr = batter_sr.get(non_striker, {}).get(phase_key, 115.0)
            bowler_eco     = bowler_eco_lookup.get(bowler, {}).get(
                "pp" if phase_name == "powerplay" else ("death" if phase_name == "death" else "overall"),
                8.5
            )

            # Partnership runs (since last wicket in this phase)
            last_wicket_idx = so_far[so_far["is_wicket"] == 1].index
            if len(last_wicket_idx) > 0:
                partner_start = last_wicket_idx[-1] + 1
                partnership   = so_far.loc[partner_start:, "total_runs"].sum() if partner_start <= so_far.index[-1] else 0
            else:
                partnership = runs_sofar

            # Target: runs in remaining balls of this phase
            target_remaining_runs = int(remains["total_runs"].sum())

            samples.append({
                # Current state
                "runs_sofar"           : runs_sofar,
                "wickets_in_phase"     : wickets_sofar,
                "balls_completed"      : balls_completed,
                "balls_remaining"      : balls_remaining,
                "phase_run_rate"       : round(phase_rr, 3),
                # Batsmen
                "striker_sr"           : round(striker_sr, 2),
                "non_striker_sr"       : round(non_striker_sr, 2),
                "partnership_runs"     : int(partnership),
                # Bowling
                "bowler_economy"       : round(bowler_eco, 3),
                # Batting strength
                "wickets_in_hand"      : wickets_in_hand,
                "batting_strength"     : round(batting_strength, 3),
                # Venue
                "venue_avg_phase_runs" : vf.get(f"avg_{phase_key if phase_key != 'mid' else 'mid'}", 55.0),
                "venue_avg_total"      : vf.get("avg_total", 160.0),
                # Target
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

    # Time-aware split: 80/20
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

    # Feature importance
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

    All parameters:
      phase_name           : "powerplay" | "middle" | "death"
      runs_sofar           : runs scored in this phase so far
      wickets_in_phase     : wickets fallen IN this phase
      balls_completed      : balls bowled in this phase
      striker_sr           : live striker career SR (this phase)
      non_striker_sr       : live non-striker career SR (this phase)
      bowler_economy       : current bowler career economy (this phase)
      wickets_in_hand      : 10 - total wickets in innings
      batting_strength     : sum of BATTING_STRENGTH_MAP for remaining batters
      partnership_runs     : runs in current partnership
      venue_avg_phase_runs : venue historical avg runs in this phase
      venue_avg_total      : venue historical avg 1st innings total
      model                : loaded XGBoost model (or None for fallback)
    """
    over_start, over_end = PHASE_RANGES[phase_name]
    total_phase_balls    = (over_end - over_start + 1) * 6
    balls_remaining      = total_phase_balls - balls_completed

    if balls_remaining <= 0:
        return 0, 0, 0, runs_sofar

    phase_rr = runs_sofar / (balls_completed / 6) if balls_completed >= 6 else (
        runs_sofar * (6 / max(balls_completed, 1))
    )

    if model is None:
        # Fallback: simple projection
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

    remaining_pred = float(model.predict(feat)[0])
    remaining_pred = max(0, remaining_pred)

    # Confidence interval: ±1 MAE (rough approximation: ~8–12 runs)
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
    print("  PITCHMIND — PHASE RUN PREDICTOR TRAINING")
    print("=" * 65 + "\n")

    # Load
    del_df, mat_df = load_data()

    # Build lookup tables
    print("\n── Building lookup tables ───────────────────────────────────")
    batter_sr_lookup    = build_batter_sr_lookup(del_df)
    bowler_eco_lookup   = build_bowler_eco_lookup(del_df)
    venue_factor_lookup = build_venue_factor_lookup(del_df, mat_df)

    # Save lookups
    joblib.dump(batter_sr_lookup,    os.path.join(MODELS_DIR, "phase_batter_sr.pkl"))
    joblib.dump(bowler_eco_lookup,   os.path.join(MODELS_DIR, "phase_bowler_eco.pkl"))
    joblib.dump(venue_factor_lookup, os.path.join(MODELS_DIR, "phase_venue_factor.pkl"))
    print("  ✅ Lookups saved")

    # Train per-phase models
    print("\n── Training phase models ────────────────────────────────────")
    results = {}
    for phase in ["powerplay", "middle", "death"]:
        print(f"\n  Building training data for [{phase.upper()}]...")
        samples   = build_phase_samples(del_df, mat_df, batter_sr_lookup, bowler_eco_lookup, venue_factor_lookup, phase)
        model, mae = train_phase_model(samples, phase)
        if model is not None:
            joblib.dump(model, os.path.join(MODELS_DIR, f"phase_{phase}_model.pkl"))
            print(f"  ✅ phase_{phase}_model.pkl saved")
            results[phase] = mae

    # Save feature column list
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "phase_feature_cols.pkl"))

    # Summary
    print("\n" + "=" * 65)
    print("  PHASE MODEL TRAINING COMPLETE")
    print("=" * 65)
    for phase, mae in results.items():
        print(f"  {phase.upper():<12}  MAE = {mae:.2f} runs")
    print("\n  Next → streamlit run 4_dashboard.py")
    print("         (Live Match tab will use these models)\n")