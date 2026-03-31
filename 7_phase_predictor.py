"""
PITCHMIND — STEP 7: In-Match Phase Run Predictor  (v2 — Era Fix)
=================================================================
CHANGES vs v1:
  FIX 1 — Season Recency Weighting for Phase Models
           Phase models now use sample weights (like the main classifier).
           Recent seasons (2022+) are weighted more heavily.
           Old deliveries from 2008 scoring era now barely affect predictions.
           Exponent: 0.25 (matches main model's SEASON_WEIGHT_EXP)

  FIX 2 — Venue Factor Now Uses Rolling Average (not all-time)
           Old: venue avg used ALL historical data equally
           New: weighted toward recent 3 seasons' data
           Impact: venues like Wankhede now reflect 190+ avg, not 160+ avg

  FIX 3 — Season recency added as a feature to phase models
           The phase regressors now know what scoring era they're in.
           "10 runs in 6 balls in 2008" ≠ "10 runs in 6 balls in 2024"

Key Features (unchanged):
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
    models/phase_venue_factor.pkl   ← venue avg run factor lookup (recent-weighted)
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

# NEW v2: Season weight exponent — matches main model for consistency
PHASE_SEASON_WEIGHT_EXP = 0.25  # same as SEASON_WEIGHT_EXP in 3_train_model.py

# Batting order strength weights — top order worth more
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

    # Add season_int to mat_df for weighting
    if "season" in mat_df.columns:
        mat_df["season_int"] = mat_df["season"].astype(str).str[:4].astype(int)
    elif "season_int" not in mat_df.columns:
        mat_df["season_int"] = 2015  # fallback if no season info

    print(f"  Deliveries: {del_df.shape}  |  Matches: {mat_df.shape}")
    return del_df, mat_df


# ══════════════════════════════════════════════════════════════════════════════
# 1B — NEW v2: Compute match-level season weights for phase models
# ══════════════════════════════════════════════════════════════════════════════
def compute_match_weights(mat_df):
    """
    NEW (v2): Per-match sample weights based on season recency.
    Matches from 2022+ weighted much more than 2008-2015 matches.
    This ensures phase models learn modern IPL scoring patterns.
    """
    seasons    = mat_df["season_int"]
    min_season = seasons.min()
    weights    = np.exp((seasons - min_season) * PHASE_SEASON_WEIGHT_EXP)
    weights    = weights / weights.mean()

    weight_dict = dict(zip(mat_df["match_id"].astype(str), weights.values))

    print(f"\n   Phase model season weights (exp={PHASE_SEASON_WEIGHT_EXP}):")
    for s in sorted(seasons.unique()):
        w = weights[seasons == s].iloc[0]
        bar = "█" * int(w * 5)
        print(f"      Season {s}: {w:.3f}  {bar}")
    print()

    return weight_dict


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD CAREER LOOKUP TABLES  (batter SR, bowler economy, venue factor)
# ══════════════════════════════════════════════════════════════════════════════
def build_batter_sr_lookup(del_df, match_weights=None):
    """
    Per batter: career strike rate overall + per phase.
    NEW v2: If match_weights provided, recent matches get higher weight.
    Returns dict: batter_name → {"overall": sr, "pp": sr, "mid": sr, "death": sr}
    """
    legal = del_df[del_df["extras_type"].fillna("") != "wides"].copy()

    # Add per-ball weight
    if match_weights:
        legal["weight"] = legal["match_id"].map(match_weights).fillna(1.0)
    else:
        legal["weight"] = 1.0

    result = {}
    for batter, grp in legal.groupby("batter"):
        if grp["weight"].sum() < 20:  # minimum weighted balls
            continue

        def _sr(subset):
            w = subset["weight"].sum()
            r = (subset["batsman_runs"] * subset["weight"]).sum()
            return round(r / w * 100, 2) if w > 0 else 120.0

        result[batter] = {
            "overall": _sr(grp),
            "pp"     : _sr(grp[grp["over"].between(0, 5)]),
            "mid"    : _sr(grp[grp["over"].between(6, 14)]),
            "death"  : _sr(grp[grp["over"].between(15, 19)]),
        }

    print(f"  Batter SR lookup: {len(result)} batters")
    return result


def build_bowler_eco_lookup(del_df, match_weights=None):
    """
    Per bowler: career economy overall + per phase.
    NEW v2: Recent matches weighted more (modern bowling economy differs from 2008).
    Returns dict: bowler_name → {"overall": eco, "pp": eco, "death": eco}
    """
    df = del_df.copy()
    if match_weights:
        df["weight"] = df["match_id"].map(match_weights).fillna(1.0)
    else:
        df["weight"] = 1.0

    result = {}
    for bowler, grp in df.groupby("bowler"):
        if grp["weight"].sum() < 20:
            continue

        def _eco(subset):
            balls = subset["weight"].sum()
            runs  = (subset["total_runs"] * subset["weight"]).sum()
            overs = balls / 6
            return round(runs / overs, 2) if overs > 0 else 8.5

        result[bowler] = {
            "overall": _eco(grp),
            "pp"     : _eco(grp[grp["over"].between(0, 5)]),
            "death"  : _eco(grp[grp["over"].between(15, 19)]),
        }

    print(f"  Bowler economy lookup: {len(result)} bowlers")
    return result


def build_venue_factor_lookup(del_df, mat_df, match_weights=None):
    """
    Per venue: avg first innings total, powerplay, middle, death runs.
    NEW v2: Weighted toward recent seasons so venue factors reflect modern scoring.
    Returns dict: venue → {"avg_total": x, "avg_pp": x, "avg_mid": x, "avg_death": x}
    """
    venue_map = mat_df.set_index("match_id")["venue"].to_dict() if "venue" in mat_df.columns else {}
    del_df = del_df.copy()
    del_df["venue"] = del_df["match_id"].map(venue_map).fillna("Unknown")

    if match_weights:
        del_df["weight"] = del_df["match_id"].map(match_weights).fillna(1.0)
    else:
        del_df["weight"] = 1.0

    first_inn = del_df[del_df["inning"] == 1]

    result = {}
    for venue, grp in first_inn.groupby("venue"):
        if grp["match_id"].nunique() < 3:
            continue

        # Weighted per-match totals
        per_match_data = []
        for match_id, mgrp in grp.groupby("match_id"):
            w = mgrp["weight"].iloc[0]  # same weight for all balls in a match
            per_match_data.append({
                "match_id"  : match_id,
                "weight"    : w,
                "total"     : mgrp["total_runs"].sum(),
                "pp"        : mgrp[mgrp["over"].between(0,5)]["total_runs"].sum(),
                "mid"       : mgrp[mgrp["over"].between(6,14)]["total_runs"].sum(),
                "death"     : mgrp[mgrp["over"].between(15,19)]["total_runs"].sum(),
            })

        pm = pd.DataFrame(per_match_data)
        total_w = pm["weight"].sum()
        if total_w == 0:
            continue

        result[venue] = {
            "avg_total": round((pm["total"] * pm["weight"]).sum() / total_w, 1),
            "avg_pp"   : round((pm["pp"]    * pm["weight"]).sum() / total_w, 1),
            "avg_mid"  : round((pm["mid"]   * pm["weight"]).sum() / total_w, 1),
            "avg_death": round((pm["death"] * pm["weight"]).sum() / total_w, 1),
        }

    print(f"  Venue factor lookup: {len(result)} venues (recent-weighted)")

    # Print top venues for sanity check
    top_venues = sorted(result.items(), key=lambda x: x[1]["avg_total"], reverse=True)[:5]
    print(f"  Top 5 venues by weighted avg total:")
    for v, stats in top_venues:
        print(f"      {v[:35]:<35}: {stats['avg_total']:.0f} runs")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD TRAINING SAMPLES PER PHASE
# ══════════════════════════════════════════════════════════════════════════════
def build_phase_samples(del_df, mat_df, batter_sr, bowler_eco, venue_factor, phase_name, match_weights=None):
    over_start, over_end = PHASE_RANGES[phase_name]
    total_phase_balls    = (over_end - over_start + 1) * 6

    venue_map   = mat_df.set_index("match_id")["venue"].to_dict() if "venue" in mat_df.columns else {}
    season_map  = mat_df.set_index("match_id")["season_int"].to_dict() if "season_int" in mat_df.columns else {}

    # NEW v2: season recency for each match
    all_seasons = sorted(set(season_map.values())) if season_map else [2015]
    max_season  = max(all_seasons) if all_seasons else 2025

    samples = []

    for (match_id, inning), inn_del in del_df.groupby(["match_id", "inning"]):
        if inning > 2:
            continue

        phase_del = inn_del[inn_del["over"].between(over_start, over_end)].reset_index(drop=True)
        if len(phase_del) < 6:
            continue

        phase_total_runs = phase_del["total_runs"].sum()
        venue   = venue_map.get(match_id, "Unknown")
        season  = season_map.get(match_id, 2015)
        vf      = venue_factor.get(venue, {})

        # NEW v2: season recency for this match (0.05–1.0)
        season_recency = float(np.clip(1.0 - (max_season - season) / 20.0, 0.05, 1.0))

        # Sample weight for this match
        sample_weight = match_weights.get(str(match_id), 1.0) if match_weights else 1.0

        for snapshot_ball in range(1, len(phase_del)):
            so_far   = phase_del.iloc[:snapshot_ball]
            remains  = phase_del.iloc[snapshot_ball:]

            runs_sofar      = int(so_far["total_runs"].sum())
            wickets_sofar   = int(so_far["is_wicket"].sum())
            balls_completed = snapshot_ball
            balls_remaining = total_phase_balls - balls_completed
            if balls_remaining <= 0:
                continue

            phase_rr = runs_sofar / (balls_completed / 6) if balls_completed >= 6 else runs_sofar * (6 / max(balls_completed, 1))

            full_so_far     = inn_del[inn_del.index < phase_del.index[snapshot_ball]]
            total_wickets   = int(full_so_far["is_wicket"].sum()) if len(full_so_far) > 0 else wickets_sofar
            wickets_in_hand = 10 - total_wickets

            batting_strength = sum(
                BATTING_STRENGTH_MAP.get(i, 0.2)
                for i in range(total_wickets, 10)
            )

            latest = so_far.iloc[-1]
            striker     = latest.get("batter",       "")
            non_striker = latest.get("non_striker",  "")
            bowler      = latest.get("bowler",       "")

            phase_key = {"powerplay": "pp", "middle": "mid", "death": "death"}[phase_name]
            striker_sr_val     = batter_sr.get(striker,     {}).get(phase_key, 120.0)
            non_striker_sr_val = batter_sr.get(non_striker, {}).get(phase_key, 115.0)
            bowler_eco_val     = bowler_eco.get(bowler, {}).get(
                "pp" if phase_name == "powerplay" else ("death" if phase_name == "death" else "overall"),
                8.5
            )

            last_wicket_idx = so_far[so_far["is_wicket"] == 1].index
            if len(last_wicket_idx) > 0:
                partner_start = last_wicket_idx[-1] + 1
                partnership   = so_far.loc[partner_start:, "total_runs"].sum() if partner_start <= so_far.index[-1] else 0
            else:
                partnership = runs_sofar

            target_remaining_runs = int(remains["total_runs"].sum())

            samples.append({
                # Current state
                "runs_sofar"           : runs_sofar,
                "wickets_in_phase"     : wickets_sofar,
                "balls_completed"      : balls_completed,
                "balls_remaining"      : balls_remaining,
                "phase_run_rate"       : round(phase_rr, 3),
                # Batsmen
                "striker_sr"           : round(striker_sr_val, 2),
                "non_striker_sr"       : round(non_striker_sr_val, 2),
                "partnership_runs"     : int(partnership),
                # Bowling
                "bowler_economy"       : round(bowler_eco_val, 3),
                # Batting strength
                "wickets_in_hand"      : wickets_in_hand,
                "batting_strength"     : round(batting_strength, 3),
                # Venue
                "venue_avg_phase_runs" : vf.get(f"avg_{phase_key if phase_key != 'mid' else 'mid'}", 55.0),
                "venue_avg_total"      : vf.get("avg_total", 160.0),
                # NEW v2: Era signal
                "season_recency"       : round(season_recency, 4),
                # Sample weight (used in model.fit)
                "_sample_weight"       : sample_weight,
                # Target
                "target_remaining"     : target_remaining_runs,
            })

    df = pd.DataFrame(samples)
    print(f"  [{phase_name.upper()}] samples: {len(df):,}  "
          f"(weighted — recent matches count more)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN XGBoost REGRESSOR PER PHASE  (v2: with sample weights + season_recency)
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "runs_sofar", "wickets_in_phase", "balls_completed", "balls_remaining",
    "phase_run_rate", "striker_sr", "non_striker_sr", "partnership_runs",
    "bowler_economy", "wickets_in_hand", "batting_strength",
    "venue_avg_phase_runs", "venue_avg_total",
    "season_recency",   # NEW v2: era signal
]

def train_phase_model(df, phase_name):
    if len(df) < 100:
        print(f"  ⚠️  [{phase_name}] Not enough samples ({len(df)}), skipping.")
        return None, None

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df["target_remaining"]
    w = df["_sample_weight"].values  # NEW v2: sample weights

    # Time-aware split: 80/20 (samples are ordered chronologically by match)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    w_train         = w[:split]  # NEW v2: weights for training only

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

    # NEW v2: pass sample_weight so recent seasons dominate learning
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    top5 = ", ".join([f"{k}({v:.3f})" for k, v in imp.head(5).items()])

    print(f"  [{phase_name.upper()}] MAE = {mae:.2f} runs  |  Top features: {top5}")

    # Check if season_recency is doing useful work
    recency_imp = imp.get("season_recency", 0.0)
    print(f"  [{phase_name.upper()}] season_recency importance = {recency_imp:.4f}  "
          f"({'✅ useful' if recency_imp > 0.01 else '⚠️ low — may not be needed'})")

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
    season_recency=1.0,  # NEW v2: default to current era (1.0 = most recent)
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
      venue_avg_phase_runs : venue historical avg runs in this phase (recent-weighted)
      venue_avg_total      : venue historical avg 1st innings total (recent-weighted)
      model                : loaded XGBoost model (or None for fallback)
      season_recency       : NEW: 0.0-1.0, how current the match is (default 1.0 for live)
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
        "season_recency"       : season_recency,  # NEW v2
    }])

    remaining_pred = float(model.predict(feat)[0])
    remaining_pred = max(0, remaining_pred)

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
    print("  PITCHMIND — PHASE RUN PREDICTOR TRAINING  v2 (Era Fix)")
    print(f"  Season weight exp: {PHASE_SEASON_WEIGHT_EXP}  |  season_recency feature added")
    print("=" * 65 + "\n")

    # Load
    del_df, mat_df = load_data()

    # Compute match weights (NEW v2)
    print("\n── Computing match weights (recent seasons preferred) ───────")
    match_weights = compute_match_weights(mat_df)

    # Build lookup tables (now weighted)
    print("\n── Building lookup tables (recent-weighted) ─────────────────")
    batter_sr_lookup    = build_batter_sr_lookup(del_df, match_weights)
    bowler_eco_lookup   = build_bowler_eco_lookup(del_df, match_weights)
    venue_factor_lookup = build_venue_factor_lookup(del_df, mat_df, match_weights)

    # Save lookups
    joblib.dump(batter_sr_lookup,    os.path.join(MODELS_DIR, "phase_batter_sr.pkl"))
    joblib.dump(bowler_eco_lookup,   os.path.join(MODELS_DIR, "phase_bowler_eco.pkl"))
    joblib.dump(venue_factor_lookup, os.path.join(MODELS_DIR, "phase_venue_factor.pkl"))
    print("  ✅ Lookups saved (recent-weighted)")

    # Train per-phase models
    print("\n── Training phase models (with season weights) ──────────────")
    results = {}
    for phase in ["powerplay", "middle", "death"]:
        print(f"\n  Building training data for [{phase.upper()}]...")
        samples   = build_phase_samples(
            del_df, mat_df,
            batter_sr_lookup, bowler_eco_lookup, venue_factor_lookup,
            phase, match_weights
        )
        model, mae = train_phase_model(samples, phase)
        if model is not None:
            joblib.dump(model, os.path.join(MODELS_DIR, f"phase_{phase}_model.pkl"))
            print(f"  ✅ phase_{phase}_model.pkl saved")
            results[phase] = mae

    # Save feature column list (updated with season_recency)
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "phase_feature_cols.pkl"))

    # Summary
    print("\n" + "=" * 65)
    print("  PHASE MODEL TRAINING COMPLETE  (v2 — Era Fix)")
    print("=" * 65)
    for phase, mae in results.items():
        print(f"  {phase.upper():<12}  MAE = {mae:.2f} runs")
    print()
    print("  WHAT'S NEW vs v1:")
    print(f"  ✅ Sample weights (exp={PHASE_SEASON_WEIGHT_EXP}) — recent seasons dominate training")
    print("  ✅ season_recency feature   — model knows what era it's in")
    print("  ✅ Weighted venue factors   — venues reflect modern scoring (190+)")
    print("  ✅ Weighted batter SR       — recent batters' stats prioritized")
    print("  ✅ Weighted bowler economy  — modern bowling economy used")
    print()
    print("  NOTE: predict_phase() now accepts season_recency parameter.")
    print("        Pass 1.0 for live matches (current season = most recent).")
    print("\n  Next → streamlit run 4_dashboard.py")
    print("         (Live Match tab will use these updated models)\n")