"""
PITCHMIND — IPL PREDICTION MODEL
Step 3: Model Training  (v5 — 2023+ ERA ONLY)
=============================================
CHANGES vs v4:
  CHANGE 1 — Cutoff moved from 2019 → 2023
             Only 3 seasons of data: 2023, 2024, 2025
             Higher signal quality because all matches are in the same meta:
               - Impact Player rule fully embedded in all matches
               - 185-195 first innings avg across all data
               - Death hitting is consistently aggressive
               - No pre-Impact-Player matches to pollute the model

  CHANGE 2 — Season Weights completely re-tuned for 2023+ range:
             Season | Weight | Reason
             -------|--------|-------
             2023   |  1.0   | Baseline modern IPL (Impact Player year 2)
             2024   |  5.0   | Highest scoring era before 2025, very relevant
             2025   | 10.0   | Most recent — gold standard for predictions

             Within this range, all seasons are high-quality data.
             The main job of weights is to emphasise 2025 over 2023,
             not to filter out pre-modern data (that's done at step 1).

             2025/2023 ratio = 10x  (still strong recency bias)

  CHANGE 3 — All defaults updated for 2023+ era:
             avg_runs default     = 185  (was 175)
             death_runs default   = 55   (was 52)
             top3_sr default      = 145  (was 135)
             economy default      = 10.0 (was 8.5)
             death_economy default= 12.0 (was 9.5)
             season_avg_runs init = 185  (was 170)

  CHANGE 4 — impact_player_era = 1 for ALL matches now (all 2023+)
             No longer a split feature; all training data has Impact Player.
             Kept in feature list for consistency with dashboard inference
             (where it remains 1 for all future predictions).

  CHANGE 5 — season_avg_runs normalised to 185 baseline (not 175)
             team1_relative_score expects ~1.0 when avg_runs ≈ 185

Run:
  python 3_train_model.py

Output:
  models/ipl_model.pkl
  models/ipl_model_rf.pkl
  models/ipl_model_calibrated.pkl
  models/feature_cols.pkl
  models/training_results.txt
  models/optuna_best_params.json
  models/season_weights_used.json
"""

import os
import json
import warnings
from collections import defaultdict, deque
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from xgboost import XGBClassifier

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/master_features.csv"
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

EXCLUDE_COLS = {"match_id", "date", "team1", "team2", "venue", "season", "target"}

# ── CUTOFF ─────────────────────────────────────────────────────────────────────
CUTOFF_SEASON = 2023   # ← CHANGED from 2019

# ── SEASON WEIGHTS (re-tuned for 2023+ range) ──────────────────────────────────
# All three seasons are high-quality data.
# Weights bias toward most-recent (2025) while keeping good representation
# of 2023 and 2024 (which together provide the bulk of training data).
#
# HOW TO TUNE:
# - Model overpredicts (overconfident): lower 2025, raise 2023/2024
# - Model underpredicts recent trends: raise 2025 further (max ~15)
# - Add 2026 below once that season completes
SEASON_WEIGHTS = {
    2023: 1.0,    # Baseline — Impact Player year 2, fully embedded
    2024: 5.0,    # Very high-scoring year, excellent signal quality
    2025: 10.0,   # Most recent — gold standard for current predictions
    # 2026: 15.0, # Uncomment when 2026 season data is available
}
DEFAULT_WEIGHT = 1.0   # fallback (shouldn't trigger with 2023+ only)

OPTUNA_TRIALS   = 30
CV_FOLDS        = 5
OPTUNA_CV_FOLDS = 3

# ── 2023+ ERA DEFAULTS ─────────────────────────────────────────────────────────
# Used in _build_master_features() fallback builder.
# These match the updated BAT_DEFAULTS / BOWL_DEFAULTS in 2_feature_engineering.py.
_BAT_D = {
    "avg_runs"    : 185,    # ↑ was 175
    "run_rate"    : 9.5,    # ↑ was 8.0
    "pp_runs"     : 55.0,   # ↑ was 45
    "middle_runs" : 65.0,   # ↑ was 55
    "death_runs"  : 55.0,   # ↑ was 52
    "boundary_pct": 0.19,
    "dot_ball_pct": 0.28,
    "top3_sr"     : 145.0,  # ↑ was 135
}
_BOWL_D = {
    "economy"      : 10.0,  # ↑ was 8.5
    "death_economy": 12.0,  # ↑ was 9.5
    "pp_wickets"   : 2.0,
    "bowling_sr"   : 17.0,  # ↓ was 20.0
}
_SEASON_AVG_RUNS_DEFAULT = 185.0   # ↑ was 170.0


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK FEATURE BUILDER (runs if master_features.csv doesn't exist yet)
# ══════════════════════════════════════════════════════════════════════════════

def _default_team_state():
    return {
        "games": 0,
        "wins": 0,
        "last_results": deque(maxlen=5),
        "runs_for": 0.0,
        "runs_against": 0.0,
        "balls_faced": 0,
        "balls_bowled": 0,
        "boundaries_for": 0,
        "dots_for": 0,
        "boundaries_against": 0,
        "dots_against": 0,
        "wickets_taken": 0,
        "wickets_lost": 0,
        "pp_runs_for": 0.0,
        "middle_runs_for": 0.0,
        "death_runs_for": 0.0,
        "pp_runs_against": 0.0,
        "middle_runs_against": 0.0,
        "death_runs_against": 0.0,
        "pp_wickets_taken": 0,
        "death_wickets_taken": 0,
        "venue_games": defaultdict(int),
        "venue_wins": defaultdict(int),
        "chase_games": 0,
        "chase_wins": 0,
        "last_match_date": None,
    }


def _team_snapshot(state, venue, season, season_state, current_date):
    games        = state["games"]
    balls_faced  = state["balls_faced"]
    balls_bowled = state["balls_bowled"]

    win_rate     = state["wins"] / games if games else 0.5
    recent_form  = float(np.mean(state["last_results"])) if state["last_results"] else 0.5

    # Updated defaults: use 2023+ era baselines (9.5 RPO, 10.0 econ)
    run_rate         = state["runs_for"] / (balls_faced / 6) if balls_faced else _BAT_D["run_rate"]
    bowling_economy  = state["runs_against"] / (balls_bowled / 6) if balls_bowled else _BOWL_D["economy"]
    batting_sr       = (state["runs_for"] / balls_faced * 100) if balls_faced else _BAT_D["top3_sr"]
    bowling_sr       = balls_bowled / max(state["wickets_taken"], 1) if balls_bowled else _BOWL_D["bowling_sr"]
    nrr              = run_rate - bowling_economy if games else 0.0

    venue_games      = state["venue_games"][venue]
    venue_win_rate   = state["venue_wins"][venue] / venue_games if venue_games else 0.5
    chase_win_rate   = state["chase_wins"] / state["chase_games"] if state["chase_games"] else 0.5

    days_rest = 7.0
    if state["last_match_date"] is not None and current_date is not None:
        try:
            days_rest = float(max((current_date - state["last_match_date"]).days, 1))
        except Exception:
            days_rest = 7.0

    season_avg_runs = (season_state["runs"] / season_state["innings"]
                       if season_state["innings"] else _SEASON_AVG_RUNS_DEFAULT)

    # season_recency: 2023 = baseline (0.71), 2025 = ~1.0
    # Recalibrated for 2023-2025 range (was 2008-based)
    season_recency = min(max((season - 2022) / 3.0, 0.0), 1.0)

    avg_runs     = state["runs_for"] / games if games else _BAT_D["avg_runs"]
    pp_runs      = state["pp_runs_for"] / games if games else _BAT_D["pp_runs"]
    middle_runs  = state["middle_runs_for"] / games if games else _BAT_D["middle_runs"]
    death_runs   = state["death_runs_for"] / games if games else _BAT_D["death_runs"]
    pp_wickets   = state["pp_wickets_taken"] / games if games else _BOWL_D["pp_wickets"]
    death_economy= (state["death_runs_against"] / ((balls_bowled / 6) if balls_bowled else 1)
                    if balls_bowled else _BOWL_D["death_economy"])
    boundary_pct = (state["boundaries_for"] / balls_faced * 100) if balls_faced else _BAT_D["boundary_pct"] * 100
    dot_ball_pct = (state["dots_for"] / balls_faced * 100) if balls_faced else _BAT_D["dot_ball_pct"] * 100

    streak = 0
    for result in reversed(state["last_results"]):
        if result == 1:
            streak += 1
        else:
            break

    # impact_player_era = 1 for all 2023+ matches (always True now)
    return {
        "win_rate"           : round(win_rate, 4),
        "recent_form"        : round(recent_form, 4),
        "nrr"                : round(nrr, 4),
        "avg_runs"           : round(avg_runs, 4),
        "powerplay_runs"     : round(pp_runs, 4),
        "middle_runs"        : round(middle_runs, 4),
        "death_runs"         : round(death_runs, 4),
        "boundary_pct"       : round(boundary_pct, 4),
        "dot_ball_pct"       : round(dot_ball_pct, 4),
        "run_rate"           : round(run_rate, 4),
        "top3_sr"            : round(batting_sr, 4),
        "bowling_economy"    : round(bowling_economy, 4),
        "death_economy"      : round(death_economy, 4),
        "pp_wickets"         : round(pp_wickets, 4),
        "bowling_sr"         : round(bowling_sr, 4),
        "venue_win_rate"     : round(venue_win_rate, 4),
        "chase_win_rate"     : round(chase_win_rate, 4),
        "home_win_rate"      : round(venue_win_rate, 4),
        "win_streak"         : float(streak),
        "days_rest"          : days_rest,
        "squad_bat_sr"       : round(batting_sr, 4),
        "squad_bowl_econ"    : round(bowling_economy, 4),
        "squad_allrounder"   : round(max(state["wickets_taken"] / games if games else 2.0, 0.0), 4),
        "season_recency"     : round(season_recency, 4),
        "season_avg_runs"    : round(season_avg_runs, 4),
        "impact_player_era"  : 1.0,   # Always 1 for 2023+ (Impact Player fully embedded)
        "team1_form_velocity": 0.0,
        "team2_form_velocity": 0.0,
        "diff_form_velocity" : 0.0,
        "team1_relative_score": round(avg_runs / max(season_avg_runs, 100.0), 4),
        "team2_relative_score": round(avg_runs / max(season_avg_runs, 100.0), 4),
        "diff_relative_score" : 0.0,
        "team1_relative_rr"   : round(run_rate / max(season_avg_runs / 20.0, 6.0), 4),
        "team2_relative_rr"   : round(run_rate / max(season_avg_runs / 20.0, 6.0), 4),
    }


def _build_master_features():
    matches_path    = os.path.join("data", "matches_clean.csv")
    deliveries_path = os.path.join("data", "deliveries_clean.csv")

    if not os.path.exists(matches_path):
        raise FileNotFoundError("data/matches_clean.csv not found. Run 1_data_cleaning.py first.")
    if not os.path.exists(deliveries_path):
        raise FileNotFoundError("data/deliveries_clean.csv not found. Run 1_data_cleaning.py first.")

    matches    = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path, low_memory=False)

    matches["date"]       = pd.to_datetime(matches["date"], errors="coerce")
    matches["id"]         = matches["id"].astype(str)
    deliveries.columns    = deliveries.columns.str.lower()
    deliveries["match_id"]= deliveries["match_id"].astype(str)

    for col in ["over", "batsman_runs", "total_runs", "is_wicket"]:
        deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce").fillna(0).astype(int)

    match_stats = {}
    for match_id, group in deliveries.groupby("match_id"):
        innings_stats = {}
        for batting_team, innings in group.groupby("batting_team"):
            legal    = innings[innings["extras_type"].fillna("") != "wides"]
            balls    = len(legal)
            runs     = int(innings["total_runs"].sum())
            wickets  = int(((innings["is_wicket"] == 1) &
                            (innings["dismissal_kind"].fillna("") != "run out")).sum())
            inning_no = int(innings["inning"].iloc[0]) if "inning" in innings.columns else 1
            pp       = innings[innings["over"].isin(range(0, 6))]
            middle   = innings[innings["over"].isin(range(6, 16))]
            death    = innings[innings["over"].isin(range(16, 20))]
            innings_stats[batting_team] = {
                "runs"         : runs,
                "balls"        : balls,
                "wickets"      : wickets,
                "inning"       : inning_no,
                "pp_runs"      : int(pp["total_runs"].sum()),
                "middle_runs"  : int(middle["total_runs"].sum()),
                "death_runs"   : int(death["total_runs"].sum()),
                "boundaries"   : int(innings[innings["batsman_runs"].isin([4, 6])].shape[0]),
                "dots"         : int((innings["total_runs"] == 0).sum()),
                "pp_wickets"   : int(((pp["is_wicket"] == 1) &
                                      (pp["dismissal_kind"].fillna("") != "run out")).sum()),
                "death_wickets": int(((death["is_wicket"] == 1) &
                                      (death["dismissal_kind"].fillna("") != "run out")).sum()),
            }
        match_stats[match_id] = innings_stats

    matches      = matches.sort_values(["date", "id"]).reset_index(drop=True)
    team_state   = defaultdict(_default_team_state)
    season_state = defaultdict(lambda: {"runs": 0.0, "innings": 0})
    h2h_state    = defaultdict(lambda: {"games": 0, "team1_wins": 0})
    rows         = []

    for _, row in matches.iterrows():
        if str(row.get("result", "")).strip().lower() not in \
                {"normal", "tie", "d/l", "dl", "no result"}:
            continue

        team1      = str(row["team1"])
        team2      = str(row["team2"])
        venue      = str(row.get("venue", "Unknown"))
        season     = int(row["season"])
        match_id   = str(row["id"])
        current_date = row.get("date")
        winner     = str(row.get("winner", ""))

        season_info = season_state[season]
        s1 = _team_snapshot(team_state[team1], venue, season, season_info, current_date)
        s2 = _team_snapshot(team_state[team2], venue, season, season_info, current_date)

        pair = tuple(sorted([team1, team2]))
        h2h_info = h2h_state[pair]
        if h2h_info["games"]:
            base_rate  = h2h_info["team1_wins"] / h2h_info["games"]
            h2h_win_rate = base_rate if pair[0] == team1 else 1.0 - base_rate
        else:
            h2h_win_rate = 0.5

        toss_winner_raw  = row.get("toss_winner")
        toss_winner      = toss_winner_raw.strip() if isinstance(toss_winner_raw, str) else ""
        if toss_winner.lower() == "nan":
            toss_winner = ""
        toss_decision    = str(row.get("toss_decision", "bat")).lower()
        toss_win         = 1 if toss_winner == team1 else 0
        toss_field       = 1 if toss_decision == "field" else 0
        toss_team1_field = 1 if (toss_winner == team1 and toss_decision == "field") else 0
        toss_known       = 1 if toss_winner in {team1, team2} else 0

        season_avg_runs = (season_info["runs"] / season_info["innings"]
                           if season_info["innings"] else _SEASON_AVG_RUNS_DEFAULT)

        feature_row = {
            "match_id"              : match_id,
            "date"                  : row["date"],
            "season"                : season,
            "team1"                 : team1,
            "team2"                 : team2,
            "venue"                 : venue,
            "target"                : 1 if winner == team1 else 0,
            "team1_win_rate"        : s1["win_rate"],
            "team2_win_rate"        : s2["win_rate"],
            "team1_recent_form"     : s1["recent_form"],
            "team2_recent_form"     : s2["recent_form"],
            "h2h_win_rate"          : round(h2h_win_rate, 4),
            "team1_nrr"             : s1["nrr"],
            "team2_nrr"             : s2["nrr"],
            "team1_avg_runs"        : s1["avg_runs"],
            "team2_avg_runs"        : s2["avg_runs"],
            "team1_powerplay_runs"  : s1["powerplay_runs"],
            "team2_powerplay_runs"  : s2["powerplay_runs"],
            "team1_middle_runs"     : s1["middle_runs"],
            "team2_middle_runs"     : s2["middle_runs"],
            "team1_death_runs"      : s1["death_runs"],
            "team2_death_runs"      : s2["death_runs"],
            "team1_boundary_pct"    : s1["boundary_pct"],
            "team2_boundary_pct"    : s2["boundary_pct"],
            "team1_dot_ball_pct"    : s1["dot_ball_pct"],
            "team2_dot_ball_pct"    : s2["dot_ball_pct"],
            "team1_run_rate"        : s1["run_rate"],
            "team2_run_rate"        : s2["run_rate"],
            "team1_top3_sr"         : s1["top3_sr"],
            "team2_top3_sr"         : s2["top3_sr"],
            "team1_bowling_economy" : s1["bowling_economy"],
            "team2_bowling_economy" : s2["bowling_economy"],
            "team1_death_economy"   : s1["death_economy"],
            "team2_death_economy"   : s2["death_economy"],
            "team1_pp_wickets"      : s1["pp_wickets"],
            "team2_pp_wickets"      : s2["pp_wickets"],
            "team1_bowling_sr"      : s1["bowling_sr"],
            "team2_bowling_sr"      : s2["bowling_sr"],
            "team1_venue_win_rate"  : s1["venue_win_rate"],
            "team2_venue_win_rate"  : s2["venue_win_rate"],
            "venue_avg_runs"        : round(season_avg_runs, 4),
            "toss_win"              : toss_win,
            "team1_toss_win"        : toss_win,
            "toss_known"            : toss_known,
            "toss_field"            : toss_field,
            "toss_team1_field"      : toss_team1_field,
            "team1_chase_win_rate"  : s1["chase_win_rate"],
            "team2_chase_win_rate"  : s2["chase_win_rate"],
            "team1_home_win_rate"   : s1["home_win_rate"],
            "team2_home_win_rate"   : s2["home_win_rate"],
            "team1_win_streak"      : s1["win_streak"],
            "team2_win_streak"      : s2["win_streak"],
            "team1_days_rest"       : s1["days_rest"],
            "team2_days_rest"       : s2["days_rest"],
            "season_stage"          : 0,
            "team1_squad_bat_sr"    : s1["squad_bat_sr"],
            "team2_squad_bat_sr"    : s2["squad_bat_sr"],
            "team1_squad_bowl_econ" : s1["squad_bowl_econ"],
            "team2_squad_bowl_econ" : s2["squad_bowl_econ"],
            "team1_squad_allrounder": s1["squad_allrounder"],
            "team2_squad_allrounder": s2["squad_allrounder"],
            "diff_win_rate"         : s1["win_rate"] - s2["win_rate"],
            "diff_recent_form"      : s1["recent_form"] - s2["recent_form"],
            "diff_avg_runs"         : s1["avg_runs"] - s2["avg_runs"],
            "diff_death_runs"       : s1["death_runs"] - s2["death_runs"],
            "diff_death_economy"    : s2["death_economy"] - s1["death_economy"],
            "diff_bowling_economy"  : s2["bowling_economy"] - s1["bowling_economy"],
            "diff_pp_wickets"       : s1["pp_wickets"] - s2["pp_wickets"],
            "diff_run_rate"         : s1["run_rate"] - s2["run_rate"],
            "diff_nrr"              : s1["nrr"] - s2["nrr"],
            "diff_venue_win_rate"   : s1["venue_win_rate"] - s2["venue_win_rate"],
            "diff_chase_win_rate"   : s1["chase_win_rate"] - s2["chase_win_rate"],
            "diff_squad_bat_sr"     : s1["squad_bat_sr"] - s2["squad_bat_sr"],
            "season_recency"        : s1["season_recency"],
            "season_avg_runs"       : round(season_avg_runs, 4),
            "impact_player_era"     : 1.0,   # always 1 for 2023+
            "team1_form_velocity"   : 0.0,
            "team2_form_velocity"   : 0.0,
            "diff_form_velocity"    : 0.0,
            "team1_relative_score"  : s1["team1_relative_score"],
            "team2_relative_score"  : s2["team2_relative_score"],
            "diff_relative_score"   : s1["team1_relative_score"] - s2["team2_relative_score"],
            "team1_relative_rr"     : s1["team1_relative_rr"],
            "team2_relative_rr"     : s2["team2_relative_rr"],
        }
        rows.append(feature_row)

        innings_stats = match_stats.get(match_id, {})
        for batting_team, stats in innings_stats.items():
            bowling_team  = team2 if batting_team == team1 else team1
            batting_state = team_state[batting_team]
            bowling_state = team_state[bowling_team]

            batting_state["games"]         += 1
            batting_state["runs_for"]      += stats["runs"]
            batting_state["balls_faced"]   += stats["balls"]
            batting_state["boundaries_for"]+= stats["boundaries"]
            batting_state["dots_for"]      += stats["dots"]
            batting_state["wickets_lost"]  += stats["wickets"]
            batting_state["pp_runs_for"]   += stats["pp_runs"]
            batting_state["middle_runs_for"]+= stats["middle_runs"]
            batting_state["death_runs_for"]+= stats["death_runs"]
            batting_state["pp_wickets_taken"] += stats["pp_wickets"]
            batting_state["death_wickets_taken"] += stats["death_wickets"]
            batting_state["last_results"].append(1 if winner == batting_team else 0)
            batting_state["venue_games"][venue] += 1
            if winner == batting_team:
                batting_state["wins"] += 1
                batting_state["venue_wins"][venue] += 1
            if stats["inning"] == 2:
                batting_state["chase_games"] += 1
                if winner == batting_team:
                    batting_state["chase_wins"] += 1
            batting_state["last_match_date"] = current_date

            bowling_state["balls_bowled"]    += stats["balls"]
            bowling_state["runs_against"]    += stats["runs"]
            bowling_state["boundaries_against"] += stats["boundaries"]
            bowling_state["dots_against"]    += stats["dots"]
            bowling_state["wickets_taken"]   += stats["wickets"]
            bowling_state["pp_runs_against"] += stats["pp_runs"]
            bowling_state["middle_runs_against"] += stats["middle_runs"]
            bowling_state["death_runs_against"] += stats["death_runs"]

        season_state[season]["runs"]   += sum(v["runs"] for v in innings_stats.values())
        season_state[season]["innings"]+= len(innings_stats)

        h2h_state[pair]["games"] += 1
        if winner == team1 and pair[0] == team1:
            h2h_state[pair]["team1_wins"] += 1
        elif winner == team2 and pair[0] == team2:
            h2h_state[pair]["team1_wins"] += 1

    features = pd.DataFrame(rows)
    features = features.sort_values(["date", "match_id"]).reset_index(drop=True)
    out_path = os.path.join("data", "master_features.csv")
    features.to_csv(out_path, index=False)
    print(f"Built fallback feature table → {out_path} ({features.shape[0]} rows, {features.shape[1]} cols)")
    return features


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """
    Load master_features.csv DIRECTLY — no rebuild.
    Run 1_data_cleaning.py + 2_feature_engineering.py first to generate it.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found.\n"
            "Run in order:\n"
            "  1. python 1_data_cleaning.py\n"
            "  2. python 2_feature_engineering.py\n"
            "Then re-run this script."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset     →  shape: {df.shape}")
    print(f"   Seasons         →  {sorted(df['season'].unique())}")
    print(f"   Target balance  →  team1 wins = {df['target'].mean():.1%}")

    # Verify shape (expect ~254 rows, 71 cols from your pipeline)
    if df.shape[0] < 200:
        print(f"WARNING: Only {df.shape[0]} rows — expected 250+. Check feature engineering step.")
    if df.shape[1] < 60:
        print(f"WARNING: Only {df.shape[1]} cols — expected 70+. Check feature engineering step.")

    # Enforce 2023+ cutoff
    if df["season"].min() < CUTOFF_SEASON:
        print(f"WARNING: Pre-{CUTOFF_SEASON} data found. Filtering...")
        df = df[df["season"] >= CUTOFF_SEASON].copy()
        print(f"   After filter: {len(df)} rows remaining.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPARE FEATURES & TARGET
# ══════════════════════════════════════════════════════════════════════════════
def prepare_data(df):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    if "team1_avg_runs" in df.columns and "season_avg_runs" in df.columns:
        df = df.copy()
        # Relative score normalised to 2023+ baseline (expect ~1.0 when avg≈185)
        df["team1_relative_score"] = (df["team1_avg_runs"] /
                                      df["season_avg_runs"].clip(lower=100)).round(4)
        df["team2_relative_score"] = (df["team2_avg_runs"] /
                                      df["season_avg_runs"].clip(lower=100)).round(4)
        df["diff_relative_score"]  = (df["team1_relative_score"] -
                                      df["team2_relative_score"]).round(4)
        df["team1_relative_rr"]    = (df["team1_run_rate"] /
                                      (df["season_avg_runs"] / 20).clip(lower=6.0)).round(4)
        df["team2_relative_rr"]    = (df["team2_run_rate"] /
                                      (df["season_avg_runs"] / 20).clip(lower=6.0)).round(4)

        for c in ["team1_relative_score", "team2_relative_score", "diff_relative_score",
                  "team1_relative_rr", "team2_relative_rr"]:
            if c not in feature_cols:
                feature_cols.append(c)

        print(f"\n   ✅ Relative score features added:")
        print(f"      team1_relative_score mean = {df['team1_relative_score'].mean():.4f}  (expect ~1.0)")
        print(f"      season_avg_runs mean      = {df['season_avg_runs'].mean():.1f}  "
              f"(expect 180-195 for 2023+)")

    print(f"\n   Feature columns    →  {len(feature_cols)}")

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["target"]
    return X, y, feature_cols, df


# ══════════════════════════════════════════════════════════════════════════════
# 3. SEASON WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def compute_season_weights(df):
    """
    Explicit per-season weights for 2023+ range.
    
    Key difference from v4:
    - v4 had a 10x spread across 2019-2025 (6 seasons)
    - v5 has a 10x spread across 2023-2025 (3 seasons)
    - This means 2024 gets 5x weight relative to 2023 (much stronger)
    - 2025 still gets the highest weight but 2023 is only 10x below
    
    All 3 seasons are genuinely modern IPL — the weights fine-tune
    recency preference, not era filtering.
    """
    seasons = df["season"].astype(int)
    weights = seasons.map(lambda s: SEASON_WEIGHTS.get(s, DEFAULT_WEIGHT))
    weights = weights / weights.mean()   # normalise so mean = 1

    print(f"\n── Season Weights (2023+ era) ───────────────────────────")
    print(f"   {'Season':<8} {'Raw Weight':>12} {'Normalized':>12} {'Matches':>10}")
    print(f"   {'-'*46}")
    for season in sorted(seasons.unique()):
        raw_w  = SEASON_WEIGHTS.get(int(season), DEFAULT_WEIGHT)
        norm_w = weights[seasons == season].iloc[0]
        n_matches = (seasons == season).sum()
        bar = "█" * int(norm_w * 5)
        print(f"   {season:<8} {raw_w:>12.1f} {norm_w:>12.3f} {n_matches:>10}   {bar}")

    seasons_avail = sorted(seasons.unique())
    if len(seasons_avail) >= 2:
        max_s = seasons_avail[-1]
        min_s = seasons_avail[0]
        ratio = weights[seasons == max_s].iloc[0] / weights[seasons == min_s].iloc[0]
        print(f"\n   Weight ratio {max_s}/{min_s}: {ratio:.1f}x\n")

    weights_path = os.path.join(MODELS_DIR, "season_weights_used.json")
    with open(weights_path, "w") as f:
        json.dump({
            "season_weights_raw": SEASON_WEIGHTS,
            "cutoff_season"     : CUTOFF_SEASON,
            "normalized"        : {str(s): round(float(weights[seasons == s].iloc[0]), 4)
                                   for s in sorted(seasons.unique())}
        }, f, indent=2)
    print(f"   ✅ Weights saved → {weights_path}")

    return weights.values


# ══════════════════════════════════════════════════════════════════════════════
# 4. TIME-AWARE TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def split_data(df, X, y, weights):
    """80/20 time-aware split. Test set = most recent 20% of matches."""
    split_idx = int(len(X) * 0.80)
    X_train   = X.iloc[:split_idx]
    X_test    = X.iloc[split_idx:]
    y_train   = y.iloc[:split_idx]
    y_test    = y.iloc[split_idx:]
    w_train   = weights[:split_idx]

    train_seasons = df["season"].iloc[:split_idx]
    test_seasons  = df["season"].iloc[split_idx:]

    print(f"── Train / Test Split ───────────────────────────────────")
    print(f"   Train samples      →  {len(X_train)}  "
          f"(seasons {train_seasons.min()}–{train_seasons.max()})")
    print(f"   Test  samples      →  {len(X_test)}   "
          f"(seasons {test_seasons.min()}–{test_seasons.max()}, no leakage)\n")
    return X_train, X_test, y_train, y_test, w_train


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def run_cross_validation(X, y, xgb_params):
    print("── TimeSeriesSplit Cross-Validation ─────────────────────")
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    xgb_cv = XGBClassifier(**xgb_params, verbosity=0, use_label_encoder=False,
                            eval_metric="logloss", random_state=42)
    xgb_scores = cross_val_score(xgb_cv, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)

    rf_cv = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   min_samples_leaf=5, max_features="sqrt",
                                   random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf_cv, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)

    print(f"   XGBoost CV         →  {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
    print(f"   Random Forest CV   →  {rf_scores.mean():.4f} ± {rf_scores.std():.4f}\n")
    return xgb_scores, rf_scores


# ══════════════════════════════════════════════════════════════════════════════
# 6. OPTUNA HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
def run_optuna_tuning(X_train, y_train, w_train):
    print("── Optuna Hyperparameter Search ─────────────────────────")
    print(f"   Trials: {OPTUNA_TRIALS}  |  Inner CV: {OPTUNA_CV_FOLDS}-fold TimeSeriesSplit\n")

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
            model = XGBClassifier(**params, verbosity=0, use_label_encoder=False,
                                  eval_metric="logloss", random_state=42)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                      sample_weight=w_train[train_idx])
            scores.append(accuracy_score(y_train.iloc[val_idx],
                                         model.predict(X_train.iloc[val_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_score  = study.best_value

    print(f"   Best CV accuracy   →  {best_score:.4f}  ({best_score:.2%})")
    for k, v in best_params.items():
        print(f"      {k:<22} = {v}")
    print()

    with open(os.path.join(MODELS_DIR, "optuna_best_params.json"), "w") as f:
        json.dump({"best_params": best_params, "best_cv_score": best_score}, f, indent=2)

    return best_params


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
def train_random_forest(X_train, y_train, X_test, y_test, w_train):
    print("── Random Forest (season-weighted) ──────────────────────")
    rf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5,
                                max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train, sample_weight=w_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"   RF Accuracy        →  {acc:.4f}  ({acc:.2%})\n")
    return rf, acc


# ══════════════════════════════════════════════════════════════════════════════
# 8. TRAIN XGBOOST
# ══════════════════════════════════════════════════════════════════════════════
def train_xgboost(X_train, y_train, X_test, y_test, w_train, best_params):
    print("── XGBoost (Optuna params + season-weighted) ────────────")
    xgb = XGBClassifier(**best_params, eval_metric="logloss",
                        use_label_encoder=False, verbosity=0, random_state=42)
    xgb.fit(X_train, y_train, sample_weight=w_train,
            eval_set=[(X_test, y_test)], verbose=False)
    acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"   XGB Accuracy       →  {acc:.4f}  ({acc:.2%})\n")
    return xgb, acc


# ══════════════════════════════════════════════════════════════════════════════
# 9. RAW ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
def build_raw_ensemble(rf, xgb, X_test, y_test):
    print("── Raw Ensemble (RF 50% + XGB 50%) ──────────────────────")
    avg_prob   = (rf.predict_proba(X_test)[:, 1] + xgb.predict_proba(X_test)[:, 1]) / 2
    final_pred = (avg_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, final_pred)
    print(f"   Ensemble Accuracy  →  {acc:.4f}  ({acc:.2%})\n")
    return acc, final_pred, avg_prob


# ══════════════════════════════════════════════════════════════════════════════
# 10. PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_models(rf, xgb, X_test, y_test):
    print("── Probability Calibration (Isotonic Regression) ────────")

    xgb_cal = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
    xgb_cal.fit(X_test, y_test)
    rf_cal  = CalibratedClassifierCV(FrozenEstimator(rf),  method="isotonic")
    rf_cal.fit(X_test, y_test)

    cal_avg_prob = (rf_cal.predict_proba(X_test)[:, 1] +
                    xgb_cal.predict_proba(X_test)[:, 1]) / 2
    cal_pred     = (cal_avg_prob >= 0.5).astype(int)
    raw_avg_prob = (rf.predict_proba(X_test)[:, 1] +
                    xgb.predict_proba(X_test)[:, 1]) / 2

    cal_acc   = accuracy_score(y_test, cal_pred)
    brier_raw = brier_score_loss(y_test, raw_avg_prob)
    brier_cal = brier_score_loss(y_test, cal_avg_prob)

    print(f"   Calibrated Accuracy →  {cal_acc:.4f}  ({cal_acc:.2%})")
    print(f"   Brier Score raw/cal →  {brier_raw:.4f} / {brier_cal:.4f}  "
          f"({'✅ improved' if brier_raw > brier_cal else '⚠️ no improvement'})\n")

    print("   Calibrated probability distribution:")
    for lo, hi, label in [(0.4, 0.6, "40-60% uncertain"), (0.6, 0.7, "60-70% moderate"),
                           (0.7, 0.8, "70-80% confident"), (0.8, 1.0, "80%+ high conf")]:
        mask = (cal_avg_prob >= lo) & (cal_avg_prob < hi)
        n = mask.sum()
        if n > 0:
            print(f"      {label}: {n:3d} predictions, actual win rate = {y_test[mask].mean():.1%}")
    print()

    return xgb_cal, rf_cal, cal_acc


# ══════════════════════════════════════════════════════════════════════════════
# 11. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def print_feature_importance(rf, xgb, feature_cols):
    print("── Top 15 Feature Importances ───────────────────────────")
    rf_imp  = pd.Series(rf.feature_importances_,  index=feature_cols).sort_values(ascending=False)
    xgb_imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)

    print(f"   {'Feature':<32} {'RF Rank':>7}  {'XGB Rank':>8}")
    for rank, (feat, val) in enumerate(rf_imp.head(15).items(), 1):
        xgb_rank = list(xgb_imp.index).index(feat) + 1 if feat in xgb_imp.index else "N/A"
        bar = "█" * int(val * 60)
        print(f"   {feat:<32}  #{rank:<5}  #{xgb_rank:<5}  {bar}  {val:.4f}")

    print(f"\n   Era/context features ranking:")
    for feat in ["season_recency", "season_avg_runs", "impact_player_era",
                 "team1_relative_score", "team2_relative_score",
                 "team1_death_runs", "team2_death_runs",
                 "team1_bowling_economy", "team2_bowling_economy"]:
        if feat in rf_imp.index:
            r = list(rf_imp.index).index(feat) + 1
            x = list(xgb_imp.index).index(feat) + 1 if feat in xgb_imp.index else "N/A"
            print(f"      {feat:<32}  RF #{r}  XGB #{x}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 12. SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def save_artifacts(rf, xgb, xgb_cal, rf_cal, feature_cols,
                   rf_acc, xgb_acc, ens_acc, cal_acc, xgb_cv_scores, best_params):
    joblib.dump(xgb,          os.path.join(MODELS_DIR, "ipl_model.pkl"))
    joblib.dump(rf,           os.path.join(MODELS_DIR, "ipl_model_rf.pkl"))
    joblib.dump(xgb_cal,      os.path.join(MODELS_DIR, "ipl_model_calibrated.pkl"))
    joblib.dump(rf_cal,       os.path.join(MODELS_DIR, "ipl_model_rf_calibrated.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    with open(os.path.join(MODELS_DIR, "training_results.txt"), "w") as f:
        f.write("PITCHMIND — IPL Model Training Results (v5 — 2023+ Era)\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Data range             : 2023–2025 only\n")
        f.write(f"Cutoff season          : {CUTOFF_SEASON}\n")
        f.write(f"Random Forest Accuracy : {rf_acc:.2%}\n")
        f.write(f"XGBoost Accuracy       : {xgb_acc:.2%}\n")
        f.write(f"Raw Ensemble Accuracy  : {ens_acc:.2%}\n")
        f.write(f"Calibrated Ensemble    : {cal_acc:.2%}\n")
        f.write(f"XGBoost CV (5-fold)    : {xgb_cv_scores.mean():.2%} ± {xgb_cv_scores.std():.2%}\n")
        f.write(f"Features Used          : {len(feature_cols)}\n\n")
        f.write(f"2023+ Era Defaults:\n")
        f.write(f"  avg_runs     = {_BAT_D['avg_runs']}\n")
        f.write(f"  death_runs   = {_BAT_D['death_runs']}\n")
        f.write(f"  top3_sr      = {_BAT_D['top3_sr']}\n")
        f.write(f"  economy      = {_BOWL_D['economy']}\n")
        f.write(f"  death_econ   = {_BOWL_D['death_economy']}\n\n")
        f.write("Season Weights:\n")
        for s, w in sorted(SEASON_WEIGHTS.items()):
            f.write(f"  {s}: {w}\n")
        f.write("\nOptuna Best Params:\n")
        for k, v in best_params.items():
            f.write(f"  {k:<22} = {v}\n")

    print(f"✅ ipl_model.pkl               saved")
    print(f"✅ ipl_model_rf.pkl            saved")
    print(f"✅ ipl_model_calibrated.pkl    saved  (USE THIS in dashboard)")
    print(f"✅ ipl_model_rf_calibrated.pkl saved")
    print(f"✅ feature_cols.pkl            saved  ({len(feature_cols)} features)")
    print(f"✅ training_results.txt        saved")
    print(f"✅ season_weights_used.json    saved")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  PITCHMIND — IPL MODEL TRAINING  v5 (2023+ Era Only)")
    print("  Seasons: 2023 | 2024 | 2025")
    print("  Weights: 2023=1x  2024=5x  2025=10x")
    print(f"  Defaults: avg={_BAT_D['avg_runs']} | death={_BAT_D['death_runs']} | "
          f"econ={_BOWL_D['economy']} | death_econ={_BOWL_D['death_economy']}")
    print("=" * 65 + "\n")

    df                                         = load_data()
    X, y, feature_cols, df                     = prepare_data(df)
    weights                                    = compute_season_weights(df)
    X_train, X_test, y_train, y_test, w_train  = split_data(df, X, y, weights)

    best_params = run_optuna_tuning(X_train, y_train, w_train)
    xgb_cv_scores, rf_cv_scores = run_cross_validation(X, y, best_params)

    rf_model,  rf_acc  = train_random_forest(X_train, y_train, X_test, y_test, w_train)
    xgb_model, xgb_acc = train_xgboost(X_train, y_train, X_test, y_test, w_train, best_params)

    ens_acc, final_pred, raw_probs = build_raw_ensemble(rf_model, xgb_model, X_test, y_test)
    xgb_cal, rf_cal, cal_acc      = calibrate_models(rf_model, xgb_model, X_test, y_test)

    print_feature_importance(rf_model, xgb_model, feature_cols)
    print("── Classification Report ────────────────────────────────")
    print(classification_report(y_test, final_pred, target_names=["Team2 Wins", "Team1 Wins"]))

    save_artifacts(rf_model, xgb_model, xgb_cal, rf_cal, feature_cols,
                   rf_acc, xgb_acc, ens_acc, cal_acc, xgb_cv_scores, best_params)

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE  (v5 — 2023+ Era)")
    print("=" * 65)
    print(f"  Random Forest Accuracy     :  {rf_acc:.2%}")
    print(f"  XGBoost Accuracy           :  {xgb_acc:.2%}")
    print(f"  Raw Ensemble Accuracy      :  {ens_acc:.2%}")
    print(f"  Calibrated Ensemble Acc    :  {cal_acc:.2%}")
    print(f"  XGB CV (5-fold, time-aware):  {xgb_cv_scores.mean():.2%} ± {xgb_cv_scores.std():.2%}")
    print()
    print("  WHAT'S NEW vs v4 (2019→2023 cutoff):")
    print("  ✅ Only 3 seasons: 2023, 2024, 2025 — all same IPL meta")
    print("  ✅ All data has Impact Player (no pre/post split needed)")
    print("  ✅ avg_runs default: 175 → 185")
    print("  ✅ death_runs default: 52 → 55")
    print("  ✅ top3_sr default: 135 → 145")
    print("  ✅ economy default: 8.5 → 10.0")
    print("  ✅ death_economy default: 9.5 → 12.0")
    print("  ✅ season_avg_runs baseline: 170 → 185")
    print("  ✅ impact_player_era = 1 for ALL training rows")
    print()
    print("  SEASON WEIGHTS:")
    for s, w in sorted(SEASON_WEIGHTS.items()):
        bar = "█" * int(w)
        print(f"    {s}: {w:>5.1f}x  {bar}")
    print()
    print("  NOTE: Smaller dataset (~3 seasons) is expected and correct.")
    print("  Higher signal quality compensates for lower quantity.")
    print("  If CV accuracy seems low, ensure 2025 data is fully loaded.")
    print()
    print("  TO TUNE: Edit SEASON_WEIGHTS at top of this file.")
    print("  If predictions seem stale → increase 2025 weight (up to 15).")
    print("  If overfitting → lower 2025, raise 2023/2024.")
    print("=" * 65)
    print("\n✅ Next → streamlit run 4_dashboard.py\n")