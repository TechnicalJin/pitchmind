"""
PITCHMIND — 2_feature_engineering.py  (v7 — PLAYING XI FEATURES)
============================================================
PLAYING XI-BASED FEATURES (GAME CHANGER)

Movement: Team-level averages → Match-specific player strength

9 CORE FEATURES + 4 PLAYING XI FEATURES = 13 TOTAL

Core Features (9):
  ✅ 4 base: team1_death_economy, season_recency, team1_win_rate, team2_squad_bowl_econ
  ✅ 5 tournament: last5_win_pct, h2h, season_points, season_nrr

NEW Playing XI Features (4):
  ✅ playing11_batting_sr - Average strike rate of XI batsmen
  ✅ playing11_bowling_economy - Average economy of XI bowlers
  ✅ num_death_bowlers - Count of death-over specialists in XI
  ✅ num_powerplay_bowlers - Count of powerplay specialists in XI

Replacement:
  ❌ team2_squad_bowl_econ (weak squad average)
  ✅ Replaced with playing11_bowling_economy (actual XI)

Run:
  python 2_feature_engineering.py

Output:
  data/master_features.csv  (13 core features + identifiers + target)
"""

import pandas as pd
import numpy as np
import os
import warnings
from utils import (
    normalize_team,
    apply_team_normalization,
    validate_toss_distribution,
    validate_team_name_consistency
)
warnings.filterwarnings("ignore")

DATA_DIR = "data"

# ══════════════════════════════════════════════════════════════════════════════
# VENUE STANDARDIZATION
# ══════════════════════════════════════════════════════════════════════════════

VENUE_MAP = {
    "Feroz Shah Kotla"                          : "Arun Jaitley Stadium",
    "Wankhede Stadium, Mumbai"                  : "Wankhede Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai"  : "MA Chidambaram Stadium, Chepauk",
    "Rajiv Gandhi International Stadium, Uppal" : "Rajiv Gandhi International Stadium",
}

# Each team's known home venues (for home win rate feature)
HOME_VENUES = {
    "Chennai Super Kings"        : ["MA Chidambaram Stadium, Chepauk", "MA Chidambaram Stadium"],
    "Mumbai Indians"             : ["Wankhede Stadium", "Dr DY Patil Sports Academy", "Brabourne Stadium"],
    "Kolkata Knight Riders"      : ["Eden Gardens"],
    "Royal Challengers Bengaluru": ["M Chinnaswamy Stadium"],
    "Sunrisers Hyderabad"        : ["Rajiv Gandhi International Stadium"],
    "Delhi Capitals"             : ["Feroz Shah Kotla", "Arun Jaitley Stadium"],
    "Punjab Kings"               : ["Punjab Cricket Association Stadium, Mohali", "IS Bindra Stadium"],
    "Rajasthan Royals"           : ["Sawai Mansingh Stadium"],
    "Gujarat Titans"             : ["Narendra Modi Stadium"],
    "Lucknow Super Giants"       : ["Ekana Cricket Stadium", "BRSABV Ekana Cricket Stadium"],
    "Gujarat Lions"              : ["Saurashtra Cricket Association Stadium"],
    "Rising Pune Supergiant"     : ["Maharashtra Cricket Association Stadium"],
    "Kochi Tuskers Kerala"       : ["Jawaharlal Nehru Stadium, Kochi"],
    "Pune Warriors"              : ["Subrata Roy Sahara Stadium"],
}

def is_home_venue(team, venue):
    """Returns 1 if this venue is the team's home ground."""
    home_list = HOME_VENUES.get(team, [])
    venue_lower = venue.lower()
    return any(hv.lower() in venue_lower or venue_lower in hv.lower() for hv in home_list)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMN REGISTRY  (v5 — 68 features)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 FEATURE SET (ENFORCED)
# ══════════════════════════════════════════════════════════════════════════════
# These 27 core features drive model performance (~99% accuracy drivers)
# NO OTHER FEATURES WILL BE USED
#
FEATURE_COLS = [
    # BASE FEATURES (4)
    # Top predictors from XGBoost importance ranking
    "team1_death_economy",
    "season_recency",
    "team1_win_rate",
    "team2_squad_bowl_econ",

    # MATCH CONTEXT FEATURES (5)
    # Real-time tournament signals capturing momentum, standings, and H2H dynamics
    "team1_last5_win_pct",
    "team2_last5_win_pct",
    "last3_h2h_team1_pct",
    "current_season_points_diff",
    "current_season_nrr_diff",

    # PLAYING XI FEATURES (8) — GAME CHANGER
    # Actual match composition replaces weak squad averages
    # Captures real player impact on match day
    "team1_playing11_bat_sr",
    "team2_playing11_bat_sr",
    "team1_playing11_bowl_econ",
    "team2_playing11_bowl_econ",
    "team1_num_death_bowlers",
    "team2_num_death_bowlers",
    "team1_num_pp_bowlers",
    "team2_num_pp_bowlers",

    # VENUE & PITCH INTELLIGENCE (5) — HIDDEN GOLD
    # Ground-specific dynamics: pitch behavior, dew impact, chasing advantage, boundary effects
    "pitch_type",
    "dew_factor",
    "boundary_size",
    "avg_chasing_success",
    "venue_avg_runs",                           # Rolling avg first innings score at venue (160 vs 220)

    # TOSS INTELLIGENCE (5) — VENUE-AWARE, NOT RANDOM
    # Replace weak generic toss features with venue-specific impact
    # Toss only matters when it unlocks venue advantage (chasing at chase-friendly ground)
    "toss_advantage_at_venue",                  # How much toss win boosts win % at this venue
    "team1_chasing_strength",                   # Team 1's historical chase win rate
    "team2_chasing_strength",                   # Team 2's historical chase win rate
    "team1_chase_venue_interaction",            # Team1 chase strength × venue chasing bias
    "team2_chase_venue_interaction",            # Team2 chase strength × venue chasing bias

    # SCORING STRENGTH FEATURES (4) — CRITICAL: ACTUAL SCORING POWER
    # Phase-specific scoring that model was NOT seeing before
    "team1_death_runs",                         # Team 1's death overs scoring (overs 15-20)
    "team2_death_runs",                         # Team 2's death overs scoring (overs 15-20)
    "team1_powerplay_runs",                     # Team 1's powerplay scoring (overs 1-6)
    "team2_powerplay_runs",                     # Team 2's powerplay scoring (overs 1-6)

    # MOMENTUM FEATURES (6) — HOT vs COLD TEAM DETECTION
    # Short-term performance signals reflecting current form
    "team1_last_3_match_avg_runs",              # Team 1's avg runs in last 3 matches (batting momentum)
    "team2_last_3_match_avg_runs",              # Team 2's avg runs in last 3 matches (batting momentum)
    "team1_last_3_match_wickets_taken",         # Team 1's avg wickets in last 3 matches (bowling momentum)
    "team2_last_3_match_wickets_taken",         # Team 2's avg wickets in last 3 matches (bowling momentum)
    "team1_form_trend_slope",                   # Team 1's performance trend (positive = improving)
    "team2_form_trend_slope",                   # Team 2's performance trend (positive = improving)

    # DIFFERENCE FEATURES (2) — TREE MODELS LOVE COMPARISONS
    # Direct comparisons improve decision boundaries
    "diff_run_rate",                            # team1_run_rate - team2_run_rate
    "diff_avg_runs",                            # team1_avg_runs - team2_avg_runs

    # INTERACTION FEATURES (2) — ACTUAL MATCH DYNAMICS
    # Correct interactions: phase scoring × opponent phase weakness
    "death_runs_vs_opponent_death_economy",     # team1 death runs × team2 death economy (weakness)
    "powerplay_runs_vs_opponent_pp_wickets",    # team1 pp runs × team2 pp wicket-taking ability
]

# Default values for teams/players with no prior history
BAT_DEFAULTS = {
    "avg_runs": 155, "run_rate": 8.0, "pp_runs": 45.0,
    "middle_runs": 55.0, "death_runs": 40.0,
    "boundary_pct": 0.16, "dot_ball_pct": 0.35, "top3_sr": 117.0,
}
BOWL_DEFAULTS = {
    "economy": 8.5, "death_economy": 9.5,
    "pp_wickets": 1.5, "bowling_sr": 20.0,
}

# VENUE DEFAULTS (for venues with insufficient history)
VENUE_DEFAULTS = {
    "pitch_type": 1,  # 0=flat, 1=slow, 2=spin, 3=pace (default: balanced slow)
    "dew_factor": 0,  # 0=day/dry, 1=night/dew
    "boundary_size": 1,  # 0=small, 1=medium, 2=large (default: standard medium)
    "avg_chasing_success": 0.45,  # typical chase win %
}

# VENUE CHARACTERISTICS (pre-defined based on IPL ground knowledge)
VENUE_CHARACTERISTICS = {
    "M Chinnaswamy Stadium": {"pitch_type": 2, "boundary_size": 0, "label": "flat/boundaries-small"},
    "Wankhede Stadium": {"pitch_type": 1, "boundary_size": 2, "label": "slow/boundaries-large"},
    "MA Chidambaram Stadium, Chepauk": {"pitch_type": 2, "boundary_size": 0, "label": "spin/boundaries-small"},
    "Eden Gardens": {"pitch_type": 1, "boundary_size": 0, "label": "slow/boundaries-small"},
    "Rajiv Gandhi International Stadium": {"pitch_type": 0, "boundary_size": 1, "label": "flat/boundaries-medium"},
    "Arun Jaitley Stadium": {"pitch_type": 1, "boundary_size": 1, "label": "slow/boundaries-medium"},
    "Punjab Cricket Association Stadium, Mohali": {"pitch_type": 0, "boundary_size": 1, "label": "flat/boundaries-medium"},
    "IS Bindra Stadium": {"pitch_type": 0, "boundary_size": 1, "label": "flat/boundaries-medium"},
    "Sawai Mansingh Stadium": {"pitch_type": 3, "boundary_size": 1, "label": "pace/boundaries-medium"},
    "Narendra Modi Stadium": {"pitch_type": 0, "boundary_size": 2, "label": "flat/boundaries-large"},
    "Ekana Cricket Stadium": {"pitch_type": 0, "boundary_size": 1, "label": "flat/boundaries-medium"},
    "BRSABV Ekana Cricket Stadium": {"pitch_type": 0, "boundary_size": 1, "label": "flat/boundaries-medium"},
    "Dr DY Patil Sports Academy": {"pitch_type": 1, "boundary_size": 1, "label": "slow/boundaries-medium"},
    "Brabourne Stadium": {"pitch_type": 1, "boundary_size": 1, "label": "slow/boundaries-medium"},
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("Loading datasets...")
    matches    = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
    deliveries = pd.read_csv(os.path.join(DATA_DIR, "deliveries.csv"))
    print(f"  matches.csv    : {matches.shape}")
    print(f"  deliveries.csv : {deliveries.shape}")

    apply_team_normalization(matches, ["team1", "team2", "toss_winner", "winner"])
    apply_team_normalization(deliveries, ["batting_team", "bowling_team"])
    validate_team_name_consistency(matches, ["team1", "team2", "toss_winner", "winner"])

    matches["venue"]      = matches["venue"].replace(VENUE_MAP)
    matches["season_int"] = matches["season"].astype(str).str[:4].astype(int)
    matches["date"]       = pd.to_datetime(matches["date"])

    matches = matches[matches["winner"].notna()].copy()
    matches = matches[matches["result"] != "no result"].copy()
    matches = matches.sort_values("date").reset_index(drop=True)

    deliveries = deliveries[deliveries["inning"].isin([1, 2])].copy()

    print(f"  After cleaning  : {matches.shape}")
    return matches, deliveries


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2A — EXISTING ROLLING FEATURES (all pre-match, no leakage)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_win_rate(matches):
    """Expanding cumulative win rate per team — shift(1) ensures no leakage."""
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")
    df["win_rate"] = df.groupby("team")["win"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["win_rate"] = df["win_rate"].fillna(0.5)
    return df.set_index(["match_id", "team"])["win_rate"].to_dict()


def compute_rolling_recent_form(matches, window=5):
    """Rolling win rate in last N matches — EWM for recency weighting."""
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")
    df["recent_form"] = df.groupby("team")["win"].transform(
        lambda x: x.shift(1).ewm(span=window, adjust=False).mean()
    )
    df["recent_form"] = df["recent_form"].fillna(0.5)
    return df.set_index(["match_id", "team"])["recent_form"].to_dict()


def compute_rolling_h2h(matches):
    """Rolling H2H win rate — only uses past matches before current date."""
    sorted_matches = matches.sort_values("date")
    records = []
    for _, row in sorted_matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        pair = tuple(sorted([t1, t2]))
        records.append({
            "match_id": row["id"], "date": row["date"],
            "team1": t1, "team2": t2, "pair": pair,
            "t1_won": int(row["winner"] == t1),
        })
    df = pd.DataFrame(records)

    result     = {}
    pair_wins  = {}
    pair_total = {}

    for _, row in df.iterrows():
        mid, t1, t2, pair = row["match_id"], row["team1"], row["team2"], row["pair"]
        if pair not in pair_total or pair_total[pair] == 0:
            result[(mid, t1, t2)] = 0.5
        else:
            t1_wins = pair_wins[pair].get(t1, 0)
            result[(mid, t1, t2)] = round(t1_wins / pair_total[pair], 4)

        if pair not in pair_wins:
            pair_wins[pair] = {}
            pair_total[pair] = 0
        pair_wins[pair][t1] = pair_wins[pair].get(t1, 0) + row["t1_won"]
        pair_wins[pair][t2] = pair_wins[pair].get(t2, 0) + (1 - row["t1_won"])
        pair_total[pair] += 1

    return result


def compute_rolling_nrr(matches, deliveries):
    """Correct NRR from deliveries: rolling avg run_rate scored - conceded."""
    match_info = matches[["id", "date"]].rename(columns={"id": "match_id"})

    inn_stats = deliveries.groupby(["match_id", "batting_team"]).agg(
        runs=("total_runs", "sum"),
        overs=("over", "nunique")
    ).reset_index()
    inn_stats["rr"] = inn_stats["runs"] / inn_stats["overs"].clip(lower=1)

    bowl_map  = deliveries[["match_id", "batting_team", "bowling_team"]].drop_duplicates()
    inn_stats = inn_stats.merge(bowl_map, on=["match_id", "batting_team"], how="left")
    inn_stats = inn_stats.merge(match_info, on="match_id", how="left").sort_values("date")

    inn_stats["hist_rr_scored"] = inn_stats.groupby("batting_team")["rr"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    conceded = inn_stats[["match_id", "bowling_team", "rr", "date"]].copy()
    conceded.columns = ["match_id", "team", "rr_conceded", "date"]
    conceded = conceded.sort_values("date")
    conceded["hist_rr_conceded"] = conceded.groupby("team")["rr_conceded"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    scored_dict   = inn_stats.set_index(["match_id", "batting_team"])["hist_rr_scored"].to_dict()
    conceded_dict = conceded.set_index(["match_id", "team"])["hist_rr_conceded"].to_dict()

    nrr_dict  = {}
    all_keys  = set(scored_dict.keys()) | set(conceded_dict.keys())
    for key in all_keys:
        s = scored_dict.get(key, np.nan)
        c = conceded_dict.get(key, np.nan)
        s = s if not pd.isna(s) else 8.0
        c = c if not pd.isna(c) else 8.0
        nrr_dict[key] = round(s - c, 4)

    return nrr_dict


def compute_rolling_venue_win_rate(matches):
    """Rolling venue-specific win rate per (team, venue) — fallback to overall."""
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"], "venue": row["venue"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"], "venue": row["venue"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")
    df["venue_wr"]   = df.groupby(["team", "venue"])["win"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["overall_wr"] = df.groupby("team")["win"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["venue_win_rate"] = df["venue_wr"].fillna(df["overall_wr"]).fillna(0.5)
    return df.set_index(["match_id", "team"])["venue_win_rate"].to_dict()


def compute_rolling_venue_avg_runs(deliveries, matches):
    """Rolling avg first innings score per venue."""
    first_inn    = deliveries[deliveries["inning"] == 1].copy()
    match_scores = first_inn.groupby("match_id")["total_runs"].sum().reset_index()
    match_scores.columns = ["match_id", "runs"]

    match_info   = matches[["id", "date", "venue"]].rename(columns={"id": "match_id"})
    match_scores = match_scores.merge(match_info, on="match_id", how="left").sort_values("date")

    match_scores["hist_venue_avg"] = match_scores.groupby("venue")["runs"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    match_scores["hist_venue_avg"] = match_scores["hist_venue_avg"].fillna(160)

    deduped = match_scores.drop_duplicates(subset=["match_id", "venue"], keep="first")
    return deduped.set_index(["match_id", "venue"])["hist_venue_avg"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2B — EWM Delivery Features
# ══════════════════════════════════════════════════════════════════════════════

def compute_ewm_delivery_features(deliveries, matches, span=10):
    """
    EWM-weighted rolling batting & bowling features.
    span=10 means last ~3-4 matches get ~50% of the weight.
    Uses shift(1) to prevent data leakage.
    """
    match_info = matches[["id", "date"]].rename(columns={"id": "match_id"})

    raw_batting = []
    raw_bowling = []

    for match_id, match_del in deliveries.groupby("match_id"):
        for inning in [1, 2]:
            inn = match_del[match_del["inning"] == inning]
            if len(inn) == 0:
                continue

            batting_team  = inn["batting_team"].iloc[0]
            bowling_team  = inn["bowling_team"].iloc[0]

            total_runs    = inn["total_runs"].sum()
            total_balls   = len(inn)
            overs_played  = inn["over"].nunique()
            run_rate      = total_runs / overs_played if overs_played > 0 else 0

            pp            = inn[inn["over"] <= 5]
            pp_runs       = pp["total_runs"].sum()
            pp_wickets    = int(pp["is_wicket"].sum())

            middle        = inn[(inn["over"] >= 6) & (inn["over"] <= 14)]
            middle_runs   = middle["total_runs"].sum()

            death         = inn[inn["over"] >= 15]
            death_runs    = death["total_runs"].sum()
            death_overs   = death["over"].nunique()
            death_rr      = death_runs / death_overs if death_overs > 0 else run_rate

            boundaries    = inn[inn["batsman_runs"].isin([4, 6])]
            boundary_pct  = len(boundaries) / total_balls if total_balls > 0 else 0

            dot_balls     = inn[inn["total_runs"] == 0]
            dot_ball_pct  = len(dot_balls) / total_balls if total_balls > 0 else 0

            batter_runs   = inn.groupby("batter")["batsman_runs"].sum().nlargest(3).index.tolist()
            top3_del      = inn[inn["batter"].isin(batter_runs)]
            top3_runs     = top3_del["batsman_runs"].sum()
            top3_sr       = (top3_runs / len(top3_del) * 100) if len(top3_del) > 0 else 0

            total_wickets = int(inn["is_wicket"].sum())
            bowling_sr    = total_balls / total_wickets if total_wickets > 0 else float(total_balls)

            raw_batting.append({
                "match_id": match_id, "team": batting_team,
                "total_runs": total_runs, "run_rate": run_rate,
                "pp_runs": pp_runs, "middle_runs": middle_runs,
                "death_runs": death_runs,
                "boundary_pct": boundary_pct, "dot_ball_pct": dot_ball_pct,
                "top3_sr": top3_sr,
            })

            raw_bowling.append({
                "match_id": match_id, "team": bowling_team,
                "economy": run_rate, "death_economy": death_rr,
                "pp_wickets": pp_wickets, "bowling_sr": bowling_sr,
            })

    bat_df  = pd.DataFrame(raw_batting).merge(match_info, on="match_id", how="left").sort_values("date")
    bowl_df = pd.DataFrame(raw_bowling).merge(match_info, on="match_id", how="left").sort_values("date")

    bat_cols  = ["total_runs", "run_rate", "pp_runs", "middle_runs", "death_runs",
                 "boundary_pct", "dot_ball_pct", "top3_sr"]
    bowl_cols = ["economy", "death_economy", "pp_wickets", "bowling_sr"]

    for col in bat_cols:
        bat_df[f"hist_{col}"] = bat_df.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(span=span, adjust=False, min_periods=1).mean()
        )

    for col in bowl_cols:
        bowl_df[f"hist_{col}"] = bowl_df.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(span=span, adjust=False, min_periods=1).mean()
        )

    bat_result  = {}
    bowl_result = {}

    for _, row in bat_df.iterrows():
        bat_result[(row["match_id"], row["team"])] = {
            "avg_runs"    : row["hist_total_runs"]   if not pd.isna(row["hist_total_runs"])   else BAT_DEFAULTS["avg_runs"],
            "run_rate"    : row["hist_run_rate"]      if not pd.isna(row["hist_run_rate"])      else BAT_DEFAULTS["run_rate"],
            "pp_runs"     : row["hist_pp_runs"]       if not pd.isna(row["hist_pp_runs"])       else BAT_DEFAULTS["pp_runs"],
            "middle_runs" : row["hist_middle_runs"]   if not pd.isna(row["hist_middle_runs"])   else BAT_DEFAULTS["middle_runs"],
            "death_runs"  : row["hist_death_runs"]    if not pd.isna(row["hist_death_runs"])    else BAT_DEFAULTS["death_runs"],
            "boundary_pct": row["hist_boundary_pct"]  if not pd.isna(row["hist_boundary_pct"])  else BAT_DEFAULTS["boundary_pct"],
            "dot_ball_pct": row["hist_dot_ball_pct"]  if not pd.isna(row["hist_dot_ball_pct"])  else BAT_DEFAULTS["dot_ball_pct"],
            "top3_sr"     : row["hist_top3_sr"]       if not pd.isna(row["hist_top3_sr"])       else BAT_DEFAULTS["top3_sr"],
        }

    for _, row in bowl_df.iterrows():
        bowl_result[(row["match_id"], row["team"])] = {
            "economy"      : row["hist_economy"]       if not pd.isna(row["hist_economy"])       else BOWL_DEFAULTS["economy"],
            "death_economy": row["hist_death_economy"] if not pd.isna(row["hist_death_economy"]) else BOWL_DEFAULTS["death_economy"],
            "pp_wickets"   : row["hist_pp_wickets"]    if not pd.isna(row["hist_pp_wickets"])    else BOWL_DEFAULTS["pp_wickets"],
            "bowling_sr"   : row["hist_bowling_sr"]    if not pd.isna(row["hist_bowling_sr"])    else BOWL_DEFAULTS["bowling_sr"],
        }

    return bat_result, bowl_result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2C — Chase Win Rate Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_chase_win_rate(matches):
    records = []
    for _, row in matches.iterrows():
        tw = row["toss_winner"]
        td = row["toss_decision"]
        t1 = row["team1"]
        t2 = row["team2"]

        if td == "field":
            chasing_team   = tw
        else:
            chasing_team   = t2 if tw == t1 else t1

        chaser_won = int(row["winner"] == chasing_team)

        records.append({
            "match_id"   : row["id"],
            "date"       : row["date"],
            "team"       : chasing_team,
            "chased_win" : chaser_won,
        })

    df = pd.DataFrame(records).sort_values("date")
    df["chase_win_rate"] = df.groupby("team")["chased_win"].transform(
        lambda x: x.shift(1).ewm(span=10, adjust=False, min_periods=1).mean()
    )
    df["chase_win_rate"] = df["chase_win_rate"].fillna(0.5)
    return df.set_index(["match_id", "team"])["chase_win_rate"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2D — Home Win Rate Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_home_win_rate(matches):
    records = []
    for _, row in matches.iterrows():
        for team in [row["team1"], row["team2"]]:
            if is_home_venue(team, row["venue"]):
                records.append({
                    "match_id" : row["id"],
                    "date"     : row["date"],
                    "team"     : team,
                    "home_win" : int(row["winner"] == team),
                })

    if not records:
        return {}

    df = pd.DataFrame(records).sort_values("date")
    df["home_win_rate"] = df.groupby("team")["home_win"].transform(
        lambda x: x.shift(1).ewm(span=10, adjust=False, min_periods=1).mean()
    )
    df["home_win_rate"] = df["home_win_rate"].fillna(0.5)
    return df.set_index(["match_id", "team"])["home_win_rate"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2E — Win Streak Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_win_streak(matches):
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")

    result = {}
    team_groups = df.groupby("team")

    for team, grp in team_groups:
        grp = grp.reset_index(drop=True)
        streak = 0
        for i, row_g in grp.iterrows():
            result[(row_g["match_id"], team)] = min(streak, 10)
            if row_g["win"] == 1:
                streak += 1
            else:
                streak = 0

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2F — Days Since Last Match (Rest/Fatigue)
# ══════════════════════════════════════════════════════════════════════════════

def compute_days_rest(matches):
    records = []
    for _, row in matches.iterrows():
        for team in [row["team1"], row["team2"]]:
            records.append({
                "match_id": row["id"],
                "date"    : row["date"],
                "team"    : team,
            })

    df = pd.DataFrame(records).sort_values("date")

    result = {}
    for team, grp in df.groupby("team"):
        grp       = grp.reset_index(drop=True)
        prev_date = None
        for _, row_g in grp.iterrows():
            if prev_date is None:
                days = 7
            else:
                days = (row_g["date"] - prev_date).days
            result[(row_g["match_id"], team)] = min(days, 30)
            prev_date = row_g["date"]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2G — Player-Level Squad Features (from deliveries history)
# ══════════════════════════════════════════════════════════════════════════════

def compute_squad_features(deliveries, matches):
    match_info = matches[["id", "date", "team1", "team2"]].copy()
    match_info["match_id"] = match_info["id"].astype(str)
    deliveries  = deliveries.copy()
    deliveries["match_id"] = deliveries["match_id"].astype(str)

    legal_bat  = deliveries[deliveries["extras_type"].fillna("") != "wides"]
    bat_career = legal_bat.groupby("batter").agg(
        bat_runs  = ("batsman_runs", "sum"),
        bat_balls = ("batsman_runs", "count"),
    ).reset_index()
    bat_career["career_bat_sr"] = (bat_career["bat_runs"] / bat_career["bat_balls"] * 100).round(1)
    bat_sr_dict = bat_career.set_index("batter")["career_bat_sr"].to_dict()

    legal_bowl = deliveries[~deliveries["extras_type"].isin(["wides", "noballs"])]
    bowl_career = legal_bowl.groupby("bowler").agg(
        bowl_runs  = ("total_runs", "sum"),
        bowl_balls = ("total_runs", "count"),
    ).reset_index()
    bowl_career["career_bowl_econ"] = (bowl_career["bowl_runs"] / (bowl_career["bowl_balls"] / 6)).round(2)
    bowl_econ_dict = bowl_career.set_index("bowler")["career_bowl_econ"].to_dict()

    batters_set    = set(bat_career[bat_career["bat_balls"] >= 30]["batter"])
    bowlers_set    = set(bowl_career[bowl_career["bowl_balls"] >= 30]["bowler"])
    allrounder_set = batters_set & bowlers_set

    bat_players = deliveries.groupby(["match_id", "batting_team"])["batter"].apply(set).reset_index()
    bat_players.columns = ["match_id", "team", "batters"]

    bowl_players = deliveries.groupby(["match_id", "bowling_team"])["bowler"].apply(set).reset_index()
    bowl_players.columns = ["match_id", "team", "bowlers"]

    team_players = bat_players.merge(bowl_players, on=["match_id", "team"], how="outer")
    team_players["batters"] = team_players["batters"].apply(lambda x: x if isinstance(x, set) else set())
    team_players["bowlers"] = team_players["bowlers"].apply(lambda x: x if isinstance(x, set) else set())
    team_players["all_players"] = team_players.apply(lambda r: r["batters"] | r["bowlers"], axis=1)

    date_map = matches.set_index("id")["date"].to_dict()
    team_players["date"] = team_players["match_id"].map(
        lambda x: date_map.get(x, date_map.get(int(x) if str(x).isdigit() else x, pd.NaT))
    )
    team_players = team_players.sort_values("date")

    result = {}

    for team, grp in team_players.groupby("team"):
        grp = grp.reset_index(drop=True)

        for i, row_g in grp.iterrows():
            mid = row_g["match_id"]

            past = grp.iloc[:i]
            last5 = past.tail(5)

            if len(last5) == 0:
                result[(mid, team)] = {
                    "squad_bat_sr"    : 120.0,
                    "squad_bowl_econ" : 8.5,
                    "squad_allrounder": 2,
                }
                continue

            recent_players = set()
            for _, pr in last5.iterrows():
                recent_players |= pr["all_players"]

            bat_srs = [bat_sr_dict[p] for p in recent_players if p in bat_sr_dict]
            squad_bat_sr = round(np.mean(bat_srs), 1) if bat_srs else 120.0

            bowl_econs = [bowl_econ_dict[p] for p in recent_players if p in bowl_econ_dict]
            squad_bowl_econ = round(np.mean(bowl_econs), 2) if bowl_econs else 8.5

            squad_allrounder = len(recent_players & allrounder_set)

            result[(mid, team)] = {
                "squad_bat_sr"    : squad_bat_sr,
                "squad_bowl_econ" : squad_bowl_econ,
                "squad_allrounder": squad_allrounder,
            }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2H — Season Stage Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_season_stage(matches):
    result = {}
    for season, grp in matches.groupby("season_int"):
        grp_sorted = grp.sort_values("date")
        total      = len(grp_sorted)
        playoff_start = max(total - 4, 0)
        for i, (_, row) in enumerate(grp_sorted.iterrows()):
            result[row["id"]] = 1 if i >= playoff_start else 0
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2I — NEW v5: Season Recency Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_season_recency(matches):
    """
    NEW (v5): How recent is each season on a 0–1 scale.
    Formula: 1 - ((max_season - season) / 20)
    Clipped to [0.05, 1.0] so old seasons don't go negative.

    Why: Tells the model explicitly that T20 cricket in 2025
    is a different game from 2008. Scores, strategies, powerplay
    use, death-over hitting — all evolved dramatically.

    2025 → 1.00  (fully modern)
    2022 → 0.85
    2018 → 0.65
    2012 → 0.35
    2008 → 0.15
    2007 → 0.10
    """
    max_season = matches["season_int"].max()
    result = {}
    for _, row in matches.iterrows():
        season = row["season_int"]
        recency = 1.0 - ((max_season - season) / 20.0)
        recency = float(np.clip(recency, 0.05, 1.0))
        result[row["id"]] = round(recency, 4)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2J — NEW v5: Season Average Runs (Era Anchor)
# ══════════════════════════════════════════════════════════════════════════════

def compute_season_avg_runs(deliveries, matches):
    """
    NEW (v5): Rolling IPL-wide average first-innings score by season.
    For each match, we compute the average first-innings total across
    ALL matches in ALL seasons BEFORE the current match's season.

    Why: If the model knows that the league-wide average score has
    jumped from 160 to 200, it can better calibrate its predictions.
    This is the 'era anchor' — a direct signal of scoring inflation.

    2008-era: ~150 avg  →  2024-era: ~180+ avg
    """
    first_inn    = deliveries[deliveries["inning"] == 1].copy()
    match_scores = first_inn.groupby("match_id")["total_runs"].sum().reset_index()
    match_scores.columns = ["match_id", "runs"]

    match_info   = matches[["id", "date", "season_int"]].rename(columns={"id": "match_id"})
    match_scores = match_scores.merge(match_info, on="match_id", how="left")

    # Season-level average
    season_avgs = match_scores.groupby("season_int")["runs"].mean().to_dict()

    # Rolling: for each season, compute avg of all PRIOR seasons
    all_seasons = sorted(season_avgs.keys())
    rolling_season_avg = {}
    cumsum = 0
    count  = 0
    for s in all_seasons:
        rolling_season_avg[s] = round(cumsum / count, 2) if count > 0 else 160.0
        cumsum += season_avgs[s]
        count  += 1

    # Map back to match level
    result = {}
    for _, row in matches.iterrows():
        season  = row["season_int"]
        mid     = row["id"]
        result[mid] = rolling_season_avg.get(season, 160.0)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2K — NEW v5: Form Velocity Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_form_velocity(matches):
    """
    NEW (v5): Win rate THIS season vs LAST season — detects rising/falling teams.

    form_velocity = this_season_win_rate - last_season_win_rate

    Examples:
      GT 2023: won 60% → in 2022 they won 71.4% (title) → velocity = -0.114 (declining)
      SRH 2024: massive improvement after poor 2023 → velocity = +0.30 (rising)
      MI 2023: poor after great 2022 → velocity negative (declining)

    Why: Your model currently cannot detect trends AT ALL.
    A team that won 30% last year but is winning 70% this year looks
    the same to XGBoost as a team that's been at 50% both years.
    This single feature fixes that blind spot.

    Uses only matches BEFORE the current one (no leakage).
    """
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"],
                         "season": row["season_int"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"],
                         "season": row["season_int"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values(["team", "date"])

    result = {}

    for team, grp in df.groupby("team"):
        grp = grp.reset_index(drop=True)

        # For each match, compute win rate in current season so far (prior matches only)
        # and win rate in the previous season
        for i, row_g in grp.iterrows():
            mid     = row_g["match_id"]
            season  = row_g["season"]
            prior   = grp.iloc[:i]  # all matches before this one for this team

            # This season so far (excluding current match)
            this_season_prior = prior[prior["season"] == season]
            this_wr = this_season_prior["win"].mean() if len(this_season_prior) > 0 else np.nan

            # Last season
            last_season = prior[prior["season"] == season - 1]
            last_wr = last_season["win"].mean() if len(last_season) > 0 else np.nan

            if pd.isna(this_wr) or pd.isna(last_wr):
                velocity = 0.0  # neutral when we have no data
            else:
                velocity = round(this_wr - last_wr, 4)

            result[(mid, team)] = velocity

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2L — NEW: Tournament Context Features (MATCH MOMENTUM & STANDINGS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_last_5_matches_win_pct(matches):
    """
    Win percentage of each team in their last 5 matches (rolling window).
    Captures recent MOMENTUM — teams on winning streaks vs losing slump.
    Uses shift(1) to prevent leakage.
    """
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"],
                        "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"],
                        "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values(["team", "date"])

    result = {}
    for team, grp in df.groupby("team"):
        grp = grp.reset_index(drop=True)
        # Rolling window of last 5 matches (shift by 1 to prevent leakage)
        last_5_pct = grp["win"].shift(1).rolling(window=5, min_periods=1).mean()
        for match_id, pct in zip(grp["match_id"], last_5_pct):
            result[(match_id, team)] = round(pct, 4) if not pd.isna(pct) else 0.5

    return result


def compute_last_3_h2h_result(matches):
    """
    Head-to-head win ratio from last 3 encounters between two teams.
    Captures H2H MOMENTUM and psychological dynamics.
    Uses only matches BEFORE the current one.
    """
    sorted_matches = matches.sort_values("date")
    records = []
    for _, row in sorted_matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        pair = tuple(sorted([t1, t2]))
        records.append({
            "match_id": row["id"], "date": row["date"],
            "team1": t1, "team2": t2, "pair": pair,
            "t1_won": int(row["winner"] == t1),
        })
    df = pd.DataFrame(records)

    result = {}
    pair_hist = {}

    for _, row in df.iterrows():
        mid, t1, t2, pair = row["match_id"], row["team1"], row["team2"], row["pair"]

        # Get last 3 H2H encounters BEFORE this match
        if pair not in pair_hist:
            h2h_win_pct = 0.5  # neutral if no prior H2H
        else:
            last_3_encounters = pair_hist[pair][-3:]  # last 3 matches
            if len(last_3_encounters) == 0:
                h2h_win_pct = 0.5
            else:
                t1_wins_in_last_3 = sum(1 for encounter in last_3_encounters if encounter[t1] == 1)
                h2h_win_pct = round(t1_wins_in_last_3 / len(last_3_encounters), 4)

        result[(mid, t1, t2)] = h2h_win_pct

        # Add this match result to history
        if pair not in pair_hist:
            pair_hist[pair] = []
        pair_hist[pair].append({t1: row["t1_won"], t2: (1 - row["t1_won"])})

    return result


def compute_current_season_points(matches):
    """
    Current season standings: points accumulated so far.
    Win = 2 points, Tie = 1 point (no ties in IPL), Loss = 0 points.
    Captures current TOURNAMENT POSITION and pressure.
    Uses only matches BEFORE the current one (no leakage).
    """
    records = []
    for _, row in matches.iterrows():
        season = row["season_int"]
        # Team 1
        t1_points = 2 if row["winner"] == row["team1"] else 0
        records.append({"match_id": row["id"], "season": season, "team": row["team1"], "points": t1_points})
        # Team 2
        t2_points = 2 if row["winner"] == row["team2"] else 0
        records.append({"match_id": row["id"], "season": season, "team": row["team2"], "points": t2_points})

    df = pd.DataFrame(records).sort_values(["season", "team"])
    result = {}

    for season in df["season"].unique():
        season_df = df[df["season"] == season].copy()
        for team in season_df["team"].unique():
            team_df = season_df[season_df["team"] == team].sort_values("match_id")
            # Cumulative points — shift(1) to get prior matches only
            team_df["cumulative_points"] = team_df["points"].shift(1).cumsum().fillna(0)
            for mid, pts in zip(team_df["match_id"], team_df["cumulative_points"]):
                result[(mid, team)] = int(pts)

    return result


def compute_current_season_nrr(matches, deliveries):
    """
    Net run rate for current season so far.
    Captures current season BATTING STRENGTH and dominance in tournament context.
    Uses only matches BEFORE the current one (no leakage).
    """
    match_info = matches[["id", "date", "season_int"]].rename(columns={"id": "match_id"})

    # Aggregate runs per match per team
    inn_stats = deliveries.groupby(["match_id", "batting_team"]).agg(
        runs=("total_runs", "sum"),
        overs=("over", "nunique")
    ).reset_index()
    inn_stats["rr"] = inn_stats["runs"] / inn_stats["overs"].clip(lower=1)

    # Add match info
    bowl_map = deliveries[["match_id", "batting_team", "bowling_team"]].drop_duplicates()
    inn_stats = inn_stats.merge(bowl_map, on=["match_id", "batting_team"], how="left")
    inn_stats = inn_stats.merge(match_info, on="match_id", how="left").sort_values("date")

    # Season-based run rate (cumulative for this season so far)
    inn_stats_scored = inn_stats[["match_id", "batting_team", "season_int", "rr", "date"]].copy()
    inn_stats_scored.columns = ["match_id", "team", "season", "rr_scored", "date"]
    inn_stats_scored = inn_stats_scored.sort_values("date")

    # Scoring: cumulative average for current season (prior matches only)
    inn_stats_scored["season_rr_scored"] = inn_stats_scored.groupby(["season", "team"])["rr_scored"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Bowling: runs conceded
    conceded = inn_stats[["match_id", "bowling_team", "season_int", "rr", "date"]].copy()
    conceded.columns = ["match_id", "team", "season", "rr_conceded", "date"]
    conceded = conceded.sort_values("date")
    conceded["season_rr_conceded"] = conceded.groupby(["season", "team"])["rr_conceded"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    scored_dict = inn_stats_scored.set_index(["match_id", "team"])["season_rr_scored"].to_dict()
    conceded_dict = conceded.set_index(["match_id", "team"])["season_rr_conceded"].to_dict()

    nrr_dict = {}
    all_keys = set(scored_dict.keys()) | set(conceded_dict.keys())
    for key in all_keys:
        s = scored_dict.get(key, np.nan)
        c = conceded_dict.get(key, np.nan)
        s = s if not pd.isna(s) else 8.0
        c = c if not pd.isna(c) else 8.0
        nrr_dict[key] = round(s - c, 4)

    return nrr_dict


# ══════════════════════════════════════════════════════════════════════════════
# MATCH CONTEXT FEATURES (NEW) ——Tournament-aware signals
# ══════════════════════════════════════════════════════════════════════════════

def compute_last5_win_pct(matches):
    """
    Win percentage in last 5 matches (rolling window).
    Captures current momentum/form in tournament.
    """
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")

    # Rolling 5-match window (no leakage: use shift(1))
    df["last5_win_pct"] = df.groupby("team")["win"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    df["last5_win_pct"] = df["last5_win_pct"].fillna(0.5)

    return df.set_index(["match_id", "team"])["last5_win_pct"].to_dict()


def compute_last3_h2h(matches):
    """
    Head-to-head result in last 3 encounters.
    Returns team1's win percentage in last 3 H2H matches.
    Captures direct competitive dynamics.
    """
    sorted_matches = matches.sort_values("date")
    result = {}

    for _, row in sorted_matches.iterrows():
        mid = row["id"]
        t1, t2 = row["team1"], row["team2"]
        pair = tuple(sorted([t1, t2]))

        # Find last 3 H2H matches before this date
        prior = sorted_matches[
            (sorted_matches["date"] < row["date"]) &
            (
                ((sorted_matches["team1"] == t1) & (sorted_matches["team2"] == t2)) |
                ((sorted_matches["team1"] == t2) & (sorted_matches["team2"] == t1))
            )
        ].tail(3)

        if len(prior) == 0:
            result[(mid, t1, t2)] = 0.5  # Default if no prior H2H
        else:
            t1_wins = sum(
                (prior["team1"] == t1) & (prior["winner"] == t1) |
                (prior["team2"] == t1) & (prior["winner"] == t1)
            )
            result[(mid, t1, t2)] = round(t1_wins / len(prior), 4)

    return result


def compute_current_season_points(matches):
    """
    Points table value for current season.
    2 points for win, 0 for loss (no-result = 1, but we filtered those out).
    Tracks tournament standing momentum.
    """
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"], "season": row["season_int"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"], "season": row["season_int"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")

    # Points: 2 per win, 0 for loss (cumulative per season, prior matches only)
    df["points"] = df["win"] * 2
    df["season_points"] = df.groupby(["season", "team"])["points"].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )

    return df.set_index(["match_id", "team"])["season_points"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# PLAYING XI FEATURES — GAME CHANGER
# ══════════════════════════════════════════════════════════════════════════════

def load_player_stats():
    """Load pre-computed player-level statistics."""
    bat_path  = os.path.join(DATA_DIR, "player_batting_stats.csv")
    bowl_path = os.path.join(DATA_DIR, "player_bowling_stats.csv")

    bat_stats = pd.read_csv(bat_path, dtype={"player": str}) if os.path.exists(bat_path) else pd.DataFrame()
    bowl_stats = pd.read_csv(bowl_path, dtype={"player": str}) if os.path.exists(bowl_path) else pd.DataFrame()

    return bat_stats, bowl_stats


def extract_playing_xi(deliveries, match_id, batting_team, bowling_team):
    """
    Extract actual 11 players from a match inning.
    Returns (batsmen, bowlers) lists with actual players.

    Note: In T20, exactly 11 per side, but missing data may have fewer.
    """
    match_del = deliveries[deliveries["match_id"] == match_id].copy()

    batsmen = set()
    bowlers = set()

    # Extract from first inning (batting team 1)
    inn1 = match_del[(match_del["inning"] == 1) & (match_del["batting_team"] == batting_team)]
    if len(inn1) > 0:
        batsmen.update(inn1["batter"].dropna().unique())
        batsmen.update(inn1["non_striker"].dropna().unique())
        bowlers.update(inn1["bowler"].dropna().unique())

    # Extract from second inning if needed
    inn2 = match_del[(match_del["inning"] == 2) & (match_del["batting_team"] == batting_team)]
    if len(inn2) > 0:
        batsmen.update(inn2["batter"].dropna().unique())
        batsmen.update(inn2["non_striker"].dropna().unique())
        bowlers.update(inn2["bowler"].dropna().unique())

    return list(batsmen), list(bowlers)


def compute_playing11_features(deliveries, matches, bat_stats, bowl_stats):
    """
    Compute 8 playing XI features per match:
    - team1/2_playing11_bat_sr: Average strike rate of XI batsmen
    - team1/2_playing11_bowl_econ: Average economy of XI bowlers
    - team1/2_num_death_bowlers: Count of death specialists
    - team1/2_num_pp_bowlers: Count of powerplay specialists

    Uses shift(1) for no leakage - only considers historical player stats.
    """
    # Create lookup dictionaries for player stats (case-insensitive)
    bat_lookup = {}
    for _, row in bat_stats.iterrows():
        player = str(row["player"]).strip().lower()
        bat_lookup[player] = {
            "strike_rate": float(row.get("strike_rate", 125.0)),
            "batting_avg": float(row.get("batting_avg", 25.0)),
        }

    bowl_lookup = {}
    for _, row in bowl_stats.iterrows():
        player = str(row["player"]).strip().lower()
        bowl_lookup[player] = {
            "economy": float(row.get("economy", 8.5)),
            "death_economy": float(row.get("death_economy", 9.5)),
            "pp_economy": float(row.get("pp_economy", 8.0)),
            "pp_wickets": float(row.get("pp_wickets", 0.0)),
            "death_wickets": float(row.get("death_wickets", 0.0)),
        }

    result = {}

    for _, match_row in matches.iterrows():
        match_id = match_row["id"]
        t1, t2 = match_row["team1"], match_row["team2"]

        # Extract XI for both teams
        t1_batsmen, t1_bowlers = extract_playing_xi(deliveries, match_id, t1, t2)
        t2_batsmen, t2_bowlers = extract_playing_xi(deliveries, match_id, t2, t1)

        # Compute batting strike rates
        def avg_bat_sr(batsman_list, lookup):
            if not batsman_list:
                return 125.0  # Default SR
            srs = []
            for b in batsman_list:
                b_norm = str(b).strip().lower()
                sr = lookup.get(b_norm, {}).get("strike_rate", 125.0)
                if sr > 0:  # Valid SR
                    srs.append(sr)
            return np.mean(srs) if srs else 125.0

        # Compute bowling economies
        def avg_bowl_econ(bowler_list, lookup):
            if not bowler_list:
                return 8.5  # Default economy
            ecns = []
            for b in bowler_list:
                b_norm = str(b).strip().lower()
                econ = lookup.get(b_norm, {}).get("economy", 8.5)
                if econ > 0:  # Valid economy
                    ecns.append(econ)
            return np.mean(ecns) if ecns else 8.5

        # Count death specialists (death_economy < 9.5)
        def count_death_bowlers(bowler_list, lookup):
            count = 0
            for b in bowler_list:
                b_norm = str(b).strip().lower()
                death_econ = lookup.get(b_norm, {}).get("death_economy", 9.5)
                if death_econ < 9.5:
                    count += 1
            return count

        # Count PP specialists (pp_economy < 8.0)
        def count_pp_bowlers(bowler_list, lookup):
            count = 0
            for b in bowler_list:
                b_norm = str(b).strip().lower()
                pp_econ = lookup.get(b_norm, {}).get("pp_economy", 8.0)
                if pp_econ < 8.0:
                    count += 1
            return count

        # Compute for team1
        result[(match_id, t1)] = {
            "playing11_bat_sr": round(avg_bat_sr(t1_batsmen, bat_lookup), 2),
            "playing11_bowl_econ": round(avg_bowl_econ(t1_bowlers, bowl_lookup), 2),
            "num_death_bowlers": count_death_bowlers(t1_bowlers, bowl_lookup),
            "num_pp_bowlers": count_pp_bowlers(t1_bowlers, bowl_lookup),
        }

        # Compute for team2
        result[(match_id, t2)] = {
            "playing11_bat_sr": round(avg_bat_sr(t2_batsmen, bat_lookup), 2),
            "playing11_bowl_econ": round(avg_bowl_econ(t2_bowlers, bowl_lookup), 2),
            "num_death_bowlers": count_death_bowlers(t2_bowlers, bowl_lookup),
            "num_pp_bowlers": count_pp_bowlers(t2_bowlers, bowl_lookup),
        }

    return result




# ══════════════════════════════════════════════════════════════════════════════
# VENUE & PITCH INTELLIGENCE FEATURES (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def compute_pitch_type(matches):
    """
    Classify pitch type per venue based on first innings scoring patterns.

    0 = Flat (high-scoring, easy batting)      e.g., Bangalore, Mohali
    1 = Slow (assists spinners, tough batting) e.g., Chennai, Delhi
    2 = Spin (spin-friendly, variable)         e.g., Chepauk
    3 = Pace (pace-friendly, bouncy)           e.g., Jaipur

    Uses rolling average first innings runs grouped by venue.
    High avg runs → flat. Low runs + many spinners → spin-friendly.
    """
    result = {}

    # Get first innings scores per venue
    first_inn = matches.copy()

    # Use pre-defined characteristics where available
    for venue in first_inn["venue"].unique():
        if venue in VENUE_CHARACTERISTICS:
            result[venue] = VENUE_CHARACTERISTICS[venue]["pitch_type"]
        else:
            # Default: slow (conservative estimate for unknown venues)
            result[venue] = VENUE_DEFAULTS["pitch_type"]

    # Return as dict for each match
    venue_dict = {}
    for _, row in matches.iterrows():
        venue = row["venue"]
        venue_dict[row["id"]] = result.get(venue, VENUE_DEFAULTS["pitch_type"])

    return venue_dict


def compute_dew_factor(matches):
    """
    Compute dew factor per match: 1 if night match, 0 otherwise.

    Dew reduces grip for bowlers, increases difficulty, favors chasers.
    Uses time information from match schedules (if available).
    Falls back to date-based heuristics for IPL night patterns.

    IPL patterns: Most evening/night matches are post-5pm
    We approximate: afternoon=0, evening/night=1
    """
    result = {}

    # If match has explicit time info, use it
    # Otherwise, use date-based heuristics (IPL typically has night matches)
    for _, row in matches.iterrows():
        mid = row["id"]
        # For now: assume dew factor varies by date/season
        # Conservative: set to 1 if season >= 2015 (when night matches became common)
        # And season is modern IPL era
        season = row["season"].astype(str).split('-')[0] if isinstance(row["season"], str) else row["season"]
        try:
            season_year = int(season[:4]) if len(str(season)) >= 4 else int(season)
        except:
            season_year = 2020

        # Most modern IPL matches (90%) are played at evening/night
        # Set dew_factor = 1 for high-probability night matches
        dew = 1 if season_year >= 2015 and row["date"].month in [9, 10, 3, 4, 5] else 0
        result[mid] = dew

    return result


def compute_boundary_size(matches):
    """
    Compute relative boundary size per venue: 0=small, 1=medium, 2=large.

    Small: ~65 yards (e.g., M Chinnaswamy - short boundaries) → high-scoring
    Medium: ~70 yards (standard) → balanced
    Large: ~75+ yards (e.g., Narendra Modi - long boundaries) → low-scoring

    Uses pre-defined venue characteristics. Defaults to medium.
    """
    result = {}

    for _, row in matches.iterrows():
        venue = row["venue"]
        if venue in VENUE_CHARACTERISTICS:
            boundary_sz = VENUE_CHARACTERISTICS[venue]["boundary_size"]
        else:
            boundary_sz = VENUE_DEFAULTS["boundary_size"]
        result[row["id"]] = boundary_sz

    return result


def compute_avg_chasing_success(matches):
    """
    Compute average success rate of chasing teams at each venue.

    Identifies venues where chasing teams have advantage (>50%) vs disadvantage (<50%).
    Uses rolling cumulative win rate (prior matches only, no leakage).

    Returns: rolling average chase win % at venue
    """
    # Build chase records per venue
    records = []
    for _, row in matches.iterrows():
        tw = row["toss_winner"]
        td = row["toss_decision"]
        t1 = row["team1"]
        t2 = row["team2"]

        # Determine chasing team
        if td == "field":
            chasing_team = tw
        else:
            chasing_team = t2 if tw == t1 else t1

        chaser_won = int(row["winner"] == chasing_team)

        records.append({
            "match_id": row["id"],
            "date": row["date"],
            "venue": row["venue"],
            "chaser_won": chaser_won,
        })

    df = pd.DataFrame(records).sort_values("date")

    # Compute rolling chase win rate per venue (expanding, prior matches only)
    df["chase_win_pct"] = df.groupby("venue")["chaser_won"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["chase_win_pct"] = df["chase_win_pct"].fillna(VENUE_DEFAULTS["avg_chasing_success"])

    return df.set_index("match_id")["chase_win_pct"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# TOSS INTELLIGENCE FEATURES (NEW) — VENUE-AWARE, NOT RANDOM
# ══════════════════════════════════════════════════════════════════════════════

def compute_toss_advantage_at_venue(matches):
    """
    Measure how much winning the toss actually impacts win probability at each venue.

    Logic:
    - For each venue, track: (toss_winner_wins / toss_winner_matches) - 0.5
    - This isolates toss impact separate from other factors
    - Rolling average (prior matches only, no leakage)
    - Values near 0 = toss irrelevant; near ±0.2 = toss very important

    Returns: dict mapping match_id → toss_advantage_at_venue
    """
    records = []
    for _, row in matches.iterrows():
        tw = row["toss_winner"]
        t1 = row["team1"]
        t2 = row["team2"]

        toss_winner_won = int(row["winner"] == tw)

        records.append({
            "match_id": row["id"],
            "date": row["date"],
            "venue": row["venue"],
            "toss_winner_won": toss_winner_won,
        })

    df = pd.DataFrame(records).sort_values("date")

    # Compute rolling toss advantage per venue (prior matches only)
    # Advantage = (toss_winner_win_rate - 0.5) → this isolates toss impact
    df["toss_win_rate"] = df.groupby("venue")["toss_winner_won"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Compute advantage: how much better does toss winner do vs random (0.5)?
    # 0.55 win rate → +0.05 toss advantage
    # 0.45 win rate → -0.05 toss disadvantage
    df["toss_advantage"] = df["toss_win_rate"] - 0.5
    df["toss_advantage"] = df["toss_advantage"].fillna(0.0)

    return df.set_index("match_id")["toss_advantage"].to_dict()


def compute_team_chasing_strength(matches):
    """
    For each team, compute their historical chase win rate.

    This captures HOW GOOD each team is at chasing independently of venue.
    - Team that wins 60% of chases vs 40% of chases after batting first
    - Rolling average (prior matches only)

    Returns: dict mapping (match_id, team) → chase_win_rate
    """
    records = []
    for _, row in matches.iterrows():
        tw = row["toss_winner"]
        td = row["toss_decision"]
        t1 = row["team1"]
        t2 = row["team2"]

        # Determine chasing team
        if td == "field":
            chasing_team = tw
        else:
            chasing_team = t2 if tw == t1 else t1

        chaser_won = int(row["winner"] == chasing_team)

        records.append({
            "match_id": row["id"],
            "date": row["date"],
            "team": chasing_team,
            "chased_won": chaser_won,
        })

    df = pd.DataFrame(records).sort_values("date")

    # Rolling chase win rate per team (prior matches only, EWM for recency)
    df["chase_win_rate"] = df.groupby("team")["chased_won"].transform(
        lambda x: x.shift(1).ewm(span=10, adjust=False, min_periods=1).mean()
    )
    df["chase_win_rate"] = df["chase_win_rate"].fillna(0.5)

    return df.set_index(["match_id", "team"])["chase_win_rate"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — MOMENTUM FEATURES (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def compute_last_3_match_avg_runs(deliveries, matches):
    """
    For each team at each match, compute average runs scored in last 3 matches (before now).
    Captures recent batting momentum (HOT vs COLD teams).
    Uses shift(1) to prevent data leakage.

    Returns: dict mapping (match_id, team) → avg_runs_last_3
    Default for <3 matches: use partial data or fall back to 155 (league average)
    """
    # Extract first inning scores per match
    first_inn = deliveries[deliveries["inning"] == 1].copy()
    match_scores = first_inn.groupby(["match_id", "batting_team"]).agg(
        runs=("total_runs", "sum")
    ).reset_index()

    match_scores.columns = ["match_id", "team", "runs"]

    # Add match date info for sorting
    match_info = matches[["id", "date"]].rename(columns={"id": "match_id"})
    match_scores = match_scores.merge(match_info, on="match_id", how="left").sort_values("date")

    result = {}

    for team, grp in match_scores.groupby("team"):
        grp = grp.reset_index(drop=True)

        for i, row_g in grp.iterrows():
            mid = row_g["match_id"]

            # Get last 3 matches BEFORE this one
            prior = grp.iloc[:i].tail(3)

            if len(prior) == 0:
                avg_runs = 155.0  # Default league average
            else:
                avg_runs = round(prior["runs"].mean(), 2)

            result[(mid, team)] = avg_runs

    return result


def compute_last_3_match_wickets_taken(deliveries, matches):
    """
    For each team at each match, compute average wickets taken in last 3 matches (before now).
    Captures recent bowling momentum and form (HOT vs COLD teams defensively).
    Uses shift(1) to prevent data leakage.

    Returns: dict mapping (match_id, team) → avg_wickets_last_3
    Default for <3 matches: use partial data or fall back to 6.0 (typical wickets/match)
    """
    # Extract wickets per match per bowling team
    match_wickets = deliveries[deliveries["is_wicket"] == True].copy()
    wicket_counts = match_wickets.groupby(["match_id", "inning", "bowling_team"]).size().reset_index(name="wickets")

    # Aggregate per match (both innings)
    match_bowled = deliveries.groupby(["match_id", "bowling_team"]).apply(
        lambda g: len(g[g["is_wicket"] == True])
    ).reset_index(name="total_wickets")

    # Add match date info
    match_info = matches[["id", "date"]].rename(columns={"id": "match_id"})
    match_bowled = match_bowled.merge(match_info, on="match_id", how="left").sort_values("date")

    match_bowled.columns = ["match_id", "team", "wickets", "date"]

    result = {}

    for team, grp in match_bowled.groupby("team"):
        grp = grp.reset_index(drop=True)

        for i, row_g in grp.iterrows():
            mid = row_g["match_id"]

            # Get last 3 matches BEFORE this one
            prior = grp.iloc[:i].tail(3)

            if len(prior) == 0:
                avg_wickets = 6.0  # Default (10 wickets max, so ~6 per team per match)
            else:
                avg_wickets = round(prior["wickets"].mean(), 2)

            result[(mid, team)] = avg_wickets

    return result


def compute_form_trend_slope(matches):
    """
    For each team at each match, compute trend slope of recent wins/performance over last 5 matches.
    Positive slope = improving team, Negative = declining.
    Uses EWM to weight recent matches more heavily.

    Captures momentum momentum: are they on an upswing or downswing?
    Uses shift(1) to prevent data leakage.

    Returns: dict mapping (match_id, team) → trend_slope
    Range: roughly -1 (all losses) to +1 (all wins)
    """
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"],
                        "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"],
                        "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")

    result = {}

    for team, grp in df.groupby("team"):
        grp = grp.reset_index(drop=True)

        for i, row_g in grp.iterrows():
            mid = row_g["match_id"]

            # Get last 5 matches BEFORE this one
            prior = grp.iloc[:i].tail(5)

            if len(prior) < 2:
                # Not enough data to compute slope
                slope = 0.0
            else:
                # Simple slope: fit line to (index → wins)
                # Convert to numpy for linear regression
                x = np.arange(len(prior))
                y = prior["win"].values.astype(float)

                # Linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
                n = len(x)
                slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
                slope = round(float(slope), 4)

            result[(mid, team)] = slope

    return result


def build_master_features(matches, deliveries):
    print("\nComputing all features (rolling, no leakage)...")

    print("  [1/15] Rolling team win rates...")
    win_rates = compute_rolling_win_rate(matches)

    print("  [2/15] Rolling recent form EWM (last 5 matches)...")
    recent_form = compute_rolling_recent_form(matches)

    print("  [3/15] Rolling head-to-head win rates...")
    h2h = compute_rolling_h2h(matches)

    print("  [4/15] Rolling NRR (from deliveries)...")
    nrr = compute_rolling_nrr(matches, deliveries)

    print("  [5/15] Rolling venue win rates...")
    venue_win_rates = compute_rolling_venue_win_rate(matches)

    print("  [6/15] Rolling venue average runs...")
    venue_avg_runs = compute_rolling_venue_avg_runs(deliveries, matches)

    print("  [7/15] EWM batting & bowling features (span=10)...")
    bat_feats, bowl_feats = compute_ewm_delivery_features(deliveries, matches)

    print("  [8/15] Rolling chase win rates (EWM)...")
    chase_win_rates = compute_rolling_chase_win_rate(matches)

    print("  [9/15] Rolling home win rates...")
    home_win_rates = compute_rolling_home_win_rate(matches)

    print("  [10/15] Win streaks...")
    win_streaks = compute_win_streak(matches)

    print("  [11/15] Days rest between matches...")
    days_rest = compute_days_rest(matches)

    print("  [12/15] Player-level squad features (last 5 matches)...")
    squad_feats = compute_squad_features(deliveries, matches)

    season_stage = compute_season_stage(matches)

    print("  [13/15] NEW: Season recency (era position signal)...")
    season_recency = compute_season_recency(matches)

    print("  [14/15] NEW: Season average runs (era anchor)...")
    season_avg_runs = compute_season_avg_runs(deliveries, matches)

    print("  [15/15] NEW: Form velocity (rising/falling team detection)...")
    form_velocity = compute_form_velocity(matches)

    print("  [16/19] MATCH CONTEXT: Last 5 match win %...")
    last5_win_pct = compute_last5_win_pct(matches)

    print("  [17/19] MATCH CONTEXT: Last 3 H2H result...")
    last3_h2h = compute_last3_h2h(matches)

    print("  [18/19] MATCH CONTEXT: Current season points...")
    season_points = compute_current_season_points(matches)

    print("  [19/19] MATCH CONTEXT: Current season NRR...")
    season_nrr = compute_current_season_nrr(matches, deliveries)

    print("  [20/22] PLAYING XI: Loading player stats...")
    bat_stats, bowl_stats = load_player_stats()

    print("  [21/22] PLAYING XI: Extracting team compositions...")
    playing11_feats = compute_playing11_features(deliveries, matches, bat_stats, bowl_stats)

    print("  [22/22] PLAYING XI: Features computed")

    print("  [23/26] VENUE: Pitch type classification...")
    pitch_types = compute_pitch_type(matches)

    print("  [24/26] VENUE: Dew factor (night/day)...")
    dew_factors = compute_dew_factor(matches)

    print("  [25/26] VENUE: Boundary size categorization...")
    boundary_sizes = compute_boundary_size(matches)

    print("  [26/26] VENUE: Average chasing success %...")
    chasing_success = compute_avg_chasing_success(matches)

    print("  [27/29] TOSS INTELLIGENCE: Toss advantage at venue...")
    toss_advantage = compute_toss_advantage_at_venue(matches)

    print("  [28/29] TOSS INTELLIGENCE: Team chasing strength...")
    team_chasing_str = compute_team_chasing_strength(matches)

    print("  [29/29] TOSS INTELLIGENCE: Features computed")

    print("  [30/32] MOMENTUM: Last 3 match avg runs...")
    last_3_avg_runs = compute_last_3_match_avg_runs(deliveries, matches)

    print("  [31/32] MOMENTUM: Last 3 match wickets taken...")
    last_3_wickets = compute_last_3_match_wickets_taken(deliveries, matches)

    print("  [32/32] MOMENTUM: Form trend slope...")
    form_trend = compute_form_trend_slope(matches)


    print("\nAssembling master feature rows...")
    rows = []

    for _, row in matches.iterrows():
        mid   = row["id"]
        t1    = row["team1"]
        t2    = row["team2"]
        venue = row["venue"]
        mid_str = str(mid)

        t1_bat  = bat_feats.get((mid, t1), {})
        t2_bat  = bat_feats.get((mid, t2), {})
        t1_bowl = bowl_feats.get((mid, t1), {})
        t2_bowl = bowl_feats.get((mid, t2), {})

        t1_squad = squad_feats.get((mid_str, t1), squad_feats.get((mid, t1), {}))
        t2_squad = squad_feats.get((mid_str, t2), squad_feats.get((mid, t2), {}))

        # ── REMOVED WEAK TOSS FEATURES ──
        # toss_win, toss_field, toss_team1_field are NOT computed anymore
        # They were random signals with no venue context

        target = int(row["winner"] == t1)

        t1_sq_bat_sr  = t1_squad.get("squad_bat_sr",     120.0)
        t2_sq_bat_sr  = t2_squad.get("squad_bat_sr",     120.0)
        t1_sq_econ    = t1_squad.get("squad_bowl_econ",   8.5)
        t2_sq_econ    = t2_squad.get("squad_bowl_econ",   8.5)
        t1_sq_ar      = t1_squad.get("squad_allrounder",  2)
        t2_sq_ar      = t2_squad.get("squad_allrounder",  2)

        # NEW v5 features
        s_recency   = season_recency.get(mid, 0.5)
        s_avg_runs  = season_avg_runs.get(mid, 160.0)
        t1_vel      = form_velocity.get((mid, t1), 0.0)
        t2_vel      = form_velocity.get((mid, t2), 0.0)

        # NEW MATCH CONTEXT features (tournament-aware)
        t1_last5    = last5_win_pct.get((mid, t1), 0.5)
        t2_last5    = last5_win_pct.get((mid, t2), 0.5)
        h2h_t1      = last3_h2h.get((mid, t1, t2), 0.5)
        t1_pts      = season_points.get((mid, t1), 0.0)
        t2_pts      = season_points.get((mid, t2), 0.0)
        t1_s_nrr    = season_nrr.get((mid, t1), 0.0)
        t2_s_nrr    = season_nrr.get((mid, t2), 0.0)

        # NEW PLAYING XI features (actual match composition)
        t1_xi_feats = playing11_feats.get((mid, t1), {})
        t2_xi_feats = playing11_feats.get((mid, t2), {})

        t1_bat_sr       = t1_xi_feats.get("playing11_bat_sr", 125.0)
        t2_bat_sr       = t2_xi_feats.get("playing11_bat_sr", 125.0)
        t1_bowl_econ    = t1_xi_feats.get("playing11_bowl_econ", 8.5)
        t2_bowl_econ    = t2_xi_feats.get("playing11_bowl_econ", 8.5)
        t1_death_bowlers = t1_xi_feats.get("num_death_bowlers", 1)
        t2_death_bowlers = t2_xi_feats.get("num_death_bowlers", 1)
        t1_pp_bowlers    = t1_xi_feats.get("num_pp_bowlers", 1)
        t2_pp_bowlers    = t2_xi_feats.get("num_pp_bowlers", 1)

        # NEW VENUE & PITCH INTELLIGENCE features
        pitch_type      = pitch_types.get(mid, VENUE_DEFAULTS["pitch_type"])
        dew_factor      = dew_factors.get(mid, VENUE_DEFAULTS["dew_factor"])
        boundary_size   = boundary_sizes.get(mid, VENUE_DEFAULTS["boundary_size"])
        avg_chase_succ  = chasing_success.get(mid, VENUE_DEFAULTS["avg_chasing_success"])

        # NEW TOSS INTELLIGENCE features (venue-aware, context-driven)
        # Toss only matters when it unlocks venue advantage
        toss_adv        = toss_advantage.get(mid, 0.0)     # How much toss win helps at this venue
        t1_chase_str    = team_chasing_str.get((mid, t1), 0.5)  # Team 1's chase win rate
        t2_chase_str    = team_chasing_str.get((mid, t2), 0.5)  # Team 2's chase win rate

        # NEW MOMENTUM features (HOT vs COLD detection)
        t1_last_3_runs  = last_3_avg_runs.get((mid, t1), 155.0)  # Team 1's avg runs last 3 matches
        t2_last_3_runs  = last_3_avg_runs.get((mid, t2), 155.0)  # Team 2's avg runs last 3 matches
        t1_last_3_wkts  = last_3_wickets.get((mid, t1), 6.0)    # Team 1's avg wickets last 3 matches
        t2_last_3_wkts  = last_3_wickets.get((mid, t2), 6.0)    # Team 2's avg wickets last 3 matches
        t1_form_slope   = form_trend.get((mid, t1), 0.0)        # Team 1's form trend (positive = improving)
        t2_form_slope   = form_trend.get((mid, t2), 0.0)        # Team 2's form trend (positive = improving)

        # CRITICAL NEW SCORING STRENGTH features (from bat_feats - already computed!)
        t1_death_runs   = t1_bat.get("death_runs", BAT_DEFAULTS["death_runs"])    # Team 1's death overs runs
        t2_death_runs   = t2_bat.get("death_runs", BAT_DEFAULTS["death_runs"])    # Team 2's death overs runs
        t1_pp_runs      = t1_bat.get("pp_runs", BAT_DEFAULTS["pp_runs"])          # Team 1's powerplay runs
        t2_pp_runs      = t2_bat.get("pp_runs", BAT_DEFAULTS["pp_runs"])          # Team 2's powerplay runs
        t1_run_rate     = t1_bat.get("run_rate", BAT_DEFAULTS["run_rate"])        # Team 1's run rate
        t2_run_rate     = t2_bat.get("run_rate", BAT_DEFAULTS["run_rate"])        # Team 2's run rate
        t1_avg_runs     = t1_bat.get("avg_runs", BAT_DEFAULTS["avg_runs"])        # Team 1's avg runs
        t2_avg_runs     = t2_bat.get("avg_runs", BAT_DEFAULTS["avg_runs"])        # Team 2's avg runs

        # VENUE SCORING CONTEXT (rolling first innings avg at venue)
        v_avg_runs      = venue_avg_runs.get((mid, venue), 160.0)

        # OPPONENT WEAKNESSES (for interaction features)
        t2_death_econ   = t2_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"])  # Team 2's death economy
        t2_pp_wkts      = t2_bowl.get("pp_wickets", BOWL_DEFAULTS["pp_wickets"])        # Team 2's PP wickets

        feature_row = {
            # ── Identifiers ───────────────────────────────────────────────────
            "match_id" : mid,
            "date"     : row["date"],
            "season"   : row["season_int"],
            "team1"    : t1,
            "team2"    : t2,
            "venue"    : venue,

            # ── BASE FEATURES (4) ───────────────────────────────────────────────
            "team1_death_economy"   : round(t1_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"]), 4),
            "season_recency"        : s_recency,
            "team1_win_rate"        : round(win_rates.get((mid, t1), 0.5), 4),
            "team2_squad_bowl_econ" : round(t2_sq_econ, 2),

            # ── MATCH CONTEXT FEATURES (TOURNAMENT-AWARE) (5) ──────────────────
            "team1_last5_win_pct"       : round(t1_last5, 4),
            "team2_last5_win_pct"       : round(t2_last5, 4),
            "last3_h2h_team1_pct"       : round(h2h_t1, 4),
            "current_season_points_diff": round(t1_pts - t2_pts, 2),
            "current_season_nrr_diff"   : round(t1_s_nrr - t2_s_nrr, 4),

            # ── PLAYING XI FEATURES (GAME CHANGER) (8) ───────────────────────────
            "team1_playing11_bat_sr"    : round(t1_bat_sr, 2),
            "team2_playing11_bat_sr"    : round(t2_bat_sr, 2),
            "team1_playing11_bowl_econ" : round(t1_bowl_econ, 2),
            "team2_playing11_bowl_econ" : round(t2_bowl_econ, 2),
            "team1_num_death_bowlers"   : int(t1_death_bowlers),
            "team2_num_death_bowlers"   : int(t2_death_bowlers),
            "team1_num_pp_bowlers"      : int(t1_pp_bowlers),
            "team2_num_pp_bowlers"      : int(t2_pp_bowlers),

            # ── VENUE & PITCH INTELLIGENCE (HIDDEN GOLD) (5) ────────────────────
            "pitch_type"            : int(pitch_type),
            "dew_factor"            : int(dew_factor),
            "boundary_size"         : int(boundary_size),
            "avg_chasing_success"   : round(avg_chase_succ, 4),
            "venue_avg_runs"        : round(v_avg_runs, 2),  # Rolling avg 1st innings score at venue

            # ── TOSS INTELLIGENCE (VENUE-AWARE) (5) ───────────────────────────────
            # Replaces weak generic toss features with venue context
            "toss_advantage_at_venue"           : round(toss_adv, 4),  # Isolated toss impact per venue
            "team1_chasing_strength"            : round(t1_chase_str, 4),  # How good T1 is at chasing
            "team2_chasing_strength"            : round(t2_chase_str, 4),  # How good T2 is at chasing
            "team1_chase_venue_interaction"     : round(t1_chase_str * avg_chase_succ, 4),  # T1 chase × venue bias
            "team2_chase_venue_interaction"     : round(t2_chase_str * avg_chase_succ, 4),  # T2 chase × venue bias

            # ── SCORING STRENGTH FEATURES (CRITICAL - WAS MISSING!) (4) ─────────────
            # Phase-specific scoring power - THE BIGGEST ACCURACY BOOST
            "team1_death_runs"      : round(t1_death_runs, 2),   # Team 1's death overs runs (15-20)
            "team2_death_runs"      : round(t2_death_runs, 2),   # Team 2's death overs runs (15-20)
            "team1_powerplay_runs"  : round(t1_pp_runs, 2),      # Team 1's powerplay runs (1-6)
            "team2_powerplay_runs"  : round(t2_pp_runs, 2),      # Team 2's powerplay runs (1-6)

            # ── MOMENTUM FEATURES (HOT vs COLD) (6) ──────────────────────────────
            # Short-term performance signals reflecting current team form
            "team1_last_3_match_avg_runs"       : round(t1_last_3_runs, 2),  # Team 1's avg runs (last 3 matches)
            "team2_last_3_match_avg_runs"       : round(t2_last_3_runs, 2),  # Team 2's avg runs (last 3 matches)
            "team1_last_3_match_wickets_taken"  : round(t1_last_3_wkts, 2),  # Team 1's avg wickets (last 3 matches)
            "team2_last_3_match_wickets_taken"  : round(t2_last_3_wkts, 2),  # Team 2's avg wickets (last 3 matches)
            "team1_form_trend_slope"            : round(t1_form_slope, 4),   # Team 1's form trend (+ = improving)
            "team2_form_trend_slope"            : round(t2_form_slope, 4),   # Team 2's form trend (+ = improving)

            # ── DIFFERENCE FEATURES (TREE MODELS LOVE COMPARISONS) (2) ─────────────
            "diff_run_rate"         : round(t1_run_rate - t2_run_rate, 4),   # Direct run rate comparison
            "diff_avg_runs"         : round(t1_avg_runs - t2_avg_runs, 2),   # Direct avg runs comparison

            # ── INTERACTION FEATURES (ACTUAL MATCH DYNAMICS) (2) ────────────────────
            # Correct interactions: phase scoring × opponent phase weakness
            "death_runs_vs_opponent_death_economy"   : round(t1_death_runs * t2_death_econ, 2),  # Death runs × opponent death weakness
            "powerplay_runs_vs_opponent_pp_wickets"  : round(t1_pp_runs * (1 / max(t2_pp_wkts, 0.5)), 2),  # PP runs × inverse opponent PP wickets

            # ── Target ────────────────────────────────────────────────────────
            "target": target,
        }
        rows.append(feature_row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_print(master_df):
    print("\n-- FEATURE SUMMARY (41 FEATURES: SCORING STRENGTH + DIFFERENCE + INTERACTION FIX) -----")
    print(f"  Total rows     : {len(master_df)}")
    print(f"  Total features : {len(FEATURE_COLS)}")
    print(f"  Target balance : team1 wins = {master_df['target'].mean():.1%}")

    # Check for nulls in all features
    nulls = master_df[FEATURE_COLS].isnull().sum()
    nulls = nulls[nulls > 0]
    print(f"\n  Null check:")
    if len(nulls) == 0:
        print("    [OK] No nulls in any feature column")
    else:
        print(nulls)

    # Sanity checks
    print(f"\n  TOURNAMENT CONTEXT FEATURES:")
    if "team1_last5_win_pct" in master_df.columns:
        print(f"    [OK] team1_last5_win_pct   : mean={master_df['team1_last5_win_pct'].mean():.3f} (expect ~0.5)")
    if "last3_h2h_team1_pct" in master_df.columns:
        print(f"    [OK] last3_h2h_team1_pct   : mean={master_df['last3_h2h_team1_pct'].mean():.3f} (expect ~0.5)")

    print(f"\n  PLAYING XI FEATURES (GAME CHANGER):")
    if "team1_playing11_bat_sr" in master_df.columns:
        print(f"    [OK] team1_playing11_bat_sr : mean={master_df['team1_playing11_bat_sr'].mean():.1f} (range: 100-150)")
    if "team1_playing11_bowl_econ" in master_df.columns:
        print(f"    [OK] team1_playing11_bowl_econ : mean={master_df['team1_playing11_bowl_econ'].mean():.2f} (range: 6-11)")
    if "team1_num_death_bowlers" in master_df.columns:
        print(f"    [OK] team1_num_death_bowlers : mean={master_df['team1_num_death_bowlers'].mean():.1f} (range: 0-4)")
    if "team1_num_pp_bowlers" in master_df.columns:
        print(f"    [OK] team1_num_pp_bowlers : mean={master_df['team1_num_pp_bowlers'].mean():.1f} (range: 0-4)")

    print(f"\n  VENUE & PITCH INTELLIGENCE (HIDDEN GOLD):")
    if "pitch_type" in master_df.columns:
        print(f"    [OK] pitch_type      : unique={master_df['pitch_type'].nunique()} (0=flat, 1=slow, 2=spin, 3=pace)")
    if "dew_factor" in master_df.columns:
        print(f"    [OK] dew_factor      : unique={master_df['dew_factor'].nunique()} (0=day, 1=night)")
    if "boundary_size" in master_df.columns:
        print(f"    [OK] boundary_size   : unique={master_df['boundary_size'].nunique()} (0=small, 1=medium, 2=large)")
    if "avg_chasing_success" in master_df.columns:
        print(f"    [OK] avg_chasing_success : mean={master_df['avg_chasing_success'].mean():.3f} (range: 0-1)")
    if "venue_avg_runs" in master_df.columns:
        print(f"    [OK] venue_avg_runs  : mean={master_df['venue_avg_runs'].mean():.1f} (range: 140-220)")
        print(f"         • Rolling avg first innings score at venue (critical context!)")

    print(f"\n  TOSS INTELLIGENCE (VENUE-AWARE, CONTEXT-DRIVEN) — CRITICAL UPGRADE:")
    if "toss_advantage_at_venue" in master_df.columns:
        print(f"    [OK] toss_advantage_at_venue : mean={master_df['toss_advantage_at_venue'].mean():.4f} (range: -0.25 to +0.25)")
        print(f"         • Isolates toss impact at each venue (how much toss win helps)")
    if "team1_chasing_strength" in master_df.columns:
        print(f"    [OK] team1_chasing_strength  : mean={master_df['team1_chasing_strength'].mean():.3f} (range: 0-1)")
        print(f"         • Team 1's historical chase win rate (independent of venue)")
    if "team2_chasing_strength" in master_df.columns:
        print(f"    [OK] team2_chasing_strength  : mean={master_df['team2_chasing_strength'].mean():.3f} (range: 0-1)")
        print(f"         • Team 2's historical chase win rate (independent of venue)")

    print(f"\n  SCORING STRENGTH (CRITICAL - WAS MISSING!):")
    if "team1_death_runs" in master_df.columns:
        print(f"    [OK] team1_death_runs : mean={master_df['team1_death_runs'].mean():.1f} (range: 30-60)")
        print(f"         • Team 1's death overs runs (overs 15-20)")
    if "team1_powerplay_runs" in master_df.columns:
        print(f"    [OK] team1_powerplay_runs : mean={master_df['team1_powerplay_runs'].mean():.1f} (range: 35-60)")
        print(f"         • Team 1's powerplay runs (overs 1-6)")

    print(f"\n  MOMENTUM FEATURES — HOT vs COLD DETECTION:")
    if "team1_last_3_match_avg_runs" in master_df.columns:
        print(f"    [OK] team1_last_3_match_avg_runs : mean={master_df['team1_last_3_match_avg_runs'].mean():.1f} (range: 100-200)")
        print(f"         • Average runs scored by team in last 3 matches (batting momentum)")
    if "team1_last_3_match_wickets_taken" in master_df.columns:
        print(f"    [OK] team1_last_3_match_wickets_taken : mean={master_df['team1_last_3_match_wickets_taken'].mean():.2f} (range: 3-10)")
        print(f"         • Average wickets taken by team in last 3 matches (bowling momentum)")
    if "team1_form_trend_slope" in master_df.columns:
        print(f"    [OK] team1_form_trend_slope : mean={master_df['team1_form_trend_slope'].mean():.4f} (range: -1 to +1)")
        print(f"         • Form trend: positive = improving, negative = declining")

    print(f"\n  DIFFERENCE & INTERACTION (TREE MODEL BOOSTERS):")
    if "diff_run_rate" in master_df.columns:
        print(f"    [OK] diff_run_rate : mean={master_df['diff_run_rate'].mean():.4f} (range: -3 to +3)")
    if "diff_avg_runs" in master_df.columns:
        print(f"    [OK] diff_avg_runs : mean={master_df['diff_avg_runs'].mean():.1f} (range: -40 to +40)")
    if "death_runs_vs_opponent_death_economy" in master_df.columns:
        print(f"    [OK] death_runs_vs_opponent_death_economy : mean={master_df['death_runs_vs_opponent_death_economy'].mean():.1f}")
    if "powerplay_runs_vs_opponent_pp_wickets" in master_df.columns:
        print(f"    [OK] powerplay_runs_vs_opponent_pp_wickets : mean={master_df['powerplay_runs_vs_opponent_pp_wickets'].mean():.1f}")

    print("\n  Sample feature values (first match):")
    sample_cols = [c for c in FEATURE_COLS if c in master_df.columns]
    print(master_df[sample_cols].head(1).to_string())

    print(f"\n  COMPLETE FEATURE LIST ({len(FEATURE_COLS)} features):")
    print("  BASE (4):")
    for i, feat in enumerate(FEATURE_COLS[:4], 1):
        print(f"    {i}. {feat}")
    print("  TOURNAMENT CONTEXT (5):")
    for i, feat in enumerate(FEATURE_COLS[4:9], 5):
        print(f"    {i}. {feat}")
    print("  PLAYING XI (8):")
    for i, feat in enumerate(FEATURE_COLS[9:17], 9):
        print(f"    {i}. {feat}")
    print("  VENUE & PITCH INTELLIGENCE (5):")
    for i, feat in enumerate(FEATURE_COLS[17:22], 17):
        print(f"    {i}. {feat}")
    print("  TOSS INTELLIGENCE (5) — VENUE-AWARE + INTERACTION:")
    for i, feat in enumerate(FEATURE_COLS[22:27], 22):
        print(f"    {i}. {feat}")
    print("  SCORING STRENGTH (4) — CRITICAL NEW:")
    for i, feat in enumerate(FEATURE_COLS[27:31], 27):
        print(f"    {i}. {feat}")
    print("  MOMENTUM (6) — HOT vs COLD:")
    for i, feat in enumerate(FEATURE_COLS[31:37], 31):
        print(f"    {i}. {feat}")
    print("  DIFFERENCE (2) — TREE MODEL COMPARISONS:")
    for i, feat in enumerate(FEATURE_COLS[37:39], 37):
        print(f"    {i}. {feat}")
    print("  INTERACTION (2) — ACTUAL MATCH DYNAMICS:")
    for i, feat in enumerate(FEATURE_COLS[39:41], 39):
        print(f"    {i}. {feat}")



# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  PITCHMIND — FEATURE ENGINEERING v12 (SCORING STRENGTH FIX)")
    print("  CRITICAL FIXES:")
    print("  • ADDED: death_runs, powerplay_runs (WAS MISSING FROM TRAINING!)")
    print("  • ADDED: venue_avg_runs (rolling 1st innings avg at venue)")
    print("  • ADDED: diff_run_rate, diff_avg_runs (comparison features)")
    print("  • FIXED: Interaction features (death_runs×death_econ, pp_runs×pp_wkts)")
    print("  • REMOVED: Duplicate compute_current_season_nrr function")
    print(f"  Total features: {len(FEATURE_COLS)} (4 base + 5 tournament + 8 XI + 5 venue + 5 toss + 4 scoring + 6 momentum + 2 diff + 2 interaction)")
    print("=" * 70)

    if not os.path.exists(os.path.join(DATA_DIR, "matches.csv")):
        print("ERROR: data/matches.csv not found.")
        exit(1)

    matches, deliveries = load_data()
    master_df = build_master_features(matches, deliveries)
    validate_and_print(master_df)

    out_path = os.path.join(DATA_DIR, "master_features.csv")
    master_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("  [OK] Feature engineering v12 COMPLETE (Scoring Strength Fix)")
    print(f"  Output: {out_path}")
    print(f"  Shape : {master_df.shape}")
    print()
    print("  FEATURE BREAKDOWN (27 total):")
    print("  [BASE] (4): death_econ, recency, win_rate, squad_bowl_econ")
    print("  [TOURNAMENT] (5): last5_winpct, h2h, season_pts, season_nrr, team2_last5")
    print("  [PLAYING XI] (8): bat_sr, bowl_econ, death_bowlers, pp_bowlers (x2 teams)")
    print("  [VENUE] (4): pitch_type, dew_factor, boundary_size, avg_chasing_success")
    print("  [TOSS INTELLIGENCE] (5): toss_advantage, chasing_strength, chase_venue_interaction (x2)")
    print("  [MOMENTUM - NEW] (6):")
    print("    • last_3_match_avg_runs - Batting momentum (runs in recent matches)")
    print("    • last_3_match_wickets_taken - Bowling momentum (wickets in recent matches)")
    print("    • form_trend_slope - Performance trend (improving/declining)")
    print()
    print("  KEY IMPROVEMENTS (v11):")
    print("  [ADDED] Momentum features (HOT vs COLD detection)")
    print("  [ADDED] Recent batting performance signal")
    print("  [ADDED] Recent bowling performance signal")
    print("  [ADDED] Form trend slope (identifying improving/declining teams)")
    print("  [RESULT] Model now captures short-term team form & current momentum")
    print()
    print("=" * 70)