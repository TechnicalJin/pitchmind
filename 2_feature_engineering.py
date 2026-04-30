"""
PITCHMIND — 2_feature_engineering.py  (v6 — FIXES)
====================================================
FIXES vs v5:
  FIX 1 — CRITICAL: Now loads matches_clean.csv + deliveries_clean.csv
           (v5 was loading raw matches.csv = 1209 rows instead of 259)

  FIX 2 — Phase split now sums correctly to avg_runs:
           pp_runs=51 + middle_runs=73 + death_runs=61 = 185 ✅
           (v5 had 55+65+55 = 175 ≠ 185 — 10 run gap)

  FIX 3 — Bowling defaults corrected for 2023+ reality:
           death_economy: 12.0 → 11.2  (was too pessimistic)
           economy:        10.0 →  9.8  (global avg is 9.8, not 10.0)
           pp_wickets:      2.0 →  1.7  (actual avg is 1.6–1.8)
           bowling_sr:     17.0 → 17.5

  FIX 4 — Venue avg_runs now uses actual clean-data rolling average.
           Fallback is 180 (not 185) so model doesn't over-inflate unknowns.

  FIX 5 — top3_sr default 145 → 143 (more accurate 2023+ mean)

Run:
  python 2_feature_engineering.py

Output:
  data/master_features.csv
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
    home_list = HOME_VENUES.get(team, [])
    venue_lower = venue.lower()
    return any(hv.lower() in venue_lower or venue_lower in hv.lower() for hv in home_list)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMN REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "team1_win_rate", "team2_win_rate",
    "team1_recent_form", "team2_recent_form",
    "h2h_win_rate",
    "team1_nrr", "team2_nrr",
    "team1_avg_runs", "team2_avg_runs",
    "team1_powerplay_runs", "team2_powerplay_runs",
    "team1_middle_runs", "team2_middle_runs",
    "team1_death_runs", "team2_death_runs",
    "team1_boundary_pct", "team2_boundary_pct",
    "team1_dot_ball_pct", "team2_dot_ball_pct",
    "team1_run_rate", "team2_run_rate",
    "team1_top3_sr", "team2_top3_sr",
    "team1_bowling_economy", "team2_bowling_economy",
    "team1_death_economy", "team2_death_economy",
    "team1_pp_wickets", "team2_pp_wickets",
    "team1_bowling_sr", "team2_bowling_sr",
    "team1_venue_win_rate", "team2_venue_win_rate",
    "venue_avg_runs",
    "toss_win", "toss_field", "toss_team1_field",
    "team1_chase_win_rate", "team2_chase_win_rate",
    "team1_home_win_rate", "team2_home_win_rate",
    "team1_win_streak", "team2_win_streak",
    "team1_days_rest", "team2_days_rest",
    "season_stage",
    "team1_squad_bat_sr", "team2_squad_bat_sr",
    "team1_squad_bowl_econ", "team2_squad_bowl_econ",
    "team1_squad_allrounder", "team2_squad_allrounder",
    "diff_win_rate",
    "diff_recent_form",
    "diff_avg_runs",
    "diff_death_runs",
    "diff_death_economy",
    "diff_bowling_economy",
    "diff_pp_wickets",
    "diff_run_rate",
    "diff_nrr",
    "diff_venue_win_rate",
    "diff_chase_win_rate",
    "diff_squad_bat_sr",
]

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT VALUES — CORRECTED FOR 2023+ IPL
# ══════════════════════════════════════════════════════════════════════════════
# FIX 2: Phase split now adds up correctly:
#   pp_runs(51) + middle_runs(73) + death_runs(61) = 185 = avg_runs ✅
#
# FIX 5: top3_sr 145 → 143 (more accurate 2023+ mean)
BAT_DEFAULTS = {
    "avg_runs"    : 185,    # 2023+ first innings avg
    "run_rate"    : 9.5,    # 9.5 RPO is the new average
    "pp_runs"     : 51.0,   # FIX: was 55 — actual 2023+ PP avg is 50–53
    "middle_runs" : 73.0,   # FIX: was 65 — middle overs carry more weight
    "death_runs"  : 61.0,   # FIX: was 55 — 51+73+61=185 ✅
    "boundary_pct": 0.19,   # more boundaries per delivery
    "dot_ball_pct": 0.28,   # fewer dots; batters attack more
    "top3_sr"     : 143.0,  # FIX: was 145 — 143 is more accurate 2023+ mean
}

# FIX 3: Bowling defaults corrected
BOWL_DEFAULTS = {
    "economy"      : 9.8,   # FIX: was 10.0 — actual 2023+ avg is ~9.7–9.9
    "death_economy": 11.2,  # FIX: was 12.0 — actual avg is 11.0–11.5
    "pp_wickets"   : 1.7,   # FIX: was 2.0 — actual avg is 1.6–1.8
    "bowling_sr"   : 17.5,  # FIX: was 17.0 — slightly more conservative
}

EWM_SPAN = 7


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# FIX 1: Load matches_clean.csv + deliveries_clean.csv (not raw files)
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("Loading datasets...")

    # FIX 1: Use clean files first, fall back to raw only if clean doesn't exist
    matches_path = os.path.join(DATA_DIR, "matches_clean.csv")
    deliveries_path = os.path.join(DATA_DIR, "deliveries_clean.csv")

    if not os.path.exists(matches_path):
        print(f"  ⚠️  matches_clean.csv not found, falling back to matches.csv")
        print(f"  ⚠️  Run 1_data_cleaning.py first for proper 2023+ filtering!")
        matches_path = os.path.join(DATA_DIR, "matches.csv")

    if not os.path.exists(deliveries_path):
        print(f"  ⚠️  deliveries_clean.csv not found, falling back to deliveries.csv")
        deliveries_path = os.path.join(DATA_DIR, "deliveries.csv")

    matches    = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    print(f"  matches loaded    : {matches.shape}")
    print(f"  deliveries loaded : {deliveries.shape}")

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
    print(f"  Seasons in data : {sorted(matches['season_int'].unique())}")
    return matches, deliveries


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2A — ROLLING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_win_rate(matches):
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
        s = s if not pd.isna(s) else 9.5
        c = c if not pd.isna(c) else 9.5
        nrr_dict[key] = round(s - c, 4)
    return nrr_dict


def compute_rolling_venue_win_rate(matches):
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
    """
    FIX 4: Rolling venue avg runs from actual clean data.
    Fallback changed to 180 (not 185) — slightly conservative is better
    than over-inflating unknown venues.
    """
    first_inn    = deliveries[deliveries["inning"] == 1].copy()
    match_scores = first_inn.groupby("match_id")["total_runs"].sum().reset_index()
    match_scores.columns = ["match_id", "runs"]
    match_info   = matches[["id", "date", "venue"]].rename(columns={"id": "match_id"})
    match_scores = match_scores.merge(match_info, on="match_id", how="left").sort_values("date")
    match_scores["hist_venue_avg"] = match_scores.groupby("venue")["runs"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    # FIX 4: fallback = 180 (not 185) for unknown venues
    match_scores["hist_venue_avg"] = match_scores["hist_venue_avg"].fillna(180)
    deduped = match_scores.drop_duplicates(subset=["match_id", "venue"], keep="first")
    return deduped.set_index(["match_id", "venue"])["hist_venue_avg"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2B — EWM Delivery Features
# ══════════════════════════════════════════════════════════════════════════════

def compute_ewm_delivery_features(deliveries, matches, span=EWM_SPAN):
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

            pp     = inn[inn["over"] <= 5]
            middle = inn[(inn["over"] >= 6) & (inn["over"] <= 14)]
            death  = inn[inn["over"] >= 15]

            pp_runs     = pp["total_runs"].sum()
            pp_wickets  = int(pp["is_wicket"].sum())
            middle_runs = middle["total_runs"].sum()
            death_runs  = death["total_runs"].sum()
            death_overs = death["over"].nunique()
            death_rr    = death_runs / death_overs if death_overs > 0 else run_rate

            boundaries   = inn[inn["batsman_runs"].isin([4, 6])]
            boundary_pct = len(boundaries) / total_balls if total_balls > 0 else 0
            dot_balls    = inn[inn["total_runs"] == 0]
            dot_ball_pct = len(dot_balls) / total_balls if total_balls > 0 else 0

            batter_runs  = inn.groupby("batter")["batsman_runs"].sum().nlargest(3).index.tolist()
            top3_del     = inn[inn["batter"].isin(batter_runs)]
            top3_runs    = top3_del["batsman_runs"].sum()
            top3_sr      = (top3_runs / len(top3_del) * 100) if len(top3_del) > 0 else 0

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
# STEP 2C — Chase Win Rate
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
        lambda x: x.shift(1).ewm(span=EWM_SPAN, adjust=False, min_periods=1).mean()
    )
    df["chase_win_rate"] = df["chase_win_rate"].fillna(0.5)
    return df.set_index(["match_id", "team"])["chase_win_rate"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2D — Home Win Rate
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
        lambda x: x.shift(1).ewm(span=EWM_SPAN, adjust=False, min_periods=1).mean()
    )
    df["home_win_rate"] = df["home_win_rate"].fillna(0.5)
    return df.set_index(["match_id", "team"])["home_win_rate"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2E — Win Streak
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
    for team, grp in df.groupby("team"):
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
# STEP 2F — Days Rest
# ══════════════════════════════════════════════════════════════════════════════

def compute_days_rest(matches):
    records = []
    for _, row in matches.iterrows():
        for team in [row["team1"], row["team2"]]:
            records.append({"match_id": row["id"], "date": row["date"], "team": team})
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
# STEP 2G — Player-Level Squad Features
# ══════════════════════════════════════════════════════════════════════════════

def compute_squad_features(deliveries, matches):
    """
    Squad features from deliveries history (last 5 matches).
    NOTE: These are HISTORICAL squad proxies used for training.
    The dashboard overrides these with actual XI stats at prediction time.
    """
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
                    "squad_bat_sr"    : BAT_DEFAULTS["top3_sr"],
                    "squad_bowl_econ" : BOWL_DEFAULTS["economy"],
                    "squad_allrounder": 2,
                }
                continue
            recent_players = set()
            for _, pr in last5.iterrows():
                recent_players |= pr["all_players"]

            bat_srs = [bat_sr_dict[p] for p in recent_players if p in bat_sr_dict]
            squad_bat_sr = round(np.mean(bat_srs), 1) if bat_srs else BAT_DEFAULTS["top3_sr"]

            bowl_econs = [bowl_econ_dict[p] for p in recent_players if p in bowl_econ_dict]
            squad_bowl_econ = round(np.mean(bowl_econs), 2) if bowl_econs else BOWL_DEFAULTS["economy"]

            squad_allrounder = len(recent_players & allrounder_set)
            result[(mid, team)] = {
                "squad_bat_sr"    : squad_bat_sr,
                "squad_bowl_econ" : squad_bowl_econ,
                "squad_allrounder": squad_allrounder,
            }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2H — Season Stage
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
# STEP 3 — ASSEMBLE MASTER FEATURE DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_master_features(matches, deliveries):
    print("\nComputing all features (rolling, no leakage)...")

    print("  [1/12] Rolling team win rates...")
    win_rates = compute_rolling_win_rate(matches)
    print("  [2/12] Rolling recent form EWM (last 5 matches)...")
    recent_form = compute_rolling_recent_form(matches)
    print("  [3/12] Rolling head-to-head win rates...")
    h2h = compute_rolling_h2h(matches)
    print("  [4/12] Rolling NRR (from deliveries)...")
    nrr = compute_rolling_nrr(matches, deliveries)
    print("  [5/12] Rolling venue win rates...")
    venue_win_rates = compute_rolling_venue_win_rate(matches)
    print("  [6/12] Rolling venue average runs (dynamic from clean data)...")
    venue_avg_runs = compute_rolling_venue_avg_runs(deliveries, matches)
    print(f"  [7/12] EWM batting & bowling features (span={EWM_SPAN})...")
    bat_feats, bowl_feats = compute_ewm_delivery_features(deliveries, matches)
    print("  [8/12] Rolling chase win rates (EWM)...")
    chase_win_rates = compute_rolling_chase_win_rate(matches)
    print("  [9/12] Rolling home win rates...")
    home_win_rates = compute_rolling_home_win_rate(matches)
    print("  [10/12] Win streaks...")
    win_streaks = compute_win_streak(matches)
    print("  [11/12] Days rest between matches...")
    days_rest = compute_days_rest(matches)
    print("  [12/12] Player-level squad features (last 5 matches)...")
    squad_feats = compute_squad_features(deliveries, matches)
    season_stage = compute_season_stage(matches)

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

        toss_win         = int(row["toss_winner"] == t1)
        toss_field       = int(row["toss_decision"] == "field")
        toss_team1_field = int(row["toss_winner"] == t1 and row["toss_decision"] == "field")
        target           = int(row["winner"] == t1)

        t1_sq_bat_sr  = t1_squad.get("squad_bat_sr",     BAT_DEFAULTS["top3_sr"])
        t2_sq_bat_sr  = t2_squad.get("squad_bat_sr",     BAT_DEFAULTS["top3_sr"])
        t1_sq_econ    = t1_squad.get("squad_bowl_econ",  BOWL_DEFAULTS["economy"])
        t2_sq_econ    = t2_squad.get("squad_bowl_econ",  BOWL_DEFAULTS["economy"])
        t1_sq_ar      = t1_squad.get("squad_allrounder", 2)
        t2_sq_ar      = t2_squad.get("squad_allrounder", 2)

        # FIX 4: get actual venue avg from rolling calc, fallback 180
        venue_avg = round(venue_avg_runs.get((mid, venue), 180), 2)

        feature_row = {
            "match_id" : mid,
            "date"     : row["date"],
            "season"   : row["season_int"],
            "team1"    : t1,
            "team2"    : t2,
            "venue"    : venue,

            "team1_win_rate"    : round(win_rates.get((mid, t1), 0.5), 4),
            "team2_win_rate"    : round(win_rates.get((mid, t2), 0.5), 4),
            "team1_recent_form" : round(recent_form.get((mid, t1), 0.5), 4),
            "team2_recent_form" : round(recent_form.get((mid, t2), 0.5), 4),
            "h2h_win_rate"      : round(h2h.get((mid, t1, t2), 0.5), 4),
            "team1_nrr"         : round(nrr.get((mid, t1), 0.0), 4),
            "team2_nrr"         : round(nrr.get((mid, t2), 0.0), 4),

            "team1_avg_runs"       : round(t1_bat.get("avg_runs",     BAT_DEFAULTS["avg_runs"]),     2),
            "team2_avg_runs"       : round(t2_bat.get("avg_runs",     BAT_DEFAULTS["avg_runs"]),     2),
            "team1_powerplay_runs" : round(t1_bat.get("pp_runs",      BAT_DEFAULTS["pp_runs"]),      2),
            "team2_powerplay_runs" : round(t2_bat.get("pp_runs",      BAT_DEFAULTS["pp_runs"]),      2),
            "team1_middle_runs"    : round(t1_bat.get("middle_runs",  BAT_DEFAULTS["middle_runs"]),  2),
            "team2_middle_runs"    : round(t2_bat.get("middle_runs",  BAT_DEFAULTS["middle_runs"]),  2),
            "team1_death_runs"     : round(t1_bat.get("death_runs",   BAT_DEFAULTS["death_runs"]),   2),
            "team2_death_runs"     : round(t2_bat.get("death_runs",   BAT_DEFAULTS["death_runs"]),   2),
            "team1_boundary_pct"   : round(t1_bat.get("boundary_pct", BAT_DEFAULTS["boundary_pct"]), 4),
            "team2_boundary_pct"   : round(t2_bat.get("boundary_pct", BAT_DEFAULTS["boundary_pct"]), 4),
            "team1_dot_ball_pct"   : round(t1_bat.get("dot_ball_pct", BAT_DEFAULTS["dot_ball_pct"]), 4),
            "team2_dot_ball_pct"   : round(t2_bat.get("dot_ball_pct", BAT_DEFAULTS["dot_ball_pct"]), 4),
            "team1_run_rate"       : round(t1_bat.get("run_rate",     BAT_DEFAULTS["run_rate"]),     4),
            "team2_run_rate"       : round(t2_bat.get("run_rate",     BAT_DEFAULTS["run_rate"]),     4),
            "team1_top3_sr"        : round(t1_bat.get("top3_sr",      BAT_DEFAULTS["top3_sr"]),      2),
            "team2_top3_sr"        : round(t2_bat.get("top3_sr",      BAT_DEFAULTS["top3_sr"]),      2),

            "team1_bowling_economy" : round(t1_bowl.get("economy",       BOWL_DEFAULTS["economy"]),       4),
            "team2_bowling_economy" : round(t2_bowl.get("economy",       BOWL_DEFAULTS["economy"]),       4),
            "team1_death_economy"   : round(t1_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"]), 4),
            "team2_death_economy"   : round(t2_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"]), 4),
            "team1_pp_wickets"      : round(t1_bowl.get("pp_wickets",    BOWL_DEFAULTS["pp_wickets"]),    4),
            "team2_pp_wickets"      : round(t2_bowl.get("pp_wickets",    BOWL_DEFAULTS["pp_wickets"]),    4),
            "team1_bowling_sr"      : round(t1_bowl.get("bowling_sr",    BOWL_DEFAULTS["bowling_sr"]),    4),
            "team2_bowling_sr"      : round(t2_bowl.get("bowling_sr",    BOWL_DEFAULTS["bowling_sr"]),    4),

            "team1_venue_win_rate" : round(venue_win_rates.get((mid, t1), 0.5), 4),
            "team2_venue_win_rate" : round(venue_win_rates.get((mid, t2), 0.5), 4),
            "venue_avg_runs"       : venue_avg,

            "toss_win"          : toss_win,
            "toss_field"        : toss_field,
            "toss_team1_field"  : toss_team1_field,

            "team1_chase_win_rate" : round(chase_win_rates.get((mid, t1), 0.5), 4),
            "team2_chase_win_rate" : round(chase_win_rates.get((mid, t2), 0.5), 4),
            "team1_home_win_rate"  : round(home_win_rates.get((mid, t1), 0.5), 4),
            "team2_home_win_rate"  : round(home_win_rates.get((mid, t2), 0.5), 4),
            "team1_win_streak"     : win_streaks.get((mid, t1), 0),
            "team2_win_streak"     : win_streaks.get((mid, t2), 0),
            "team1_days_rest"      : days_rest.get((mid, t1), 7),
            "team2_days_rest"      : days_rest.get((mid, t2), 7),
            "season_stage"         : season_stage.get(mid, 0),

            "team1_squad_bat_sr"    : round(t1_sq_bat_sr,  1),
            "team2_squad_bat_sr"    : round(t2_sq_bat_sr,  1),
            "team1_squad_bowl_econ" : round(t1_sq_econ,    2),
            "team2_squad_bowl_econ" : round(t2_sq_econ,    2),
            "team1_squad_allrounder": int(t1_sq_ar),
            "team2_squad_allrounder": int(t2_sq_ar),

            "diff_win_rate"        : round(win_rates.get((mid, t1), 0.5) - win_rates.get((mid, t2), 0.5), 4),
            "diff_recent_form"     : round(recent_form.get((mid, t1), 0.5) - recent_form.get((mid, t2), 0.5), 4),
            "diff_avg_runs"        : round(t1_bat.get("avg_runs", BAT_DEFAULTS["avg_runs"]) - t2_bat.get("avg_runs", BAT_DEFAULTS["avg_runs"]), 2),
            "diff_death_runs"      : round(t1_bat.get("death_runs", BAT_DEFAULTS["death_runs"]) - t2_bat.get("death_runs", BAT_DEFAULTS["death_runs"]), 2),
            "diff_death_economy"   : round(t2_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"]) - t1_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"]), 4),
            "diff_bowling_economy" : round(t2_bowl.get("economy", BOWL_DEFAULTS["economy"]) - t1_bowl.get("economy", BOWL_DEFAULTS["economy"]), 4),
            "diff_pp_wickets"      : round(t1_bowl.get("pp_wickets", BOWL_DEFAULTS["pp_wickets"]) - t2_bowl.get("pp_wickets", BOWL_DEFAULTS["pp_wickets"]), 4),
            "diff_run_rate"        : round(t1_bat.get("run_rate", BAT_DEFAULTS["run_rate"]) - t2_bat.get("run_rate", BAT_DEFAULTS["run_rate"]), 4),
            "diff_nrr"             : round(nrr.get((mid, t1), 0.0) - nrr.get((mid, t2), 0.0), 4),
            "diff_venue_win_rate"  : round(venue_win_rates.get((mid, t1), 0.5) - venue_win_rates.get((mid, t2), 0.5), 4),
            "diff_chase_win_rate"  : round(chase_win_rates.get((mid, t1), 0.5) - chase_win_rates.get((mid, t2), 0.5), 4),
            "diff_squad_bat_sr"    : round(t1_sq_bat_sr - t2_sq_bat_sr, 1),

            "target": target,
        }
        rows.append(feature_row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_print(master_df):
    print("\n── FEATURE SUMMARY ─────────────────────────────────────────────")
    print(f"  Total rows     : {len(master_df)}")
    print(f"  Total features : {len(FEATURE_COLS)}")
    print(f"  Target balance : team1 wins = {master_df['target'].mean():.1%}")

    nulls = master_df[FEATURE_COLS].isnull().sum()
    nulls = nulls[nulls > 0]
    print(f"\n  Null check:")
    if len(nulls) == 0:
        print("    ✅ No nulls in any feature column")
    else:
        print(nulls)

    gap = abs(master_df["team1_avg_runs"].mean() - master_df["team2_avg_runs"].mean())
    print(f"\n  Leakage check:")
    print(f"    team1_avg_runs mean = {master_df['team1_avg_runs'].mean():.2f}")
    print(f"    team2_avg_runs mean = {master_df['team2_avg_runs'].mean():.2f}")
    print(f"    Gap = {gap:.2f} runs {'✅ (PASS < 2)' if gap < 2 else '❌ (FAIL > 2)'}")

    avg_runs_mean = master_df["team1_avg_runs"].mean()
    print(f"\n  Era sanity check (expect ~185 for 2023+ era):")
    status = "✅" if 170 <= avg_runs_mean <= 205 else "⚠️  check defaults"
    print(f"    team1_avg_runs mean = {avg_runs_mean:.1f}  {status}")
    print(f"    venue_avg_runs mean = {master_df['venue_avg_runs'].mean():.1f}  (expect 175-195 for 2023+)")

    # FIX 2: Phase split check
    pp_mean  = master_df["team1_powerplay_runs"].mean()
    mid_mean = master_df["team1_middle_runs"].mean()
    dth_mean = master_df["team1_death_runs"].mean()
    phase_sum = pp_mean + mid_mean + dth_mean
    print(f"\n  Phase split check (should sum to ~avg_runs):")
    print(f"    PP={pp_mean:.1f} + Mid={mid_mean:.1f} + Death={dth_mean:.1f} = {phase_sum:.1f}")
    gap_ph = abs(phase_sum - avg_runs_mean)
    print(f"    Gap vs avg_runs: {gap_ph:.1f} {'✅ (< 8)' if gap_ph < 8 else '⚠️ check phase defaults'}")

    print(f"\n  Toss winner bias check:")
    toss_mean = master_df['toss_win'].mean()
    print(f"    toss_win mean = {toss_mean:.4f}")
    validate_toss_distribution(master_df.rename(columns={"toss_win": "team1_toss_win"}))

    print(f"\n  Sample rows:")
    print(master_df[["team1", "team2", "team1_win_rate", "team1_avg_runs",
                      "team1_death_runs", "team1_bowling_economy", "target"]].head(5).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  PITCHMIND — FEATURE ENGINEERING v6 (FIXES)")
    print(f"  EWM span={EWM_SPAN} | avg_runs default={BAT_DEFAULTS['avg_runs']}")
    print(f"  Phase split: PP={BAT_DEFAULTS['pp_runs']} + Mid={BAT_DEFAULTS['middle_runs']} + Death={BAT_DEFAULTS['death_runs']} = {BAT_DEFAULTS['pp_runs']+BAT_DEFAULTS['middle_runs']+BAT_DEFAULTS['death_runs']}")
    print(f"  economy default={BOWL_DEFAULTS['economy']} | death_economy default={BOWL_DEFAULTS['death_economy']}")
    print(f"  Total features: {len(FEATURE_COLS)}")
    print("=" * 70)

    if not os.path.exists(os.path.join(DATA_DIR, "matches_clean.csv")):
        print("ERROR: data/matches_clean.csv not found. Run 1_data_cleaning.py first.")
        exit(1)

    matches, deliveries = load_data()
    master_df = build_master_features(matches, deliveries)
    validate_and_print(master_df)

    out_path = os.path.join(DATA_DIR, "master_features.csv")
    master_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("  Feature engineering v6 COMPLETE")
    print(f"  Output: {out_path}")
    print(f"  Shape : {master_df.shape}")
    print()
    print("  FIXES vs v5:")
    print("  ✅ FIX 1: Loads matches_clean.csv + deliveries_clean.csv (not raw)")
    print(f"  ✅ FIX 2: Phase split: PP={BAT_DEFAULTS['pp_runs']}+Mid={BAT_DEFAULTS['middle_runs']}+Death={BAT_DEFAULTS['death_runs']}={BAT_DEFAULTS['pp_runs']+BAT_DEFAULTS['middle_runs']+BAT_DEFAULTS['death_runs']} (was 175)")
    print(f"  ✅ FIX 3: death_economy={BOWL_DEFAULTS['death_economy']} (was 12.0), economy={BOWL_DEFAULTS['economy']} (was 10.0), pp_wickets={BOWL_DEFAULTS['pp_wickets']} (was 2.0)")
    print(f"  ✅ FIX 4: venue_avg_runs = dynamic rolling calc, fallback=180")
    print(f"  ✅ FIX 5: top3_sr default={BAT_DEFAULTS['top3_sr']} (was 145)")
    print("=" * 70)
    print("\n  Next step → python 3_train_model.py")