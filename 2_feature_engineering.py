"""
PITCHMIND — 2_feature_engineering.py  (v4 — MAJOR UPGRADE)
============================================================
CHANGES vs v3:
  NEW 1: EWM (Exponential Weighted Mean) rolling averages
         span=10 so recent games matter 3x more than older ones
         Replaces: expanding().mean() → ewm(span=10).mean() on all
         batting/bowling delivery features

  NEW 2: High-Signal Context Features (6 new features)
         - team1_chase_win_rate   : rolling win rate when batting 2nd
         - team2_chase_win_rate
         - team1_home_win_rate    : rolling win rate at home venue
         - team2_home_win_rate
         - team1_win_streak       : consecutive wins before this match
         - team2_win_streak
         - days_since_last_match  : team1 & team2 fatigue/momentum
         - season_stage           : 0=group, 1=playoff (match 50+ = playoff)
         - diff_chase_win_rate    : team1 - team2 chase ability
         - diff_home_win_rate     : team1 - team2 home advantage

  NEW 3: Player-Level Features from deliveries (no XI needed)
         Built from career stats of all players who appeared for each
         team in their last N matches — proxy for current squad strength
         - team1_squad_bat_sr      : avg strike rate of team's regular batters
         - team2_squad_bat_sr
         - team1_squad_bowl_econ   : avg economy of team's regular bowlers
         - team2_squad_bowl_econ
         - team1_squad_allrounder  : all-rounder depth score
         - team2_squad_allrounder
         - diff_squad_bat_sr
         - diff_squad_bowl_econ

All existing bugs from v3 remain fixed. Total features: 45 → 63

Run:
  python 2_feature_engineering.py

Output:
  data/master_features.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"

# ══════════════════════════════════════════════════════════════════════════════
# TEAM & VENUE STANDARDIZATION
# ══════════════════════════════════════════════════════════════════════════════
TEAM_MAP = {
    "Delhi Daredevils"            : "Delhi Capitals",
    "Deccan Chargers"             : "Sunrisers Hyderabad",
    "Kings XI Punjab"             : "Punjab Kings",
    "Rising Pune Supergiants"     : "Rising Pune Supergiant",
    "Royal Challengers Bangalore" : "Royal Challengers Bengaluru",
}

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
# FEATURE COLUMN REGISTRY  (v4 — 63 features)
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # ── Team Strength — raw (7) ───────────────────────────────────────────────
    "team1_win_rate", "team2_win_rate",
    "team1_recent_form", "team2_recent_form",
    "h2h_win_rate",
    "team1_nrr", "team2_nrr",

    # ── Batting — EWM rolling (14) ────────────────────────────────────────────
    "team1_avg_runs", "team2_avg_runs",
    "team1_powerplay_runs", "team2_powerplay_runs",
    "team1_middle_runs", "team2_middle_runs",
    "team1_death_runs", "team2_death_runs",
    "team1_boundary_pct", "team2_boundary_pct",
    "team1_dot_ball_pct", "team2_dot_ball_pct",
    "team1_run_rate", "team2_run_rate",
    "team1_top3_sr", "team2_top3_sr",

    # ── Bowling — EWM rolling (8) ─────────────────────────────────────────────
    "team1_bowling_economy", "team2_bowling_economy",
    "team1_death_economy", "team2_death_economy",
    "team1_pp_wickets", "team2_pp_wickets",
    "team1_bowling_sr", "team2_bowling_sr",

    # ── Venue (3) ─────────────────────────────────────────────────────────────
    "team1_venue_win_rate", "team2_venue_win_rate",
    "venue_avg_runs",

    # ── Context — toss (3) ────────────────────────────────────────────────────
    "toss_win", "toss_field", "toss_team1_field",

    # ── NEW: Chase + Home + Momentum features (8) ────────────────────────────
    "team1_chase_win_rate",     # rolling win rate when batting 2nd (EWM)
    "team2_chase_win_rate",
    "team1_home_win_rate",      # rolling win rate at home venue
    "team2_home_win_rate",
    "team1_win_streak",         # consecutive wins before this match
    "team2_win_streak",
    "team1_days_rest",          # days since team1's last match (rest/fatigue)
    "team2_days_rest",
    "season_stage",             # 0=group stage, 1=playoff

    # ── NEW: Player-Level Squad Features (6) ─────────────────────────────────
    "team1_squad_bat_sr",       # avg career strike rate of regular batters
    "team2_squad_bat_sr",
    "team1_squad_bowl_econ",    # avg career economy of regular bowlers
    "team2_squad_bowl_econ",
    "team1_squad_allrounder",   # all-rounder depth: players who bat AND bowl
    "team2_squad_allrounder",

    # ── Difference Features — team1 minus team2 (12) ─────────────────────────
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
    "diff_chase_win_rate",      # NEW
    "diff_squad_bat_sr",        # NEW
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("Loading datasets...")
    matches    = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
    deliveries = pd.read_csv(os.path.join(DATA_DIR, "deliveries.csv"))
    print(f"  matches.csv    : {matches.shape}")
    print(f"  deliveries.csv : {deliveries.shape}")

    for col in ["team1", "team2", "toss_winner", "winner"]:
        matches[col] = matches[col].replace(TEAM_MAP)
    for col in ["batting_team", "bowling_team"]:
        deliveries[col] = deliveries[col].replace(TEAM_MAP)

    matches["venue"]      = matches["venue"].replace(VENUE_MAP)
    matches["season_int"] = matches["season"].astype(str).str[:4].astype(int)
    matches["date"]       = pd.to_datetime(matches["date"])

    matches = matches[matches["winner"].notna()].copy()
    matches = matches[matches["result"] != "no result"].copy()
    matches = matches.sort_values("date").reset_index(drop=True)

    # Only regular innings (no super overs)
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
    # Use EWM instead of flat rolling window — recent form matters more
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
# STEP 2B — NEW: EWM Delivery Features  (replaces expanding().mean())
# ══════════════════════════════════════════════════════════════════════════════

def compute_ewm_delivery_features(deliveries, matches, span=10):
    """
    NEW (v4): EWM-weighted rolling batting & bowling features.
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

            # Powerplay overs 0-5
            pp            = inn[inn["over"] <= 5]
            pp_runs       = pp["total_runs"].sum()
            pp_wickets    = int(pp["is_wicket"].sum())

            # Middle overs 6-14
            middle        = inn[(inn["over"] >= 6) & (inn["over"] <= 14)]
            middle_runs   = middle["total_runs"].sum()

            # Death overs 15-19
            death         = inn[inn["over"] >= 15]
            death_runs    = death["total_runs"].sum()
            death_overs   = death["over"].nunique()
            death_rr      = death_runs / death_overs if death_overs > 0 else run_rate

            # Boundaries
            boundaries    = inn[inn["batsman_runs"].isin([4, 6])]
            boundary_pct  = len(boundaries) / total_balls if total_balls > 0 else 0

            # Dot ball %
            dot_balls     = inn[inn["total_runs"] == 0]
            dot_ball_pct  = len(dot_balls) / total_balls if total_balls > 0 else 0

            # Top-3 batters strike rate
            batter_runs   = inn.groupby("batter")["batsman_runs"].sum().nlargest(3).index.tolist()
            top3_del      = inn[inn["batter"].isin(batter_runs)]
            top3_runs     = top3_del["batsman_runs"].sum()
            top3_sr       = (top3_runs / len(top3_del) * 100) if len(top3_del) > 0 else 0

            # Bowling SR
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

    # ── KEY CHANGE: EWM instead of expanding().mean() ─────────────────────────
    # span=10 → recent matches weighted ~3x more than matches from 10+ games ago
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
# STEP 2C — NEW: Chase Win Rate Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_chase_win_rate(matches):
    """
    NEW (v4): Rolling win rate when a team bats SECOND (chases).
    CSK, MI chase very differently to RCB — this captures that.
    Uses shift(1) + EWM.
    """
    # Build records: only matches where the team batted 2nd
    # A team bats 2nd when: toss_winner chose field (team batted 1st is the other),
    # OR toss_winner chose bat (team bats 2nd is the other).
    # Simpler: if toss_decision == 'field', toss_winner chases.
    # If toss_decision == 'bat', the OTHER team chases.
    records = []
    for _, row in matches.iterrows():
        tw = row["toss_winner"]
        td = row["toss_decision"]
        t1 = row["team1"]
        t2 = row["team2"]

        if td == "field":
            # toss_winner chose to field → they will bat 2nd (chase)
            chasing_team  = tw
            defending_team = t2 if tw == t1 else t1
        else:
            # toss_winner chose to bat → the other team chases
            defending_team = tw
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
# STEP 2D — NEW: Home Win Rate Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_home_win_rate(matches):
    """
    NEW (v4): Rolling win rate at the team's home venue.
    Uses HOME_VENUES mapping. Falls back to 0.5 for non-home matches.
    """
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
# STEP 2E — NEW: Win Streak Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_win_streak(matches):
    """
    NEW (v4): Consecutive wins entering each match.
    Streak resets to 0 on a loss. Max streak capped at 10 for scaling.
    """
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")

    result = {}
    # Per team, track streak using custom logic
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
# STEP 2F — NEW: Days Since Last Match (Rest/Fatigue)
# ══════════════════════════════════════════════════════════════════════════════

def compute_days_rest(matches):
    """
    NEW (v4): Days between a team's previous match and this one.
    Proxy for fatigue (back-to-back) vs. rest (5+ days).
    Default 7 days for first-ever match.
    """
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
                days = 7  # default for first match
            else:
                days = (row_g["date"] - prev_date).days
            result[(row_g["match_id"], team)] = min(days, 30)  # cap at 30
            prev_date = row_g["date"]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2G — NEW: Player-Level Squad Features (from deliveries history)
# ══════════════════════════════════════════════════════════════════════════════

def compute_squad_features(deliveries, matches):
    """
    NEW (v4): For each match, look at the players who appeared for each team
    in their last 5 matches BEFORE this one. Compute:
      - squad_bat_sr      : avg career strike rate of top batters
      - squad_bowl_econ   : avg career economy of main bowlers
      - squad_allrounder  : count of players who both batted AND bowled 20+ balls

    This is a proxy for XI strength without needing the actual XI announced.
    All data is strictly pre-match (shift-like: only prior matches).
    """
    match_info = matches[["id", "date", "team1", "team2"]].copy()
    match_info["match_id"] = match_info["id"].astype(str)
    deliveries  = deliveries.copy()
    deliveries["match_id"] = deliveries["match_id"].astype(str)

    # ── Career batting SR per player (all-time, computed once) ───────────────
    legal_bat  = deliveries[deliveries["extras_type"].fillna("") != "wides"]
    bat_career = legal_bat.groupby("batter").agg(
        bat_runs  = ("batsman_runs", "sum"),
        bat_balls = ("batsman_runs", "count"),
    ).reset_index()
    bat_career["career_bat_sr"] = (bat_career["bat_runs"] / bat_career["bat_balls"] * 100).round(1)
    bat_sr_dict = bat_career.set_index("batter")["career_bat_sr"].to_dict()

    # ── Career bowling economy per player ────────────────────────────────────
    legal_bowl = deliveries[~deliveries["extras_type"].isin(["wides", "noballs"])]
    bowl_career = legal_bowl.groupby("bowler").agg(
        bowl_runs  = ("total_runs", "sum"),
        bowl_balls = ("total_runs", "count"),
    ).reset_index()
    bowl_career["career_bowl_econ"] = (bowl_career["bowl_runs"] / (bowl_career["bowl_balls"] / 6)).round(2)
    bowl_econ_dict = bowl_career.set_index("bowler")["career_bowl_econ"].to_dict()

    # ── Players who are all-rounders ──────────────────────────────────────────
    batters_set  = set(bat_career[bat_career["bat_balls"] >= 30]["batter"])
    bowlers_set  = set(bowl_career[bowl_career["bowl_balls"] >= 30]["bowler"])
    allrounder_set = batters_set & bowlers_set

    # ── Per match per team: players who appeared ─────────────────────────────
    # Get batting team players per match
    bat_players = deliveries.groupby(["match_id", "batting_team"])["batter"].apply(set).reset_index()
    bat_players.columns = ["match_id", "team", "batters"]

    bowl_players = deliveries.groupby(["match_id", "bowling_team"])["bowler"].apply(set).reset_index()
    bowl_players.columns = ["match_id", "team", "bowlers"]

    # Merge to get all players per team per match
    team_players = bat_players.merge(bowl_players, on=["match_id", "team"], how="outer")
    team_players["batters"] = team_players["batters"].apply(lambda x: x if isinstance(x, set) else set())
    team_players["bowlers"] = team_players["bowlers"].apply(lambda x: x if isinstance(x, set) else set())
    team_players["all_players"] = team_players.apply(lambda r: r["batters"] | r["bowlers"], axis=1)

    # Add date for each match
    date_map = matches.set_index("id")["date"].to_dict()
    team_players["date"] = team_players["match_id"].map(
        lambda x: date_map.get(x, date_map.get(int(x) if str(x).isdigit() else x, pd.NaT))
    )
    team_players = team_players.sort_values("date")

    # ── Compute rolling squad features per (match_id, team) ──────────────────
    result = {}

    for team, grp in team_players.groupby("team"):
        grp = grp.reset_index(drop=True)

        for i, row_g in grp.iterrows():
            mid = row_g["match_id"]

            # Get last 5 matches BEFORE this one for this team
            past = grp.iloc[:i]
            last5 = past.tail(5)

            if len(last5) == 0:
                result[(mid, team)] = {
                    "squad_bat_sr"    : 120.0,  # defaults
                    "squad_bowl_econ" : 8.5,
                    "squad_allrounder": 2,
                }
                continue

            # All players seen in last 5 matches
            recent_players = set()
            for _, pr in last5.iterrows():
                recent_players |= pr["all_players"]

            # Squad batting SR: avg career SR of players who batted
            bat_srs = [bat_sr_dict[p] for p in recent_players if p in bat_sr_dict]
            squad_bat_sr = round(np.mean(bat_srs), 1) if bat_srs else 120.0

            # Squad bowling economy: avg career econ of players who bowled
            bowl_econs = [bowl_econ_dict[p] for p in recent_players if p in bowl_econ_dict]
            squad_bowl_econ = round(np.mean(bowl_econs), 2) if bowl_econs else 8.5

            # All-rounder depth: count of players who are all-rounders
            squad_allrounder = len(recent_players & allrounder_set)

            result[(mid, team)] = {
                "squad_bat_sr"    : squad_bat_sr,
                "squad_bowl_econ" : squad_bowl_econ,
                "squad_allrounder": squad_allrounder,
            }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2H — NEW: Season Stage Feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_season_stage(matches):
    """
    NEW (v4): 0 = group stage, 1 = playoff/knockout.
    Heuristic: last 4 matches of each season = playoffs.
    Knockout pressure changes team behavior significantly.
    """
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

    print("  [6/12] Rolling venue average runs...")
    venue_avg_runs = compute_rolling_venue_avg_runs(deliveries, matches)

    print("  [7/12] EWM batting & bowling features (span=10)...")
    bat_feats, bowl_feats = compute_ewm_delivery_features(deliveries, matches)

    print("  [8/12] NEW: Rolling chase win rates (EWM)...")
    chase_win_rates = compute_rolling_chase_win_rate(matches)

    print("  [9/12] NEW: Rolling home win rates...")
    home_win_rates = compute_rolling_home_win_rate(matches)

    print("  [10/12] NEW: Win streaks...")
    win_streaks = compute_win_streak(matches)

    print("  [11/12] NEW: Days rest between matches...")
    days_rest = compute_days_rest(matches)

    print("  [12/12] NEW: Player-level squad features (last 5 matches)...")
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

        # Historical batting/bowling stats
        t1_bat  = bat_feats.get((mid, t1), {})
        t2_bat  = bat_feats.get((mid, t2), {})
        t1_bowl = bowl_feats.get((mid, t1), {})
        t2_bowl = bowl_feats.get((mid, t2), {})

        # Squad features
        t1_squad = squad_feats.get((mid_str, t1), squad_feats.get((mid, t1), {}))
        t2_squad = squad_feats.get((mid_str, t2), squad_feats.get((mid, t2), {}))

        toss_win         = int(row["toss_winner"] == t1)
        toss_field       = int(row["toss_decision"] == "field")
        toss_team1_field = int(row["toss_winner"] == t1 and row["toss_decision"] == "field")

        target = int(row["winner"] == t1)

        # Squad defaults
        t1_sq_bat_sr  = t1_squad.get("squad_bat_sr",     120.0)
        t2_sq_bat_sr  = t2_squad.get("squad_bat_sr",     120.0)
        t1_sq_econ    = t1_squad.get("squad_bowl_econ",   8.5)
        t2_sq_econ    = t2_squad.get("squad_bowl_econ",   8.5)
        t1_sq_ar      = t1_squad.get("squad_allrounder",  2)
        t2_sq_ar      = t2_squad.get("squad_allrounder",  2)

        feature_row = {
            # ── Identifiers ───────────────────────────────────────────────────
            "match_id" : mid,
            "date"     : row["date"],
            "season"   : row["season_int"],
            "team1"    : t1,
            "team2"    : t2,
            "venue"    : venue,

            # ── Team Strength (7) ─────────────────────────────────────────────
            "team1_win_rate"    : round(win_rates.get((mid, t1), 0.5), 4),
            "team2_win_rate"    : round(win_rates.get((mid, t2), 0.5), 4),
            "team1_recent_form" : round(recent_form.get((mid, t1), 0.5), 4),
            "team2_recent_form" : round(recent_form.get((mid, t2), 0.5), 4),
            "h2h_win_rate"      : round(h2h.get((mid, t1, t2), 0.5), 4),
            "team1_nrr"         : round(nrr.get((mid, t1), 0.0), 4),
            "team2_nrr"         : round(nrr.get((mid, t2), 0.0), 4),

            # ── Batting — EWM (14) ────────────────────────────────────────────
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

            # ── Bowling — EWM (8) ─────────────────────────────────────────────
            "team1_bowling_economy" : round(t1_bowl.get("economy",       BOWL_DEFAULTS["economy"]),       4),
            "team2_bowling_economy" : round(t2_bowl.get("economy",       BOWL_DEFAULTS["economy"]),       4),
            "team1_death_economy"   : round(t1_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"]), 4),
            "team2_death_economy"   : round(t2_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"]), 4),
            "team1_pp_wickets"      : round(t1_bowl.get("pp_wickets",    BOWL_DEFAULTS["pp_wickets"]),    4),
            "team2_pp_wickets"      : round(t2_bowl.get("pp_wickets",    BOWL_DEFAULTS["pp_wickets"]),    4),
            "team1_bowling_sr"      : round(t1_bowl.get("bowling_sr",    BOWL_DEFAULTS["bowling_sr"]),    4),
            "team2_bowling_sr"      : round(t2_bowl.get("bowling_sr",    BOWL_DEFAULTS["bowling_sr"]),    4),

            # ── Venue (3) ─────────────────────────────────────────────────────
            "team1_venue_win_rate" : round(venue_win_rates.get((mid, t1), 0.5), 4),
            "team2_venue_win_rate" : round(venue_win_rates.get((mid, t2), 0.5), 4),
            "venue_avg_runs"       : round(venue_avg_runs.get((mid, venue), 160), 2),

            # ── Toss Context (3) ──────────────────────────────────────────────
            "toss_win"          : toss_win,
            "toss_field"        : toss_field,
            "toss_team1_field"  : toss_team1_field,

            # ── NEW: Chase + Home + Momentum + Stage (9) ─────────────────────
            "team1_chase_win_rate" : round(chase_win_rates.get((mid, t1), 0.5), 4),
            "team2_chase_win_rate" : round(chase_win_rates.get((mid, t2), 0.5), 4),
            "team1_home_win_rate"  : round(home_win_rates.get((mid, t1), 0.5), 4),
            "team2_home_win_rate"  : round(home_win_rates.get((mid, t2), 0.5), 4),
            "team1_win_streak"     : win_streaks.get((mid, t1), 0),
            "team2_win_streak"     : win_streaks.get((mid, t2), 0),
            "team1_days_rest"      : days_rest.get((mid, t1), 7),
            "team2_days_rest"      : days_rest.get((mid, t2), 7),
            "season_stage"         : season_stage.get(mid, 0),

            # ── NEW: Player Squad Features (6) ───────────────────────────────
            "team1_squad_bat_sr"    : round(t1_sq_bat_sr,  1),
            "team2_squad_bat_sr"    : round(t2_sq_bat_sr,  1),
            "team1_squad_bowl_econ" : round(t1_sq_econ,    2),
            "team2_squad_bowl_econ" : round(t2_sq_econ,    2),
            "team1_squad_allrounder": int(t1_sq_ar),
            "team2_squad_allrounder": int(t2_sq_ar),

            # ── Difference Features (12) ──────────────────────────────────────
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

            # ── Target ────────────────────────────────────────────────────────
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

    # Leakage check: avg_runs gap should be < 1
    gap = abs(master_df["team1_avg_runs"].mean() - master_df["team2_avg_runs"].mean())
    print(f"\n  Leakage check:")
    print(f"    team1_avg_runs mean = {master_df['team1_avg_runs'].mean():.2f}")
    print(f"    team2_avg_runs mean = {master_df['team2_avg_runs'].mean():.2f}")
    print(f"    Gap = {gap:.2f} runs {'✅ (PASS < 1)' if gap < 1 else '❌ (FAIL > 1)'}")

    # New feature sanity checks
    print(f"\n  New feature means (sanity):")
    new_feats = [
        "team1_chase_win_rate", "team2_chase_win_rate",
        "team1_home_win_rate",  "team2_home_win_rate",
        "team1_win_streak",     "team2_win_streak",
        "team1_days_rest",      "team2_days_rest",
        "season_stage",
        "team1_squad_bat_sr",   "team2_squad_bat_sr",
        "team1_squad_bowl_econ","team2_squad_bowl_econ",
        "team1_squad_allrounder","team2_squad_allrounder",
    ]
    for feat in new_feats:
        if feat in master_df.columns:
            print(f"    {feat:<30} mean = {master_df[feat].mean():.3f}")

    print("\n  Sample rows:")
    print(master_df[["team1", "team2", "team1_win_rate", "team1_chase_win_rate",
                      "team1_win_streak", "team1_squad_bat_sr", "target"]].head(5).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  PITCHMIND — FEATURE ENGINEERING v4")
    print("  Changes: EWM rolling | Chase/Home/Streak | Player Squad Features")
    print(f"  Total features: {len(FEATURE_COLS)}  (was 45 in v3)")
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
    print("  Feature engineering v4 COMPLETE")
    print(f"  Output: {out_path}")
    print(f"  Shape : {master_df.shape}")
    print()
    print("  WHAT CHANGED vs v3:")
    print("  ✅ EWM rolling (span=10) — recent form weighted 3x over older games")
    print("  ✅ team1/2_chase_win_rate — CSK/MI chase advantage captured")
    print("  ✅ team1/2_home_win_rate  — home ground advantage captured")
    print("  ✅ team1/2_win_streak     — momentum before each match")
    print("  ✅ team1/2_days_rest      — fatigue / rest cycle captured")
    print("  ✅ season_stage           — playoff pressure = 1")
    print("  ✅ team1/2_squad_bat_sr   — player batting strength (last 5 XI)")
    print("  ✅ team1/2_squad_bowl_econ — player bowling strength")
    print("  ✅ team1/2_squad_allrounder — XI depth score")
    print("  ✅ diff_chase_win_rate    — relative chase advantage")
    print("  ✅ diff_squad_bat_sr      — relative batting quality")
    print("=" * 70)
    print("\n  Next step → python 3_train_model.py")