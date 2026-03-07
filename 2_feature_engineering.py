"""
IPL PREDICTION MODEL — 2_feature_engineering.py  (CORRECTED v2)
================================================================
BUGS FIXED vs previous version:
  BUG 1 FIXED: team1_avg_runs was current match score -> DATA LEAKAGE
               Now: rolling historical average BEFORE each match
  BUG 2 FIXED: powerplay_runs, death_runs, boundary_pct, run_rate,
               top3_sr, bowling_economy, death_economy, pp_wickets,
               bowling_sr — all from current match -> DATA LEAKAGE
               Now: rolling historical averages using shift(1)
  BUG 3 FIXED: team1_avg_runs vs team2_avg_runs asymmetry (~7 runs)
               Now: historical avg regardless of innings
  BUG 4 FIXED: NRR was wrong (used target_runs column incorrectly)
               Now: computed from deliveries as run_rate_scored - run_rate_conceded
  BUG 5 FIXED: is_day_night hardcoded to 1 for ALL matches
               Removed from feature set (zero predictive value)
  BUG 6 FIXED: h2h_win_rate used ALL historical data including future
               Now: rolling h2h computed before each match date
  BUG 7 FIXED: team_win_rate and venue_win_rate used global (future) data
               Now: rolling cumulative win rates up to each match date

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
# TEAM & VENUE NAME STANDARDIZATION
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

# ML feature columns (v3 — 45 features: raw + difference + middle overs + dot ball)
FEATURE_COLS = [
    # Team Strength — raw (7)
    "team1_win_rate", "team2_win_rate",
    "team1_recent_form", "team2_recent_form",
    "h2h_win_rate",
    "team1_nrr", "team2_nrr",
    # Batting — raw (14)
    "team1_avg_runs", "team2_avg_runs",
    "team1_powerplay_runs", "team2_powerplay_runs",
    "team1_middle_runs", "team2_middle_runs",
    "team1_death_runs", "team2_death_runs",
    "team1_boundary_pct", "team2_boundary_pct",
    "team1_dot_ball_pct", "team2_dot_ball_pct",
    "team1_run_rate", "team2_run_rate",
    "team1_top3_sr", "team2_top3_sr",
    # Bowling — raw (8)
    "team1_bowling_economy", "team2_bowling_economy",
    "team1_death_economy", "team2_death_economy",
    "team1_pp_wickets", "team2_pp_wickets",
    "team1_bowling_sr", "team2_bowling_sr",
    # Venue (3)
    "team1_venue_win_rate", "team2_venue_win_rate",
    "venue_avg_runs",
    # Context (3)
    "toss_win", "toss_field", "toss_team1_field",
    # ── DIFFERENCE FEATURES (team1 - team2) — highest signal (10) ──────────
    "diff_win_rate",        # team1_win_rate - team2_win_rate
    "diff_recent_form",     # team1_recent_form - team2_recent_form
    "diff_avg_runs",        # team1_avg_runs - team2_avg_runs
    "diff_death_runs",      # team1_death_runs - team2_death_runs
    "diff_death_economy",   # team2_death_economy - team1_death_economy (lower=better)
    "diff_bowling_economy", # team2_bowling_economy - team1_bowling_economy
    "diff_pp_wickets",      # team1_pp_wickets - team2_pp_wickets
    "diff_run_rate",        # team1_run_rate - team2_run_rate
    "diff_nrr",             # team1_nrr - team2_nrr
    "diff_venue_win_rate",  # team1_venue_win_rate - team2_venue_win_rate
]

# Default values for teams with no prior history
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
# STEP 2 — ROLLING TEAM-LEVEL FEATURES (no data leakage)
# All features use shift(1) — only data BEFORE the current match
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_win_rate(matches):
    """
    FIX BUG 7: Rolling cumulative win rate per team up to (not including)
    each match. Uses expanding().mean() with shift(1).
    """
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
    """
    Rolling win rate in last N matches — correctly uses shift(1).
    """
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")
    df["recent_form"] = df.groupby("team")["win"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df["recent_form"] = df["recent_form"].fillna(0.5)
    return df.set_index(["match_id", "team"])["recent_form"].to_dict()


def compute_rolling_h2h(matches):
    """
    FIX BUG 6: Rolling H2H win rate computed BEFORE each match date.
    For each match, only uses H2H results from prior matches.
    """
    sorted_matches = matches.sort_values("date")
    # Build a pair key that is order-independent for lookup
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
    # Track cumulative H2H per pair
    pair_wins  = {}  # pair -> {team: wins}
    pair_total = {}  # pair -> total games

    for _, row in df.iterrows():
        mid, t1, t2, pair = row["match_id"], row["team1"], row["team2"], row["pair"]

        # Lookup BEFORE updating
        if pair not in pair_total or pair_total[pair] == 0:
            result[(mid, t1, t2)] = 0.5
        else:
            t1_wins = pair_wins[pair].get(t1, 0)
            result[(mid, t1, t2)] = round(t1_wins / pair_total[pair], 4)

        # Update counts AFTER lookup
        if pair not in pair_wins:
            pair_wins[pair] = {}
            pair_total[pair] = 0
        pair_wins[pair][t1] = pair_wins[pair].get(t1, 0) + row["t1_won"]
        pair_wins[pair][t2] = pair_wins[pair].get(t2, 0) + (1 - row["t1_won"])
        pair_total[pair] += 1

    return result


def compute_rolling_nrr(matches, deliveries):
    """
    FIX BUG 4: Correct NRR from deliveries.
    NRR = rolling avg run_rate scored - rolling avg run_rate conceded.
    """
    match_info = matches[["id", "date"]].rename(columns={"id": "match_id"})

    # Per match per batting team: run rate
    inn_stats = deliveries.groupby(["match_id", "batting_team"]).agg(
        runs=("total_runs", "sum"),
        overs=("over", "nunique")
    ).reset_index()
    inn_stats["rr"] = inn_stats["runs"] / inn_stats["overs"].clip(lower=1)

    # Get bowling_team for each (match_id, batting_team)
    bowl_map = deliveries[["match_id", "batting_team", "bowling_team"]].drop_duplicates()
    inn_stats = inn_stats.merge(bowl_map, on=["match_id", "batting_team"], how="left")
    inn_stats = inn_stats.merge(match_info, on="match_id", how="left").sort_values("date")

    # Rolling avg run rate SCORED (as batting team)
    inn_stats["hist_rr_scored"] = inn_stats.groupby("batting_team")["rr"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Build conceded stats: for bowling_team, the run rate conceded = batting team's rr
    conceded = inn_stats[["match_id", "bowling_team", "rr", "date"]].copy()
    conceded.columns = ["match_id", "team", "rr_conceded", "date"]
    conceded = conceded.sort_values("date")
    conceded["hist_rr_conceded"] = conceded.groupby("team")["rr_conceded"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Build dicts
    scored_dict = inn_stats.set_index(["match_id", "batting_team"])["hist_rr_scored"].to_dict()
    conceded_dict = conceded.set_index(["match_id", "team"])["hist_rr_conceded"].to_dict()

    nrr_dict = {}
    # Collect all (match_id, team) combinations from both sides
    all_keys = set(scored_dict.keys()) | set(conceded_dict.keys())
    for key in all_keys:
        s = scored_dict.get(key, np.nan)
        c = conceded_dict.get(key, np.nan)
        s = s if not pd.isna(s) else 8.0
        c = c if not pd.isna(c) else 8.0
        nrr_dict[key] = round(s - c, 4)

    return nrr_dict


def compute_rolling_venue_win_rate(matches):
    """
    FIX BUG 7: Rolling venue win rate per (team, venue) up to each match date.
    Falls back to team's overall rolling win rate, then 0.5.
    """
    records = []
    for _, row in matches.iterrows():
        records.append({"match_id": row["id"], "date": row["date"], "venue": row["venue"],
                         "team": row["team1"], "win": int(row["winner"] == row["team1"])})
        records.append({"match_id": row["id"], "date": row["date"], "venue": row["venue"],
                         "team": row["team2"], "win": int(row["winner"] == row["team2"])})

    df = pd.DataFrame(records).sort_values("date")

    # Rolling venue-specific win rate
    df["venue_wr"] = df.groupby(["team", "venue"])["win"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    # Rolling overall win rate as fallback
    df["overall_wr"] = df.groupby("team")["win"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["venue_win_rate"] = df["venue_wr"].fillna(df["overall_wr"]).fillna(0.5)
    return df.set_index(["match_id", "team"])["venue_win_rate"].to_dict()


def compute_rolling_venue_avg_runs(deliveries, matches):
    """
    FIX BUG 7: Rolling avg first innings score per venue before each match.
    """
    first_inn = deliveries[deliveries["inning"] == 1].copy()
    match_scores = first_inn.groupby("match_id")["total_runs"].sum().reset_index()
    match_scores.columns = ["match_id", "runs"]

    match_info = matches[["id", "date", "venue"]].rename(columns={"id": "match_id"})
    match_scores = match_scores.merge(match_info, on="match_id", how="left").sort_values("date")

    match_scores["hist_venue_avg"] = match_scores.groupby("venue")["runs"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    match_scores["hist_venue_avg"] = match_scores["hist_venue_avg"].fillna(160)

    # Some matches may have duplicate venue entries; take first per match
    deduped = match_scores.drop_duplicates(subset=["match_id", "venue"], keep="first")
    return deduped.set_index(["match_id", "venue"])["hist_venue_avg"].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ROLLING DELIVERY-LEVEL FEATURES (no data leakage)
# FIX BUGS 1, 2, 3: All delivery stats are rolling historical averages
# BEFORE each match, not from the current match itself.
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_delivery_features(deliveries, matches):
    """
    For each team, compute rolling historical averages of:
      Batting: avg_runs, powerplay_runs, death_runs, boundary_pct, run_rate, top3_sr
      Bowling: bowling_economy, death_economy, pp_wickets, bowling_sr

    All stats use shift(1).expanding().mean() — no current match data.
    """
    match_info = matches[["id", "date"]].rename(columns={"id": "match_id"})

    # ── Per match per innings: compute raw stats ───────────────────────────
    raw_batting = []
    raw_bowling = []

    for match_id, match_del in deliveries.groupby("match_id"):
        for inning in [1, 2]:
            inn = match_del[match_del["inning"] == inning]
            if len(inn) == 0:
                continue

            batting_team = inn["batting_team"].iloc[0]
            bowling_team = inn["bowling_team"].iloc[0]

            total_runs   = inn["total_runs"].sum()
            total_balls  = len(inn)
            overs_played = inn["over"].nunique()
            run_rate     = total_runs / overs_played if overs_played > 0 else 0

            # Powerplay overs 0-5
            pp         = inn[inn["over"] <= 5]
            pp_runs    = pp["total_runs"].sum()
            pp_wickets = int(pp["is_wicket"].sum())

            # Middle overs 6-14 (NEW)
            middle      = inn[(inn["over"] >= 6) & (inn["over"] <= 14)]
            middle_runs = middle["total_runs"].sum()

            # Death overs 15-19
            death       = inn[inn["over"] >= 15]
            death_runs  = death["total_runs"].sum()
            death_overs = death["over"].nunique()
            death_rr    = death_runs / death_overs if death_overs > 0 else run_rate

            # Boundaries
            boundaries   = inn[inn["batsman_runs"].isin([4, 6])]
            boundary_pct = len(boundaries) / total_balls if total_balls > 0 else 0

            # Dot ball % (NEW) — balls where total_runs == 0
            dot_balls    = inn[inn["total_runs"] == 0]
            dot_ball_pct = len(dot_balls) / total_balls if total_balls > 0 else 0

            # Top-3 batters strike rate — FIX: sort by runs scored not order of appearance
            batter_runs  = inn.groupby("batter")["batsman_runs"].sum().nlargest(3).index.tolist()
            top3_del     = inn[inn["batter"].isin(batter_runs)]
            top3_runs    = top3_del["batsman_runs"].sum()
            top3_sr      = (top3_runs / len(top3_del) * 100) if len(top3_del) > 0 else 0

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

    # ── Rolling historical averages using shift(1) ─────────────────────────
    bat_cols  = ["total_runs", "run_rate", "pp_runs", "middle_runs", "death_runs",
                 "boundary_pct", "dot_ball_pct", "top3_sr"]
    bowl_cols = ["economy", "death_economy", "pp_wickets", "bowling_sr"]

    for col in bat_cols:
        bat_df[f"hist_{col}"] = bat_df.groupby("team")[col].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

    for col in bowl_cols:
        bowl_df[f"hist_{col}"] = bowl_df.groupby("team")[col].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

    # ── Build result dicts ─────────────────────────────────────────────────
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
# STEP 4 — ASSEMBLE MASTER FEATURE DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_master_features(matches, deliveries):
    print("\nComputing all features (rolling, no leakage)...")

    print("  [1/7] Rolling team win rates...")
    win_rates = compute_rolling_win_rate(matches)

    print("  [2/7] Rolling recent form (last 5 matches)...")
    recent_form = compute_rolling_recent_form(matches)

    print("  [3/7] Rolling head-to-head win rates...")
    h2h = compute_rolling_h2h(matches)

    print("  [4/7] Rolling NRR (from deliveries)...")
    nrr = compute_rolling_nrr(matches, deliveries)

    print("  [5/7] Rolling venue win rates...")
    venue_win_rates = compute_rolling_venue_win_rate(matches)

    print("  [6/7] Rolling venue average runs...")
    venue_avg_runs = compute_rolling_venue_avg_runs(deliveries, matches)

    print("  [7/7] Rolling batting & bowling features...")
    bat_feats, bowl_feats = compute_rolling_delivery_features(deliveries, matches)

    print("\nAssembling master feature rows...")
    rows = []

    for _, row in matches.iterrows():
        mid   = row["id"]
        t1    = row["team1"]
        t2    = row["team2"]
        venue = row["venue"]

        # Historical batting stats
        t1_bat  = bat_feats.get((mid, t1), BAT_DEFAULTS)
        t2_bat  = bat_feats.get((mid, t2), BAT_DEFAULTS)

        # Historical bowling stats
        t1_bowl = bowl_feats.get((mid, t1), BOWL_DEFAULTS)
        t2_bowl = bowl_feats.get((mid, t2), BOWL_DEFAULTS)

        toss_win         = int(row["toss_winner"] == t1)
        toss_field       = int(row["toss_decision"] == "field")
        toss_team1_field = int(row["toss_winner"] == t1 and row["toss_decision"] == "field")

        target = int(row["winner"] == t1)

        feature_row = {
            # Identifiers
            "match_id" : mid,
            "date"     : row["date"],
            "season"   : row["season_int"],
            "team1"    : t1,
            "team2"    : t2,
            "venue"    : venue,

            # ── Team Strength (7) ──────────────────────────────────────────
            "team1_win_rate"    : round(win_rates.get((mid, t1), 0.5), 4),
            "team2_win_rate"    : round(win_rates.get((mid, t2), 0.5), 4),
            "team1_recent_form" : round(recent_form.get((mid, t1), 0.5), 4),
            "team2_recent_form" : round(recent_form.get((mid, t2), 0.5), 4),
            "h2h_win_rate"      : round(h2h.get((mid, t1, t2), 0.5), 4),
            "team1_nrr"         : round(nrr.get((mid, t1), 0.0), 4),
            "team2_nrr"         : round(nrr.get((mid, t2), 0.0), 4),

            # ── Batting (14) — rolling historical averages ─────────────────
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

            # ── Bowling (8) — rolling historical averages ──────────────────
            "team1_bowling_economy" : round(t1_bowl.get("economy",       BOWL_DEFAULTS["economy"]),       4),
            "team2_bowling_economy" : round(t2_bowl.get("economy",       BOWL_DEFAULTS["economy"]),       4),
            "team1_death_economy"   : round(t1_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"]), 4),
            "team2_death_economy"   : round(t2_bowl.get("death_economy", BOWL_DEFAULTS["death_economy"]), 4),
            "team1_pp_wickets"      : round(t1_bowl.get("pp_wickets",    BOWL_DEFAULTS["pp_wickets"]),    4),
            "team2_pp_wickets"      : round(t2_bowl.get("pp_wickets",    BOWL_DEFAULTS["pp_wickets"]),    4),
            "team1_bowling_sr"      : round(t1_bowl.get("bowling_sr",    BOWL_DEFAULTS["bowling_sr"]),    4),
            "team2_bowling_sr"      : round(t2_bowl.get("bowling_sr",    BOWL_DEFAULTS["bowling_sr"]),    4),

            # ── Venue (3) ──────────────────────────────────────────────────
            "team1_venue_win_rate" : round(venue_win_rates.get((mid, t1), 0.5), 4),
            "team2_venue_win_rate" : round(venue_win_rates.get((mid, t2), 0.5), 4),
            "venue_avg_runs"       : round(venue_avg_runs.get((mid, venue), 160), 2),

            # ── Context (3) — is_day_night REMOVED (was always 1) ─────────
            "toss_win"          : toss_win,
            "toss_field"        : toss_field,
            "toss_team1_field"  : toss_team1_field,

            # ── DIFFERENCE FEATURES — team1 minus team2 ───────────────────
            # These capture relative advantage directly, highest signal features
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

            # ── Target ────────────────────────────────────────────────────
            "target": target,
        }
        rows.append(feature_row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VALIDATE
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
        print("    No nulls in any feature column")
    else:
        print(nulls)

    # Verify BUG 3 fix: avg_runs gap should be < 1 run
    gap = abs(master_df["team1_avg_runs"].mean() - master_df["team2_avg_runs"].mean())
    print(f"\n  Leakage check (BUG 3 fix):")
    print(f"    team1_avg_runs mean = {master_df['team1_avg_runs'].mean():.2f}")
    print(f"    team2_avg_runs mean = {master_df['team2_avg_runs'].mean():.2f}")
    print(f"    Gap = {gap:.2f} runs {'(PASS < 1)' if gap < 1 else '(FAIL > 1)'}")

    print("\n  Feature means (sanity check):")
    categories = {
        "Team Strength" : ["team1_win_rate", "team2_win_rate", "team1_recent_form",
                           "team2_recent_form", "h2h_win_rate", "team1_nrr", "team2_nrr"],
        "Batting"       : ["team1_avg_runs", "team2_avg_runs", "team1_powerplay_runs",
                           "team2_powerplay_runs", "team1_middle_runs", "team2_middle_runs",
                           "team1_death_runs", "team2_death_runs",
                           "team1_boundary_pct", "team2_boundary_pct",
                           "team1_dot_ball_pct", "team2_dot_ball_pct",
                           "team1_run_rate", "team2_run_rate",
                           "team1_top3_sr", "team2_top3_sr"],
        "Bowling"       : ["team1_bowling_economy", "team2_bowling_economy",
                           "team1_death_economy", "team2_death_economy",
                           "team1_pp_wickets", "team2_pp_wickets",
                           "team1_bowling_sr", "team2_bowling_sr"],
        "Venue"         : ["team1_venue_win_rate", "team2_venue_win_rate", "venue_avg_runs"],
        "Context"       : ["toss_win", "toss_field", "toss_team1_field"],
        "Differences"   : ["diff_win_rate", "diff_recent_form", "diff_avg_runs",
                           "diff_death_runs", "diff_death_economy", "diff_bowling_economy",
                           "diff_pp_wickets", "diff_run_rate", "diff_nrr", "diff_venue_win_rate"],
    }
    for cat, cols in categories.items():
        print(f"\n  [{cat}]")
        for col in cols:
            if col in master_df.columns:
                print(f"    {col:<30} mean = {master_df[col].mean():.3f}")

    print("\n  Sample rows:")
    print(master_df[["team1", "team2", "team1_win_rate", "team1_recent_form",
                      "h2h_win_rate", "team1_death_economy", "target"]].head(5).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  IPL FEATURE ENGINEERING  (v3 — +difference features, +middle overs, +dot ball, top3 fix)")
    print("  Matches: 1,146  |  Features: 45")
    print("=" * 65)

    if not os.path.exists(os.path.join(DATA_DIR, "matches.csv")):
        print("ERROR: data/matches.csv not found.")
        exit(1)

    matches, deliveries = load_data()
    master_df = build_master_features(matches, deliveries)
    validate_and_print(master_df)

    out_path = os.path.join(DATA_DIR, "master_features.csv")
    master_df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("  Feature engineering completed")
    print(f"  master_features.csv created -> {out_path}")
    print(f"  Shape: {master_df.shape}")
    print("  No data leakage — all features are pre-match historical stats")
    print("  New: middle_runs, dot_ball_pct, 10 difference features, top3 fix")
    print("=" * 65)
    print("\n  Next step -> python 3_train_model.py")