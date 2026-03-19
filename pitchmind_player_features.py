"""
PITCHMIND — STEP 6: Player Stats Engine
========================================
Computes per-player batting & bowling stats from deliveries data.
Used by the dashboard's Player Scout tab.

Usage (standalone):
    python 6_player_features.py

Or import in dashboard:
    from pitchmind_player_features import get_player_stats, get_squad_stats
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR          = "data"
DELIVERIES_CLEAN  = os.path.join(DATA_DIR, "deliveries_clean.csv")
DELIVERIES_RAW    = os.path.join(DATA_DIR, "deliveries.csv")
MATCHES_CLEAN     = os.path.join(DATA_DIR, "matches_clean.csv")
MATCHES_RAW       = os.path.join(DATA_DIR, "matches.csv")
OUTPUT_PATH       = os.path.join(DATA_DIR, "player_features.csv")

# Minimum deliveries faced / bowled to be included
MIN_BALLS_BATTED  = 30
MIN_BALLS_BOWLED  = 30

# Phase over ranges
POWERPLAY   = range(0, 6)
MIDDLE      = range(6, 16)
DEATH       = range(16, 20)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_deliveries():
    """Load deliveries from clean or raw CSV."""
    for path in [DELIVERIES_CLEAN, DELIVERIES_RAW]:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.lower()
            df["over"] = pd.to_numeric(df["over"], errors="coerce").fillna(0).astype(int)
            df["batsman_runs"]  = pd.to_numeric(df["batsman_runs"],  errors="coerce").fillna(0).astype(int)
            df["extra_runs"]    = pd.to_numeric(df["extra_runs"],    errors="coerce").fillna(0).astype(int)
            df["total_runs"]    = pd.to_numeric(df["total_runs"],    errors="coerce").fillna(0).astype(int)
            df["is_wicket"]     = pd.to_numeric(df["is_wicket"],     errors="coerce").fillna(0).astype(int)
            return df
    return None


def load_matches():
    """Load matches from clean or raw CSV."""
    for path in [MATCHES_CLEAN, MATCHES_RAW]:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            # normalise id column name
            if "id" in df.columns and "match_id" not in df.columns:
                df = df.rename(columns={"id": "match_id"})
            df["match_id"] = df["match_id"].astype(str)
            df["date"]     = pd.to_datetime(df["date"], errors="coerce")
            return df
    return None


# ── BATTING STATS ─────────────────────────────────────────────────────────────
def compute_batting_stats(del_df, matches_df):
    """
    Returns DataFrame with one row per batter with their career stats.
    """
    # Legal balls = no wide (extras_type != 'wides')
    legal = del_df[del_df["extras_type"].fillna("") != "wides"].copy()

    records = []
    for batter, grp in legal.groupby("batter"):
        balls_faced = len(grp)
        if balls_faced < MIN_BALLS_BATTED:
            continue

        runs        = grp["batsman_runs"].sum()
        innings     = grp["match_id"].nunique()
        strike_rate = (runs / balls_faced * 100) if balls_faced > 0 else 0
        avg         = runs / max(innings, 1)

        # Dismissals
        dismissals = legal[
            (legal["player_dismissed"] == batter) &
            (legal["is_wicket"] == 1)
        ]["match_id"].nunique()
        bat_avg = runs / max(dismissals, 1)

        # Boundaries
        fours  = (grp["batsman_runs"] == 4).sum()
        sixes  = (grp["batsman_runs"] == 6).sum()
        dot_balls = (grp["batsman_runs"] == 0).sum()
        boundary_pct = (fours + sixes) / balls_faced if balls_faced > 0 else 0
        dot_ball_pct = dot_balls / balls_faced if balls_faced > 0 else 0

        # Phase stats
        def phase_sr(phase_range):
            ph = grp[grp["over"].isin(phase_range)]
            b = len(ph)
            r = ph["batsman_runs"].sum()
            return round(r / b * 100, 1) if b >= 6 else None

        # Recent form (last 5 matches)
        recent_matches = sorted(grp["match_id"].unique())[-5:]
        recent_grp = grp[grp["match_id"].isin(recent_matches)]
        recent_runs_list = recent_grp.groupby("match_id")["batsman_runs"].sum().values
        recent_avg  = float(np.mean(recent_runs_list)) if len(recent_runs_list) > 0 else 0
        recent_sr   = (recent_grp["batsman_runs"].sum() / max(len(recent_grp), 1) * 100)

        records.append({
            "player"           : batter,
            "role"             : "batter",
            "innings"          : innings,
            "runs"             : int(runs),
            "balls_faced"      : int(balls_faced),
            "strike_rate"      : round(strike_rate, 1),
            "batting_avg"      : round(bat_avg, 1),
            "fours"            : int(fours),
            "sixes"            : int(sixes),
            "boundary_pct"     : round(boundary_pct * 100, 1),
            "dot_ball_pct"     : round(dot_ball_pct * 100, 1),
            "pp_sr"            : phase_sr(POWERPLAY),
            "middle_sr"        : phase_sr(MIDDLE),
            "death_sr"         : phase_sr(DEATH),
            "recent_avg"       : round(recent_avg, 1),
            "recent_sr"        : round(recent_sr, 1),
        })

    return pd.DataFrame(records)


# ── BOWLING STATS ─────────────────────────────────────────────────────────────
def compute_bowling_stats(del_df):
    """
    Returns DataFrame with one row per bowler with their career stats.
    """
    records = []
    for bowler, grp in del_df.groupby("bowler"):
        # Legal balls bowled = total - no-balls - wides
        legal_bowled = grp[~grp["extras_type"].isin(["wides", "noballs"])].copy()
        balls_bowled = len(legal_bowled)
        if balls_bowled < MIN_BALLS_BOWLED:
            continue

        overs_bowled = balls_bowled / 6
        runs_given   = grp["total_runs"].sum()
        wickets      = grp[
            grp["is_wicket"] == 1
        ]["match_id"].count()  # total wicket deliveries (incl. run outs deducted below)

        # Exclude run outs from bowler's wicket count
        run_out_deliveries = grp[
            (grp["is_wicket"] == 1) &
            (grp["dismissal_kind"].fillna("") == "run out")
        ]
        wickets = max(int(wickets) - len(run_out_deliveries), 0)

        innings = grp["match_id"].nunique()
        economy = runs_given / overs_bowled if overs_bowled > 0 else 0
        bowl_sr = balls_bowled / max(wickets, 1)
        bowl_avg = runs_given / max(wickets, 1)

        dot_balls = (grp["total_runs"] == 0).sum()
        dot_ball_pct = dot_balls / balls_bowled if balls_bowled > 0 else 0

        boundary_conceded = ((grp["batsman_runs"] == 4) | (grp["batsman_runs"] == 6)).sum()
        boundary_pct = boundary_conceded / balls_bowled if balls_bowled > 0 else 0

        # Phase stats
        def phase_econ(phase_range):
            ph = grp[grp["over"].isin(phase_range)]
            b  = len(ph[~ph["extras_type"].isin(["wides", "noballs"])])
            r  = ph["total_runs"].sum()
            ov = b / 6
            return round(r / ov, 2) if ov >= 1 else None

        def phase_wkts(phase_range):
            ph = grp[grp["over"].isin(phase_range)]
            w = ph[ph["is_wicket"] == 1]
            # subtract run outs
            ro = w[w["dismissal_kind"].fillna("") == "run out"]
            return max(len(w) - len(ro), 0)

        # Recent form (last 5 matches)
        recent_matches = sorted(grp["match_id"].unique())[-5:]
        recent_grp = grp[grp["match_id"].isin(recent_matches)]
        recent_balls = len(recent_grp[~recent_grp["extras_type"].isin(["wides", "noballs"])])
        recent_runs  = recent_grp["total_runs"].sum()
        recent_econ  = (recent_runs / (recent_balls / 6)) if recent_balls >= 6 else economy
        recent_wkts  = recent_grp[recent_grp["is_wicket"] == 1].shape[0]

        records.append({
            "player"           : bowler,
            "role"             : "bowler",
            "innings"          : innings,
            "wickets"          : int(wickets),
            "balls_bowled"     : int(balls_bowled),
            "economy"          : round(economy, 2),
            "bowling_avg"      : round(bowl_avg, 1),
            "bowling_sr"       : round(bowl_sr, 1),
            "dot_ball_pct"     : round(dot_ball_pct * 100, 1),
            "boundary_pct_given": round(boundary_pct * 100, 1),
            "pp_economy"       : phase_econ(POWERPLAY),
            "middle_economy"   : phase_econ(MIDDLE),
            "death_economy"    : phase_econ(DEATH),
            "pp_wickets"       : int(phase_wkts(POWERPLAY)),
            "death_wickets"    : int(phase_wkts(DEATH)),
            "recent_economy"   : round(recent_econ, 2),
            "recent_wickets"   : int(recent_wkts),
        })

    return pd.DataFrame(records)


# ── HEAD-TO-HEAD ──────────────────────────────────────────────────────────────
def compute_h2h(del_df):
    """
    Returns a dict keyed by (batter, bowler) with matchup stats.
    """
    legal = del_df[del_df["extras_type"].fillna("") != "wides"].copy()
    h2h = {}
    for (batter, bowler), grp in legal.groupby(["batter", "bowler"]):
        balls = len(grp)
        if balls < 6:
            continue
        runs  = grp["batsman_runs"].sum()
        wkts  = grp[
            (grp["is_wicket"] == 1) &
            (grp["player_dismissed"] == batter) &
            (grp["dismissal_kind"].fillna("") != "run out")
        ].shape[0]
        h2h[(batter, bowler)] = {
            "balls": int(balls),
            "runs" : int(runs),
            "wkts" : int(wkts),
            "sr"   : round(runs / balls * 100, 1),
        }
    return h2h


# ── SQUAD STATS ENTRY POINT ───────────────────────────────────────────────────
def get_squad_stats(team1_xi, team2_xi, del_df=None, matches_df=None):
    """
    Given two lists of player names, return:
        {
          "team1": { player: {batting: {...}, bowling: {...}} },
          "team2": { ... },
          "h2h":   { (batter, bowler): {...} }
        }
    """
    if del_df is None:
        del_df = load_deliveries()
    if matches_df is None:
        matches_df = load_matches()
    if del_df is None:
        return None

    bat_df  = compute_batting_stats(del_df, matches_df)
    bowl_df = compute_bowling_stats(del_df)
    h2h     = compute_h2h(del_df)

    bat_lookup  = bat_df.set_index("player").to_dict("index") if len(bat_df) > 0 else {}
    bowl_lookup = bowl_df.set_index("player").to_dict("index") if len(bowl_df) > 0 else {}

    result = {"team1": {}, "team2": {}, "h2h": h2h}

    for squad_key, xi in [("team1", team1_xi), ("team2", team2_xi)]:
        for player in xi:
            result[squad_key][player] = {
                "batting" : bat_lookup.get(player, {}),
                "bowling" : bowl_lookup.get(player, {}),
            }

    return result


# ── SINGLE PLAYER LOOKUP ──────────────────────────────────────────────────────
def get_player_stats(player_name, del_df=None, matches_df=None):
    """
    Returns dict with batting and bowling stats for a single player.
    """
    if del_df is None:
        del_df = load_deliveries()
    if matches_df is None:
        matches_df = load_matches()
    if del_df is None:
        return {}

    bat_df  = compute_batting_stats(del_df, matches_df)
    bowl_df = compute_bowling_stats(del_df)

    bat = bat_df[bat_df["player"] == player_name].to_dict("records")
    bow = bowl_df[bowl_df["player"] == player_name].to_dict("records")

    return {
        "batting" : bat[0] if bat else {},
        "bowling" : bow[0] if bow else {},
    }


# ── GET ALL PLAYER STATS (used by dashboard) ─────────────────────────────────
def get_all_player_stats(del_df=None, matches_df=None):
    """
    Compute and return both batting and bowling DataFrames.
    Used by dashboard's Player Scout tab.

    Returns:
        tuple: (bat_df, bowl_df) DataFrames
    """
    if del_df is None:
        del_df = load_deliveries()
    if del_df is None:
        return None, None

    if matches_df is None:
        matches_df = load_matches()

    bat_df = compute_batting_stats(del_df, matches_df)
    bowl_df = compute_bowling_stats(del_df)

    return bat_df, bowl_df


# ── SEARCH PLAYERS BY PARTIAL NAME ────────────────────────────────────────────
def search_players(query, del_df=None):
    """Return list of player names matching query (case-insensitive)."""
    if del_df is None:
        del_df = load_deliveries()
    if del_df is None:
        return []

    all_batters = set(del_df["batter"].dropna().unique())
    all_bowlers = set(del_df["bowler"].dropna().unique())
    all_players = sorted(all_batters | all_bowlers)

    q = query.lower()
    return [p for p in all_players if q in p.lower()]


# ── MAIN: Build & Save Full Player Features ───────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PITCHMIND — Player Features Builder")
    print("=" * 60)

    del_df     = load_deliveries()
    matches_df = load_matches()

    if del_df is None:
        print("❌ deliveries.csv not found in data/")
        print("   Run 0_json_to_csv.py first, then 1_data_cleaning.py")
        exit(1)

    print(f"\n✅ Deliveries loaded  →  {del_df.shape[0]:,} rows")
    print(f"✅ Matches loaded     →  {matches_df.shape[0] if matches_df is not None else 0:,} rows")

    print("\n── Computing batting stats ──────────────────────────────")
    bat_df = compute_batting_stats(del_df, matches_df)
    print(f"   Batters with ≥{MIN_BALLS_BATTED} balls  →  {len(bat_df)} players")

    print("── Computing bowling stats ──────────────────────────────")
    bowl_df = compute_bowling_stats(del_df)
    print(f"   Bowlers with ≥{MIN_BALLS_BOWLED} balls  →  {len(bowl_df)} players")

    # Save combined
    bat_df.to_csv(os.path.join(DATA_DIR, "player_batting_stats.csv"), index=False)
    bowl_df.to_csv(os.path.join(DATA_DIR, "player_bowling_stats.csv"), index=False)

    print(f"\n✅ Saved → data/player_batting_stats.csv")
    print(f"✅ Saved → data/player_bowling_stats.csv")

    # Sample output
    print("\n── Sample: Top 10 Batters by Strike Rate ────────────────")
    top = bat_df.nlargest(10, "strike_rate")[["player", "innings", "runs", "strike_rate", "batting_avg"]]
    print(top.to_string(index=False))

    print("\n── Sample: Top 10 Bowlers by Economy ───────────────────")
    top_b = bowl_df.nsmallest(10, "economy")[["player", "innings", "wickets", "economy", "bowling_avg"]]
    print(top_b.to_string(index=False))

    print("\n" + "=" * 60)
    print("  DONE — import this module in 4_dashboard.py")
    print("=" * 60)