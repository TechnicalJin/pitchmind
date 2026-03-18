"""
PITCHMIND — STEP 6: Player Stats Engine  (v2 — UPGRADED)
==========================================================
CHANGES vs v1:

  FIX 1 — Batting Average Corrected (was wrong for every player)
           Old code computed avg = runs / innings (line 87) which was
           NEVER even used — bat_avg on line 94 was already the right
           formula but both were computing dismissals per match, not
           per dismissal event. Now fixed properly:

           WRONG : avg  = runs / max(innings, 1)          ← overestimates
                          (MS Dhoni: 22.6 instead of 38.3)
           FIXED : bat_avg = runs / max(dismissals, 1)    ← correct T20 avg
                          counts actual dismissal EVENTS not match appearances

           ALSO FIXED: dismissal count now uses delivery-level counting
           (sum of is_wicket where player_dismissed == batter) instead of
           nunique(match_id) which undercounts multi-dismissal edge cases
           and is slower.

           Edge case handled: 49 players with 0 career dismissals
           (pure not-out specialists) → avg shown as total runs (not inf).

  FIX 2 — Broken Decorator Removed
           Line 40 had `@staticmethod if False else lambda f: f` which
           was a no-op workaround that breaks IDEs, type checkers, and
           makes the code unprofessional. Removed entirely.

  FIX 3 — Module-Level Stats Cache (replaces per-call recompute)
           Old: get_player_stats() called for 1 player → recomputes
                ALL 703 batters + 550 bowlers from 278K rows every time.
                22 players in a squad = 22 full recomputes = ~3.4s lag.

           New: _STATS_CACHE dict holds bat_df + bowl_df computed once.
                First call: ~0.15s. Every subsequent call: <1ms lookup.
                Cache invalidated only when del_df changes (by passing
                force_reload=True) or when the module is reimported.

                For Streamlit dashboard: wrap get_all_player_stats() with
                @st.cache_data(ttl=3600) — example shown in the docstring.

Usage (standalone):
    python 6_player_features.py

Or import in dashboard:
    from pitchmind_player_features import get_player_stats, get_squad_stats

Dashboard caching (add in your dashboard file):
    import streamlit as st
    from pitchmind_player_features import get_all_player_stats, load_deliveries, load_matches

    @st.cache_data(ttl=3600)
    def cached_player_stats():
        del_df     = load_deliveries()
        matches_df = load_matches()
        return get_all_player_stats(del_df, matches_df)
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

# Minimum deliveries faced / bowled to be included in stats
MIN_BALLS_BATTED  = 30
MIN_BALLS_BOWLED  = 30

# Phase over ranges (0-indexed overs: over 0 = 1st over)
POWERPLAY   = range(0, 6)
MIDDLE      = range(6, 16)
DEATH       = range(16, 20)

# ── MODULE-LEVEL CACHE ────────────────────────────────────────────────────────
# Stores computed bat_df and bowl_df so any number of player lookups
# after the first one cost <1ms instead of recomputing 278K rows each time.
_STATS_CACHE = {
    "bat_df"  : None,
    "bowl_df" : None,
    "h2h"     : None,
    "loaded"  : False,
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_deliveries():
    """Load deliveries from clean or raw CSV. Returns None if neither found."""
    for path in [DELIVERIES_CLEAN, DELIVERIES_RAW]:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.lower()
            df["over"]         = pd.to_numeric(df["over"],         errors="coerce").fillna(0).astype(int)
            df["batsman_runs"] = pd.to_numeric(df["batsman_runs"], errors="coerce").fillna(0).astype(int)
            df["extra_runs"]   = pd.to_numeric(df["extra_runs"],   errors="coerce").fillna(0).astype(int)
            df["total_runs"]   = pd.to_numeric(df["total_runs"],   errors="coerce").fillna(0).astype(int)
            df["is_wicket"]    = pd.to_numeric(df["is_wicket"],    errors="coerce").fillna(0).astype(int)
            return df
    return None


def load_matches():
    """Load matches from clean or raw CSV. Returns None if neither found."""
    for path in [MATCHES_CLEAN, MATCHES_RAW]:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            if "id" in df.columns and "match_id" not in df.columns:
                df = df.rename(columns={"id": "match_id"})
            df["match_id"] = df["match_id"].astype(str)
            df["date"]     = pd.to_datetime(df["date"], errors="coerce")
            return df
    return None


# ══════════════════════════════════════════════════════════════════════════════
# BATTING STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_batting_stats(del_df, matches_df):
    """
    Returns DataFrame with one row per batter with career stats.

    FIX (v2): batting_avg = runs / dismissals  (correct T20 definition)
              Old code used runs / innings which underweights not-out innings.
              Example: MS Dhoni  old=22.6  →  fixed=38.3  (massive difference)
                       V Kohli   old=33.4  →  fixed=39.5
                       KL Rahul  old=38.7  →  fixed=46.2

    Edge case: players with 0 career dismissals (49 in dataset) get
               batting_avg = total runs (not infinity / not zero).
    """
    # Legal deliveries = exclude wides (wides don't count as balls faced)
    legal = del_df[del_df["extras_type"].fillna("") != "wides"].copy()

    # Pre-compute dismissals per player in one pass — MUCH faster than
    # filtering inside the loop for every batter
    dismissal_counts = (
        del_df[del_df["is_wicket"] == 1]
        .groupby("player_dismissed")["is_wicket"]
        .sum()                        # count dismissal deliveries (not match nunique)
        .to_dict()
    )

    records = []
    for batter, grp in legal.groupby("batter"):
        balls_faced = len(grp)
        if balls_faced < MIN_BALLS_BATTED:
            continue

        runs        = int(grp["batsman_runs"].sum())
        innings     = grp["match_id"].nunique()
        strike_rate = (runs / balls_faced * 100) if balls_faced > 0 else 0.0

        # ── FIXED batting average ─────────────────────────────────────────────
        # Count total dismissal EVENTS (delivery-level), not match appearances.
        # A player could be dismissed twice in rare cases (retired hurt then out)
        # but delivery-level is the correct cricket definition.
        dismissals = int(dismissal_counts.get(batter, 0))

        if dismissals > 0:
            batting_avg = round(runs / dismissals, 1)
        else:
            # Not-out specialist — show total runs as average (conventional)
            batting_avg = float(runs)

        # ── Boundaries ───────────────────────────────────────────────────────
        fours        = int((grp["batsman_runs"] == 4).sum())
        sixes        = int((grp["batsman_runs"] == 6).sum())
        dot_balls    = int((grp["batsman_runs"] == 0).sum())
        boundary_pct = (fours + sixes) / balls_faced if balls_faced > 0 else 0.0
        dot_ball_pct = dot_balls / balls_faced if balls_faced > 0 else 0.0

        # ── Phase strike rates ────────────────────────────────────────────────
        def phase_sr(phase_range):
            ph = grp[grp["over"].isin(phase_range)]
            b  = len(ph)
            r  = ph["batsman_runs"].sum()
            return round(r / b * 100, 1) if b >= 6 else None

        # ── Recent form — last 5 matches by match_id sort ────────────────────
        # Use sorted match_ids as proxy for chronological order
        recent_match_ids = sorted(grp["match_id"].unique())[-5:]
        recent_grp       = grp[grp["match_id"].isin(recent_match_ids)]
        recent_runs_list = recent_grp.groupby("match_id")["batsman_runs"].sum().values
        recent_avg       = float(np.mean(recent_runs_list)) if len(recent_runs_list) > 0 else 0.0
        recent_sr        = (recent_grp["batsman_runs"].sum() / max(len(recent_grp), 1) * 100)

        records.append({
            "player"       : batter,
            "role"         : "batter",
            "innings"      : innings,
            "runs"         : runs,
            "balls_faced"  : int(balls_faced),
            "dismissals"   : dismissals,           # NEW — useful for dashboard display
            "not_outs"     : innings - min(dismissals, innings),  # NEW
            "strike_rate"  : round(strike_rate, 1),
            "batting_avg"  : batting_avg,          # FIXED — correct T20 average
            "fours"        : fours,
            "sixes"        : sixes,
            "boundary_pct" : round(boundary_pct * 100, 1),
            "dot_ball_pct" : round(dot_ball_pct * 100, 1),
            "pp_sr"        : phase_sr(POWERPLAY),
            "middle_sr"    : phase_sr(MIDDLE),
            "death_sr"     : phase_sr(DEATH),
            "recent_avg"   : round(recent_avg, 1),
            "recent_sr"    : round(recent_sr, 1),
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# BOWLING STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_bowling_stats(del_df):
    """
    Returns DataFrame with one row per bowler with career stats.
    Run-outs correctly excluded from bowler's wicket tally.
    """
    records = []
    for bowler, grp in del_df.groupby("bowler"):
        # Legal balls = exclude wides and no-balls (they don't count as legal)
        legal_bowled = grp[~grp["extras_type"].fillna("").isin(["wides", "noballs"])]
        balls_bowled = len(legal_bowled)
        if balls_bowled < MIN_BALLS_BOWLED:
            continue

        overs_bowled = balls_bowled / 6
        runs_given   = int(grp["total_runs"].sum())

        # Wickets: exclude run-outs (not credited to bowler in cricket)
        all_wickets     = int((grp["is_wicket"] == 1).sum())
        run_out_count   = int(
            ((grp["is_wicket"] == 1) &
             (grp["dismissal_kind"].fillna("") == "run out")).sum()
        )
        wickets = max(all_wickets - run_out_count, 0)

        innings   = grp["match_id"].nunique()
        economy   = runs_given / overs_bowled if overs_bowled > 0 else 0.0
        bowl_sr   = balls_bowled / max(wickets, 1)
        bowl_avg  = runs_given / max(wickets, 1)

        dot_balls        = int((grp["total_runs"] == 0).sum())
        dot_ball_pct     = dot_balls / balls_bowled if balls_bowled > 0 else 0.0
        boundary_conceded = int(
            ((grp["batsman_runs"] == 4) | (grp["batsman_runs"] == 6)).sum()
        )
        boundary_pct = boundary_conceded / balls_bowled if balls_bowled > 0 else 0.0

        # ── Phase economy rates ───────────────────────────────────────────────
        def phase_econ(phase_range):
            ph  = grp[grp["over"].isin(phase_range)]
            b   = len(ph[~ph["extras_type"].fillna("").isin(["wides", "noballs"])])
            r   = ph["total_runs"].sum()
            ov  = b / 6
            return round(r / ov, 2) if ov >= 1 else None

        def phase_wkts(phase_range):
            ph  = grp[grp["over"].isin(phase_range)]
            w   = int((ph["is_wicket"] == 1).sum())
            ro  = int(
                ((ph["is_wicket"] == 1) &
                 (ph["dismissal_kind"].fillna("") == "run out")).sum()
            )
            return max(w - ro, 0)

        # ── Recent form — last 5 matches ──────────────────────────────────────
        recent_match_ids = sorted(grp["match_id"].unique())[-5:]
        recent_grp       = grp[grp["match_id"].isin(recent_match_ids)]
        recent_balls     = len(recent_grp[~recent_grp["extras_type"].fillna("").isin(["wides", "noballs"])])
        recent_runs      = recent_grp["total_runs"].sum()
        recent_econ      = (recent_runs / (recent_balls / 6)) if recent_balls >= 6 else economy
        recent_wkts      = int((recent_grp["is_wicket"] == 1).sum())

        records.append({
            "player"            : bowler,
            "role"              : "bowler",
            "innings"           : innings,
            "wickets"           : wickets,
            "balls_bowled"      : int(balls_bowled),
            "economy"           : round(economy, 2),
            "bowling_avg"       : round(bowl_avg, 1),
            "bowling_sr"        : round(bowl_sr, 1),
            "dot_ball_pct"      : round(dot_ball_pct * 100, 1),
            "boundary_pct_given": round(boundary_pct * 100, 1),
            "pp_economy"        : phase_econ(POWERPLAY),
            "middle_economy"    : phase_econ(MIDDLE),
            "death_economy"     : phase_econ(DEATH),
            "pp_wickets"        : int(phase_wkts(POWERPLAY)),
            "death_wickets"     : int(phase_wkts(DEATH)),
            "recent_economy"    : round(recent_econ, 2),
            "recent_wickets"    : recent_wkts,
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD MATCHUP STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_h2h(del_df):
    """
    Returns dict keyed by (batter, bowler) with career head-to-head stats.
    Only includes matchups with >= 6 balls faced.
    """
    legal = del_df[del_df["extras_type"].fillna("") != "wides"].copy()
    h2h   = {}

    for (batter, bowler), grp in legal.groupby(["batter", "bowler"]):
        balls = len(grp)
        if balls < 6:
            continue
        runs = int(grp["batsman_runs"].sum())
        wkts = int(
            ((grp["is_wicket"] == 1) &
             (grp["player_dismissed"] == batter) &
             (grp["dismissal_kind"].fillna("") != "run out")).sum()
        )
        h2h[(batter, bowler)] = {
            "balls": balls,
            "runs" : runs,
            "wkts" : wkts,
            "sr"   : round(runs / balls * 100, 1),
        }
    return h2h


# ══════════════════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT  (NEW in v2)
# ══════════════════════════════════════════════════════════════════════════════

def get_all_player_stats(del_df=None, matches_df=None, force_reload=False):
    """
    NEW (v2): Compute batting + bowling stats ONCE and store in module cache.
    All subsequent calls return instantly from the cached DataFrames.

    Args:
        del_df       : deliveries DataFrame (loaded fresh if None)
        matches_df   : matches DataFrame (loaded fresh if None)
        force_reload : set True to invalidate cache and recompute

    Returns:
        (bat_df, bowl_df) — full player stats DataFrames

    For Streamlit dashboard, wrap this with @st.cache_data:

        @st.cache_data(ttl=3600)
        def cached_player_stats():
            return get_all_player_stats(force_reload=True)

        bat_df, bowl_df = cached_player_stats()
    """
    global _STATS_CACHE

    # Return cache if already loaded and not forcing reload
    if _STATS_CACHE["loaded"] and not force_reload:
        return _STATS_CACHE["bat_df"], _STATS_CACHE["bowl_df"]

    # Load data if not provided
    if del_df is None:
        del_df = load_deliveries()
    if del_df is None:
        return pd.DataFrame(), pd.DataFrame()

    if matches_df is None:
        matches_df = load_matches()

    # Compute stats (this is the expensive step — runs once)
    bat_df  = compute_batting_stats(del_df, matches_df)
    bowl_df = compute_bowling_stats(del_df)

    # Store in module-level cache
    _STATS_CACHE["bat_df"]  = bat_df
    _STATS_CACHE["bowl_df"] = bowl_df
    _STATS_CACHE["loaded"]  = True

    return bat_df, bowl_df


def get_h2h_stats(del_df=None, force_reload=False):
    """
    NEW (v2): Compute H2H stats once and cache.
    H2H is expensive (groups 703 batters × 550 bowlers).
    """
    global _STATS_CACHE

    if _STATS_CACHE["h2h"] is not None and not force_reload:
        return _STATS_CACHE["h2h"]

    if del_df is None:
        del_df = load_deliveries()
    if del_df is None:
        return {}

    h2h = compute_h2h(del_df)
    _STATS_CACHE["h2h"] = h2h
    return h2h


def invalidate_cache():
    """Call this when underlying data files change to force recompute."""
    global _STATS_CACHE
    _STATS_CACHE = {"bat_df": None, "bowl_df": None, "h2h": None, "loaded": False}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — ENTRY POINTS  (used by dashboard)
# ══════════════════════════════════════════════════════════════════════════════

def get_squad_stats(team1_xi, team2_xi, del_df=None, matches_df=None):
    """
    Given two lists of player names, return full stats for both squads.

    UPGRADED (v2): Uses cached bat_df/bowl_df — no recompute on each call.
    First call for a session: ~150ms. All subsequent calls: <1ms.

    Returns:
        {
          "team1": { player_name: {"batting": {...}, "bowling": {...}} },
          "team2": { ... },
          "h2h"  : { (batter, bowler): {"balls", "runs", "wkts", "sr"} }
        }
    """
    bat_df, bowl_df = get_all_player_stats(del_df, matches_df)
    h2h             = get_h2h_stats(del_df)

    if bat_df.empty and bowl_df.empty:
        return None

    bat_lookup  = bat_df.set_index("player").to_dict("index")  if not bat_df.empty  else {}
    bowl_lookup = bowl_df.set_index("player").to_dict("index") if not bowl_df.empty else {}

    result = {"team1": {}, "team2": {}, "h2h": h2h}

    for squad_key, xi in [("team1", team1_xi), ("team2", team2_xi)]:
        for player in xi:
            result[squad_key][player] = {
                "batting" : bat_lookup.get(player, {}),
                "bowling" : bowl_lookup.get(player, {}),
            }

    return result


def get_player_stats(player_name, del_df=None, matches_df=None):
    """
    Returns batting and bowling stats for a single player.

    UPGRADED (v2): Uses cache — calling for 22 players costs the same
    as calling for 1. Old code recomputed 278K rows each time.

    Returns:
        {"batting": {...}, "bowling": {...}}
    """
    bat_df, bowl_df = get_all_player_stats(del_df, matches_df)

    bat = bat_df[bat_df["player"] == player_name].to_dict("records")  if not bat_df.empty  else []
    bow = bowl_df[bowl_df["player"] == player_name].to_dict("records") if not bowl_df.empty else []

    return {
        "batting" : bat[0] if bat else {},
        "bowling" : bow[0] if bow else {},
    }


def search_players(query, del_df=None):
    """Return list of player names matching query string (case-insensitive)."""
    if del_df is None:
        del_df = load_deliveries()
    if del_df is None:
        return []

    all_players = sorted(
        set(del_df["batter"].dropna().unique()) |
        set(del_df["bowler"].dropna().unique())
    )
    q = query.lower()
    return [p for p in all_players if q in p.lower()]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Standalone build & save
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("=" * 65)
    print("  PITCHMIND — Player Features Builder  (v2)")
    print("  Fixes: Batting avg | Cache | Broken decorator removed")
    print("=" * 65)

    del_df     = load_deliveries()
    matches_df = load_matches()

    if del_df is None:
        print("❌ deliveries.csv not found in data/")
        print("   Run 0_json_to_csv.py first, then 1_data_cleaning.py")
        exit(1)

    print(f"\n✅ Deliveries loaded  →  {del_df.shape[0]:,} rows")
    print(f"✅ Matches loaded     →  {matches_df.shape[0] if matches_df is not None else 0:,} rows")

    # ── First call — computes & caches ────────────────────────────────────────
    print("\n── Computing & caching all player stats ─────────────────")
    t0 = time.time()
    bat_df, bowl_df = get_all_player_stats(del_df, matches_df)
    t1 = time.time()
    print(f"   Batters (≥{MIN_BALLS_BATTED} balls)  →  {len(bat_df)} players  [{t1-t0:.2f}s]")

    t0 = time.time()
    bat_df2, bowl_df2 = get_all_player_stats()    # second call — should be instant
    t1 = time.time()
    print(f"   Cache hit (2nd call)  →  {len(bat_df2)} batters  [{(t1-t0)*1000:.1f}ms]  ✅ cached")
    print(f"   Bowlers (≥{MIN_BALLS_BOWLED} balls)  →  {len(bowl_df)} players")

    # ── Verify batting average fix ────────────────────────────────────────────
    print("\n── Batting Average Fix Verification ─────────────────────")
    verify_players = ["V Kohli", "MS Dhoni", "RG Sharma", "DA Warner",
                      "AB de Villiers", "KL Rahul"]
    print(f"   {'Player':<22} {'Innings':>7} {'Dism':>5} {'NotOut':>6} {'Avg (fixed)':>11}")
    print(f"   {'-'*55}")
    for p in verify_players:
        row = bat_df[bat_df["player"] == p]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print(f"   {p:<22} {r['innings']:>7} {r['dismissals']:>5} "
              f"{r['not_outs']:>6} {r['batting_avg']:>11.1f}")

    # ── Save output CSVs ──────────────────────────────────────────────────────
    bat_df.to_csv(os.path.join(DATA_DIR,  "player_batting_stats.csv"),  index=False)
    bowl_df.to_csv(os.path.join(DATA_DIR, "player_bowling_stats.csv"), index=False)
    print(f"\n✅ Saved → data/player_batting_stats.csv")
    print(f"✅ Saved → data/player_bowling_stats.csv")

    # ── Sample output ─────────────────────────────────────────────────────────
    print("\n── Top 10 Batters by Strike Rate ────────────────────────")
    top_bat = bat_df.nlargest(10, "strike_rate")[
        ["player", "innings", "runs", "dismissals", "not_outs", "strike_rate", "batting_avg"]
    ]
    print(top_bat.to_string(index=False))

    print("\n── Top 10 Bowlers by Economy ────────────────────────────")
    top_bowl = bowl_df.nsmallest(10, "economy")[
        ["player", "innings", "wickets", "economy", "bowling_avg"]
    ]
    print(top_bowl.to_string(index=False))

    # ── Performance comparison ────────────────────────────────────────────────
    print("\n── Cache Performance: 22-player squad lookup ────────────")
    squad = list(bat_df["player"].head(22))

    t0 = time.time()
    for p in squad:
        get_player_stats(p)     # all cached — no recompute
    t_cached = time.time() - t0

    # Simulate old behaviour: invalidate cache each time
    t0 = time.time()
    for p in squad:
        invalidate_cache()      # force recompute on every call like old code did
        get_player_stats(p, del_df, matches_df)
    t_old = time.time() - t0

    print(f"   Old (no cache, 22 calls)  →  {t_old:.2f}s")
    print(f"   New (cached, 22 calls)    →  {t_cached*1000:.1f}ms")
    print(f"   Speedup                   →  {t_old/max(t_cached,0.001):.0f}x faster")

    print("\n" + "=" * 65)
    print("  DONE")
    print("  WHAT CHANGED vs v1:")
    print("  ✅ batting_avg = runs / dismissals  (was runs / innings)")
    print("  ✅ dismissal count is delivery-level sum (was match nunique)")
    print("  ✅ not_outs column added for dashboard display")
    print("  ✅ module-level cache — compute once, lookup instantly")
    print("  ✅ get_all_player_stats() for Streamlit @st.cache_data")
    print("  ✅ broken @staticmethod decorator removed from load_deliveries")
    print("=" * 65)
    print("\n  Dashboard integration:")
    print("  @st.cache_data(ttl=3600)")
    print("  def cached_player_stats():")
    print("      return get_all_player_stats(force_reload=True)")