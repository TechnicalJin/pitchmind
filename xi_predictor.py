"""
PITCHMIND — xi_predictor.py
============================
Playing XI-aware prediction engine.

This module bridges the gap between player-level stats and match prediction.
When a Playing XI is provided, it computes XI-level batting/bowling features
and applies them as an ADJUSTMENT on top of the base team-history prediction.

WHY THIS APPROACH:
  The base model was trained on team-level rolling stats (avg_runs, economy etc).
  We cannot replace those features — the model doesn't know individual players.
  Instead, we:
    1. Compute the "expected" team-level stats from the actual XI
    2. Find the difference vs what the base model used
    3. Apply a bounded adjustment to the raw probability

This gives you real XI influence without retraining the model.

USAGE (in 4_dashboard.py):
    from xi_predictor import compute_xi_adjustment, get_xi_feature_summary

    # After getting base prob_t1, prob_t2 from predict_winner():
    xi_adj = compute_xi_adjustment(
        team1_xi, team2_xi,
        bat_df, bowl_df,
        base_avg_runs_t1, base_avg_runs_t2,
        base_economy_t1, base_economy_t2,
    )
    adjusted_prob_t1 = clip_prob(prob_t1 + xi_adj)
    adjusted_prob_t2 = 1.0 - adjusted_prob_t1
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict

# ── CONFIG ────────────────────────────────────────────────────────────────────

# How much the XI adjustment can shift the raw probability (max ±8%)
# Keeps the model from being overridden by individual player stats alone
XI_MAX_ADJUSTMENT = 0.08

# Weights for batting vs bowling contribution to adjustment
BATTING_WEIGHT  = 0.55   # batting matters slightly more in T20
BOWLING_WEIGHT  = 0.45

# 2023+ era reference values (used for normalization)
ERA_AVG_BAT_SR   = 143.0   # avg strike rate for batters in 2023+ IPL
ERA_AVG_BOWL_ECO = 9.8     # avg economy for bowlers in 2023+ IPL
ERA_AVG_RUNS     = 185.0   # avg first innings score

# Min innings/matches for a player to be considered "reliable" stats
MIN_INNINGS_RELIABLE = 5


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_get(stats: dict, key: str, default: float) -> float:
    val = stats.get(key, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def clip_prob(p: float) -> float:
    """Clip probability to [0.05, 0.95] — never predict certainty."""
    return max(0.05, min(0.95, p))


def _resolve_player(name: str, bat_lookup: dict, bowl_lookup: dict,
                    use_resolver: bool = True) -> Tuple[dict, dict]:
    """
    Looks up player in batting and bowling dicts.
    Tries name resolution if available.
    Returns (batting_stats, bowling_stats) — empty dicts if not found.
    """
    if use_resolver:
        try:
            from name_resolver import resolve_name
            resolved = resolve_name(name)
            if resolved:
                name = resolved
        except ImportError:
            pass

    bat = bat_lookup.get(name, {})
    bow = bowl_lookup.get(name, {})
    return bat, bow


def _classify_players(xi: List[str], bat_lookup: dict, bowl_lookup: dict) -> Dict:
    """
    For a given XI, classify each player as batter / bowler / allrounder
    and return aggregated squad-level batting and bowling metrics.

    Returns dict with:
      - squad_avg_sr:     avg strike rate of top-7 batters in XI
      - squad_avg_eco:    avg economy of top-4 bowlers in XI
      - squad_avg_bat_avg: avg batting average
      - squad_pp_sr:      avg powerplay SR
      - squad_death_sr:   avg death SR
      - squad_pp_eco:     avg PP economy
      - squad_death_eco:  avg death economy
      - allrounder_count: number of true allrounders
      - reliable_batters: count with ≥ MIN_INNINGS_RELIABLE innings
      - reliable_bowlers: count with ≥ MIN_INNINGS_RELIABLE matches
    """
    bat_srs     = []
    bat_avgs    = []
    pp_srs      = []
    death_srs   = []
    bowl_ecos   = []
    pp_ecos     = []
    death_ecos  = []
    allrounders = 0
    rel_bat     = 0
    rel_bowl    = 0

    for player in xi:
        bat, bow = _resolve_player(player, bat_lookup, bowl_lookup)

        has_bat  = bool(bat) and _safe_get(bat, "innings", 0) >= 3
        has_bowl = bool(bow) and _safe_get(bow, "innings", 0) >= 3

        if has_bat:
            sr = _safe_get(bat, "strike_rate", ERA_AVG_BAT_SR)
            if sr > 50:  # sanity filter
                bat_srs.append(sr)
                bat_avgs.append(_safe_get(bat, "batting_avg", 25.0))

            pp_sr = bat.get("pp_sr")
            if pp_sr and pp_sr > 50:
                pp_srs.append(float(pp_sr))

            d_sr = bat.get("death_sr")
            if d_sr and d_sr > 50:
                death_srs.append(float(d_sr))

            if _safe_get(bat, "innings", 0) >= MIN_INNINGS_RELIABLE:
                rel_bat += 1

        if has_bowl:
            eco = _safe_get(bow, "economy", ERA_AVG_BOWL_ECO)
            if 4 < eco < 16:  # sanity filter
                bowl_ecos.append(eco)

            pp_eco = bow.get("pp_economy")
            if pp_eco and 4 < float(pp_eco) < 14:
                pp_ecos.append(float(pp_eco))

            d_eco = bow.get("death_economy")
            if d_eco and 4 < float(d_eco) < 18:
                death_ecos.append(float(d_eco))

            if _safe_get(bow, "innings", 0) >= MIN_INNINGS_RELIABLE:
                rel_bowl += 1

        if has_bat and has_bowl:
            allrounders += 1

    # Use top-7 batters by SR (most impactful)
    top7_srs = sorted(bat_srs, reverse=True)[:7]

    return {
        "squad_avg_sr"      : float(np.mean(top7_srs))  if top7_srs  else ERA_AVG_BAT_SR,
        "squad_avg_bat_avg" : float(np.mean(bat_avgs))   if bat_avgs  else 25.0,
        "squad_pp_sr"       : float(np.mean(pp_srs))     if pp_srs    else 140.0,
        "squad_death_sr"    : float(np.mean(death_srs))  if death_srs else 155.0,
        "squad_avg_eco"     : float(np.mean(bowl_ecos))  if bowl_ecos else ERA_AVG_BOWL_ECO,
        "squad_pp_eco"      : float(np.mean(pp_ecos))    if pp_ecos   else 9.5,
        "squad_death_eco"   : float(np.mean(death_ecos)) if death_ecos else 11.2,
        "allrounder_count"  : allrounders,
        "reliable_batters"  : rel_bat,
        "reliable_bowlers"  : rel_bowl,
        "players_found"     : len(bat_srs) + len(bowl_ecos),
        "total_xi"          : len(xi),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: COMPUTE XI ADJUSTMENT
# ══════════════════════════════════════════════════════════════════════════════

def compute_xi_adjustment(
    team1_xi: List[str],
    team2_xi: List[str],
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    base_avg_runs_t1: float = ERA_AVG_RUNS,
    base_avg_runs_t2: float = ERA_AVG_RUNS,
    base_economy_t1 : float = ERA_AVG_BOWL_ECO,
    base_economy_t2 : float = ERA_AVG_BOWL_ECO,
) -> Tuple[float, dict]:
    """
    Computes how much to adjust the base win probability based on XI quality.

    Returns:
        (adjustment, debug_info)
        adjustment: float in [-XI_MAX_ADJUSTMENT, +XI_MAX_ADJUSTMENT]
                    positive = favours team1, negative = favours team2
        debug_info: dict with breakdown for display

    Logic:
      1. Compute XI batting strength score for each team (via avg SR vs era baseline)
      2. Compute XI bowling strength score (via avg economy vs era baseline)
      3. Batting advantage = (t1_bat_score - t2_bat_score) normalized
      4. Bowling advantage = (t2_bowl_score - t1_bowl_score) normalized
         (lower economy = better, so we invert)
      5. Combined adjustment = BATTING_WEIGHT * bat_adv + BOWLING_WEIGHT * bowl_adv
      6. Scale to [-XI_MAX_ADJUSTMENT, +XI_MAX_ADJUSTMENT]
    """
    if not team1_xi and not team2_xi:
        return 0.0, {"note": "No XI provided, no adjustment applied"}

    # Build lookups
    bat_lookup  = bat_df.set_index("player").to_dict("index") if bat_df is not None and len(bat_df) > 0 else {}
    bowl_lookup = bowl_df.set_index("player").to_dict("index") if bowl_df is not None and len(bowl_df) > 0 else {}

    t1_xi_valid = [p for p in team1_xi if p and str(p).strip()]
    t2_xi_valid = [p for p in team2_xi if p and str(p).strip()]

    if not t1_xi_valid and not t2_xi_valid:
        return 0.0, {"note": "Empty XI lists"}

    # Classify each team's XI
    t1_stats = _classify_players(t1_xi_valid, bat_lookup, bowl_lookup) if t1_xi_valid else None
    t2_stats = _classify_players(t2_xi_valid, bat_lookup, bowl_lookup) if t2_xi_valid else None

    # If only one team's XI provided, can't compute relative advantage
    if t1_stats is None or t2_stats is None:
        return 0.0, {"note": "Only one team's XI provided — need both for adjustment"}

    # Check we found enough players to be meaningful
    t1_found = t1_stats["players_found"]
    t2_found = t2_stats["players_found"]
    min_found = min(t1_found, t2_found)

    if min_found < 3:
        note = f"Not enough players found in data (T1: {t1_found}, T2: {t2_found}). Check name spelling."
        return 0.0, {"note": note, "t1": t1_stats, "t2": t2_stats}

    # ── Batting advantage ─────────────────────────────────────────────────────
    # SR difference normalised to era baseline
    # +10 SR advantage ≈ meaningful edge
    bat_diff     = t1_stats["squad_avg_sr"] - t2_stats["squad_avg_sr"]
    bat_adv_raw  = bat_diff / ERA_AVG_BAT_SR   # fraction of era baseline

    # Powerplay + Death bonus (weight: 30% each, overall 40%)
    pp_diff      = (t1_stats["squad_pp_sr"]    - t2_stats["squad_pp_sr"])    / ERA_AVG_BAT_SR
    death_diff   = (t1_stats["squad_death_sr"] - t2_stats["squad_death_sr"]) / ERA_AVG_BAT_SR
    bat_adv      = 0.4 * bat_adv_raw + 0.3 * pp_diff + 0.3 * death_diff

    # ── Bowling advantage ─────────────────────────────────────────────────────
    # Lower economy = better = positive advantage for team1
    eco_diff     = t2_stats["squad_avg_eco"] - t1_stats["squad_avg_eco"]   # inverted
    eco_adv_raw  = eco_diff / ERA_AVG_BOWL_ECO

    pp_eco_diff  = (t2_stats["squad_pp_eco"]    - t1_stats["squad_pp_eco"])    / ERA_AVG_BOWL_ECO
    dth_eco_diff = (t2_stats["squad_death_eco"] - t1_stats["squad_death_eco"]) / ERA_AVG_BOWL_ECO
    bowl_adv     = 0.4 * eco_adv_raw + 0.3 * pp_eco_diff + 0.3 * dth_eco_diff

    # ── Allrounder depth bonus ────────────────────────────────────────────────
    ar_diff = t1_stats["allrounder_count"] - t2_stats["allrounder_count"]
    ar_adv  = ar_diff * 0.01   # 1% per allrounder advantage

    # ── Combine ───────────────────────────────────────────────────────────────
    raw_adjustment = (
        BATTING_WEIGHT * bat_adv
        + BOWLING_WEIGHT * bowl_adv
        + ar_adv
    )

    # Scale: raw_adjustment is already a small fraction. Apply sigmoid-like scaling
    # so large raw values don't blow past the cap, but small values are preserved.
    scaled = np.tanh(raw_adjustment * 3) * XI_MAX_ADJUSTMENT

    debug = {
        "t1_squad_avg_sr"   : round(t1_stats["squad_avg_sr"], 1),
        "t2_squad_avg_sr"   : round(t2_stats["squad_avg_sr"], 1),
        "t1_squad_avg_eco"  : round(t1_stats["squad_avg_eco"], 2),
        "t2_squad_avg_eco"  : round(t2_stats["squad_avg_eco"], 2),
        "t1_squad_pp_sr"    : round(t1_stats["squad_pp_sr"], 1),
        "t2_squad_pp_sr"    : round(t2_stats["squad_pp_sr"], 1),
        "t1_squad_death_sr" : round(t1_stats["squad_death_sr"], 1),
        "t2_squad_death_sr" : round(t2_stats["squad_death_sr"], 1),
        "t1_squad_pp_eco"   : round(t1_stats["squad_pp_eco"], 2),
        "t2_squad_pp_eco"   : round(t2_stats["squad_pp_eco"], 2),
        "t1_squad_death_eco": round(t1_stats["squad_death_eco"], 2),
        "t2_squad_death_eco": round(t2_stats["squad_death_eco"], 2),
        "t1_allrounders"    : t1_stats["allrounder_count"],
        "t2_allrounders"    : t2_stats["allrounder_count"],
        "bat_advantage"     : round(bat_adv, 4),
        "bowl_advantage"    : round(bowl_adv, 4),
        "raw_adjustment"    : round(raw_adjustment, 4),
        "final_adjustment"  : round(float(scaled), 4),
        "t1_players_found"  : t1_found,
        "t2_players_found"  : t2_found,
        "note"              : f"T1 found {t1_found}/{len(t1_xi_valid)}, T2 found {t2_found}/{len(t2_xi_valid)} players",
    }

    return float(scaled), debug


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY FOR DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def get_xi_feature_summary(
    team1_xi: List[str],
    team2_xi: List[str],
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    team1_name: str = "Team 1",
    team2_name: str = "Team 2",
) -> dict:
    """
    Returns a human-readable summary of XI stats for display in the dashboard.
    Used to show WHY the XI is changing the prediction.
    """
    bat_lookup  = bat_df.set_index("player").to_dict("index") if bat_df is not None and len(bat_df) > 0 else {}
    bowl_lookup = bowl_df.set_index("player").to_dict("index") if bowl_df is not None and len(bowl_df) > 0 else {}

    t1_xi_valid = [p for p in (team1_xi or []) if p and str(p).strip()]
    t2_xi_valid = [p for p in (team2_xi or []) if p and str(p).strip()]

    t1 = _classify_players(t1_xi_valid, bat_lookup, bowl_lookup) if t1_xi_valid else {}
    t2 = _classify_players(t2_xi_valid, bat_lookup, bowl_lookup) if t2_xi_valid else {}

    return {
        "team1_name": team1_name,
        "team2_name": team2_name,
        "team1": t1,
        "team2": t2,
        "metrics": [
            {
                "label"  : "Squad Avg Batting SR",
                "team1"  : round(t1.get("squad_avg_sr",    ERA_AVG_BAT_SR), 1)  if t1 else "—",
                "team2"  : round(t2.get("squad_avg_sr",    ERA_AVG_BAT_SR), 1)  if t2 else "—",
                "better" : "team1" if t1 and t2 and t1.get("squad_avg_sr", 0) > t2.get("squad_avg_sr", 0) else "team2",
                "higher_is_better": True,
            },
            {
                "label"  : "Squad Avg Economy",
                "team1"  : round(t1.get("squad_avg_eco",   ERA_AVG_BOWL_ECO), 2) if t1 else "—",
                "team2"  : round(t2.get("squad_avg_eco",   ERA_AVG_BOWL_ECO), 2) if t2 else "—",
                "better" : "team1" if t1 and t2 and t1.get("squad_avg_eco", 99) < t2.get("squad_avg_eco", 99) else "team2",
                "higher_is_better": False,
            },
            {
                "label"  : "Powerplay SR",
                "team1"  : round(t1.get("squad_pp_sr",  140.0), 1) if t1 else "—",
                "team2"  : round(t2.get("squad_pp_sr",  140.0), 1) if t2 else "—",
                "better" : "team1" if t1 and t2 and t1.get("squad_pp_sr", 0) > t2.get("squad_pp_sr", 0) else "team2",
                "higher_is_better": True,
            },
            {
                "label"  : "Death Overs SR",
                "team1"  : round(t1.get("squad_death_sr", 155.0), 1) if t1 else "—",
                "team2"  : round(t2.get("squad_death_sr", 155.0), 1) if t2 else "—",
                "better" : "team1" if t1 and t2 and t1.get("squad_death_sr", 0) > t2.get("squad_death_sr", 0) else "team2",
                "higher_is_better": True,
            },
            {
                "label"  : "Death Economy",
                "team1"  : round(t1.get("squad_death_eco", 11.2), 2) if t1 else "—",
                "team2"  : round(t2.get("squad_death_eco", 11.2), 2) if t2 else "—",
                "better" : "team1" if t1 and t2 and t1.get("squad_death_eco", 99) < t2.get("squad_death_eco", 99) else "team2",
                "higher_is_better": False,
            },
            {
                "label"  : "All-rounders",
                "team1"  : t1.get("allrounder_count", 0) if t1 else "—",
                "team2"  : t2.get("allrounder_count", 0) if t2 else "—",
                "better" : "team1" if t1 and t2 and t1.get("allrounder_count", 0) > t2.get("allrounder_count", 0) else "team2",
                "higher_is_better": True,
            },
        ],
    }