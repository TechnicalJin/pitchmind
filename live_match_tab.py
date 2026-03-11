"""
PITCHMIND — Live Match Tab
===========================
Add this to 4_dashboard.py as a new tab.

HOW TO INTEGRATE:
  1. Copy this file as  live_match_tab.py  next to 4_dashboard.py
  2. In 4_dashboard.py, add this import at the top:
         from live_match_tab import render_live_match_tab
  3. In the tabs section of 4_dashboard.py, add a new tab:
         tab_live, tab_predict, tab_stats, ... = st.tabs(["🔴 Live Match", ...])
         with tab_live:
             render_live_match_tab()

FEATURES:
  ✅ Ball-by-ball live score input (manual OR auto-fetch via ESPNcricinfo)
  ✅ Phase tracker (Powerplay / Middle / Death)
  ✅ Batsman strike rate display (live + career)
  ✅ Bowler economy display (live + career)
  ✅ Predicted phase total with confidence range
  ✅ Predicted final innings total
  ✅ Phase progress bar + run rate gauge
  ✅ Auto-refresh every 30 seconds (toggle)
"""

import os
import json
import time
import requests
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_DIR   = "data"

PHASE_RANGES = {
    "Powerplay": (0,  5),
    "Middle"   : (6,  14),
    "Death"    : (15, 19),
}

# Batting strength weights (same as training)
BATTING_STRENGTH_MAP = {
    0: 1.0, 1: 1.0, 2: 0.95, 3: 0.90, 4: 0.85,
    5: 0.80, 6: 0.75, 7: 0.65, 8: 0.50, 9: 0.35, 10: 0.25,
}

DEFAULT_BATTER_SR   = 120.0
DEFAULT_BOWLER_ECO  = 8.5
DEFAULT_VENUE_TOTAL = 160.0


# ══════════════════════════════════════════════════════════════════════════════
# LOAD PHASE MODELS & LOOKUPS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_phase_models():
    """Load all 3 phase models + lookup dicts. Returns (models_dict, lookups_dict)."""
    models  = {}
    lookups = {}

    for phase in ["powerplay", "middle", "death"]:
        path = os.path.join(MODELS_DIR, f"phase_{phase}_model.pkl")
        if os.path.exists(path):
            models[phase] = joblib.load(path)

    for key in ["batter_sr", "bowler_eco", "venue_factor"]:
        path = os.path.join(MODELS_DIR, f"phase_{key}.pkl")
        if os.path.exists(path):
            lookups[key] = joblib.load(path)

    return models, lookups


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINE CURRENT PHASE
# ══════════════════════════════════════════════════════════════════════════════
def get_phase(current_over):
    """Returns phase name and (start_over, end_over) tuple."""
    if current_over <= 5:
        return "Powerplay", (0, 5)
    elif current_over <= 14:
        return "Middle", (6, 14)
    else:
        return "Death", (15, 19)


def get_phase_key(phase_name):
    return {"Powerplay": "powerplay", "Middle": "middle", "Death": "death"}[phase_name]


# ══════════════════════════════════════════════════════════════════════════════
# GET BATTER / BOWLER STATS
# ══════════════════════════════════════════════════════════════════════════════
def get_batter_sr(name, phase_key, batter_sr_lookup):
    """Returns career SR for given batter in given phase."""
    info = batter_sr_lookup.get(name, {})
    return info.get(phase_key, info.get("overall", DEFAULT_BATTER_SR))


def get_bowler_eco(name, phase_key, bowler_eco_lookup):
    """Returns career economy for given bowler in given phase."""
    eco_key = {"powerplay": "pp", "middle": "overall", "death": "death"}[phase_key]
    info = bowler_eco_lookup.get(name, {})
    return info.get(eco_key, info.get("overall", DEFAULT_BOWLER_ECO))


def get_venue_factor(venue, phase_key, venue_factor_lookup):
    """Returns venue avg runs for given phase and total."""
    info = venue_factor_lookup.get(venue, {})
    phase_avg_key = f"avg_{'pp' if phase_key == 'powerplay' else phase_key}"
    return (
        info.get(phase_avg_key, 55.0),
        info.get("avg_total", DEFAULT_VENUE_TOTAL),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT PHASE
# ══════════════════════════════════════════════════════════════════════════════
def predict_phase_runs(
    phase_key, model,
    runs_sofar, wickets_in_phase, balls_completed,
    striker_sr, non_striker_sr, bowler_eco,
    wickets_in_hand, batting_strength, partnership_runs,
    venue_avg_phase, venue_avg_total,
):
    """
    Returns (remaining_pred, low, high, total_pred)
    total_pred = runs_sofar + remaining_pred
    """
    over_start, over_end = PHASE_RANGES[
        {"powerplay": "Powerplay", "middle": "Middle", "death": "Death"}[phase_key]
    ]
    total_phase_balls = (over_end - over_start + 1) * 6
    balls_remaining   = max(0, total_phase_balls - balls_completed)

    if balls_remaining == 0:
        return 0, 0, 0, runs_sofar

    phase_rr = runs_sofar / (balls_completed / 6) if balls_completed >= 6 else (
        runs_sofar * 6 / max(balls_completed, 1)
    )

    if model is None:
        # Fallback: simple run-rate projection
        remaining_pred = (balls_remaining / 6) * phase_rr
        remaining_pred = max(0, remaining_pred)
        margin = max(8, remaining_pred * 0.15)
        return round(remaining_pred), round(remaining_pred - margin), round(remaining_pred + margin), round(runs_sofar + remaining_pred)

    feat = pd.DataFrame([{
        "runs_sofar"           : runs_sofar,
        "wickets_in_phase"     : wickets_in_phase,
        "balls_completed"      : balls_completed,
        "balls_remaining"      : balls_remaining,
        "phase_run_rate"       : phase_rr,
        "striker_sr"           : striker_sr,
        "non_striker_sr"       : non_striker_sr,
        "partnership_runs"     : partnership_runs,
        "bowler_economy"       : bowler_eco,
        "wickets_in_hand"      : wickets_in_hand,
        "batting_strength"     : batting_strength,
        "venue_avg_phase_runs" : venue_avg_phase,
        "venue_avg_total"      : venue_avg_total,
    }])

    remaining_pred = max(0.0, float(model.predict(feat)[0]))
    margin = max(8, remaining_pred * 0.15)
    lo = max(0, remaining_pred - margin)
    hi = remaining_pred + margin

    return round(remaining_pred), round(lo), round(hi), round(runs_sofar + remaining_pred)


# ══════════════════════════════════════════════════════════════════════════════
# PROJECT FINAL SCORE (runs all 3 phases through models or historical averages)
# ══════════════════════════════════════════════════════════════════════════════
def project_final_score(
    current_over, current_ball, total_runs, total_wickets,
    phase_key, models, batter_sr_lookup, bowler_eco_lookup,
    striker, non_striker, bowler, venue,
    venue_factor_lookup,
    pp_runs=None, mid_runs=None, death_runs=None,
):
    """
    Projects total innings score from current state.
    - Completed phases: use actual runs
    - Current phase: use model prediction
    - Future phases: use venue historical averages
    """
    vf   = venue_factor_lookup.get(venue, {})
    v_pp = vf.get("avg_pp", 50.0)
    v_mid= vf.get("avg_mid", 60.0)
    v_dth= vf.get("avg_death", 50.0)
    v_tot= vf.get("avg_total", DEFAULT_VENUE_TOTAL)

    # --- Powerplay ---
    if current_over < 6:
        # Currently in powerplay
        balls_in_pp    = current_over * 6 + current_ball
        pp_so_far      = total_runs  # rough: all runs are PP runs
        wkts_in_pp     = total_wickets
        wkts_in_hand   = 10 - total_wickets
        bat_str        = sum(BATTING_STRENGTH_MAP.get(i, 0.2) for i in range(total_wickets, 10))
        s_sr = get_batter_sr(striker,     "pp", batter_sr_lookup)
        ns_sr= get_batter_sr(non_striker, "pp", batter_sr_lookup)
        b_eco= get_bowler_eco(bowler,     "powerplay", bowler_eco_lookup)

        _, _, _, pp_pred = predict_phase_runs(
            "powerplay", models.get("powerplay"),
            pp_so_far, wkts_in_pp, balls_in_pp,
            s_sr, ns_sr, b_eco, wkts_in_hand, bat_str, pp_so_far,
            v_pp, v_tot,
        )
        mid_pred   = v_mid   * (wkts_in_hand / 10)
        death_pred = v_dth   * (wkts_in_hand / 10)
        final      = pp_pred + mid_pred + death_pred

    elif current_over < 15:
        # Currently in middle overs
        pp_actual   = pp_runs if pp_runs else v_pp
        balls_in_mid= (current_over - 6) * 6 + current_ball
        mid_so_far  = total_runs - pp_actual
        wkts_in_hand= 10 - total_wickets
        bat_str     = sum(BATTING_STRENGTH_MAP.get(i, 0.2) for i in range(total_wickets, 10))
        s_sr = get_batter_sr(striker,     "mid", batter_sr_lookup)
        ns_sr= get_batter_sr(non_striker, "mid", batter_sr_lookup)
        b_eco= get_bowler_eco(bowler,     "middle", bowler_eco_lookup)

        _, _, _, mid_pred = predict_phase_runs(
            "middle", models.get("middle"),
            max(0, mid_so_far), 0, balls_in_mid,
            s_sr, ns_sr, b_eco, wkts_in_hand, bat_str, max(0, mid_so_far),
            v_mid, v_tot,
        )
        death_pred = v_dth * (wkts_in_hand / 10)
        final      = pp_actual + mid_pred + death_pred

    else:
        # Currently in death overs
        pp_actual   = pp_runs   if pp_runs   else v_pp
        mid_actual  = mid_runs  if mid_runs  else v_mid
        balls_in_dth= (current_over - 15) * 6 + current_ball
        death_sofar = total_runs - pp_actual - mid_actual
        wkts_in_hand= 10 - total_wickets
        bat_str     = sum(BATTING_STRENGTH_MAP.get(i, 0.2) for i in range(total_wickets, 10))
        s_sr = get_batter_sr(striker,     "death", batter_sr_lookup)
        ns_sr= get_batter_sr(non_striker, "death", batter_sr_lookup)
        b_eco= get_bowler_eco(bowler,     "death", bowler_eco_lookup)

        _, _, _, death_pred = predict_phase_runs(
            "death", models.get("death"),
            max(0, death_sofar), 0, balls_in_dth,
            s_sr, ns_sr, b_eco, wkts_in_hand, bat_str, max(0, death_sofar),
            v_dth, v_tot,
        )
        final = pp_actual + mid_actual + death_pred

    return round(final)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE SCORE FETCH (ESPNcricinfo — extends your 5_live_data_fetch.py)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_live_innings(match_id):
    """
    Fetches current live innings state from ESPNcricinfo.
    Returns dict with score/wickets/over/striker/non_striker/bowler or None.
    """
    if not match_id:
        return None

    url = f"https://www.espncricinfo.com/matches/engine/match/{match_id}.json"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()

        innings = data.get("innings", [])
        if not innings:
            return None

        # Latest active innings
        active_inn = None
        for inn in reversed(innings):
            if inn.get("live_current_name") or inn.get("runs") is not None:
                active_inn = inn
                break

        if not active_inn:
            active_inn = innings[-1]

        runs     = int(active_inn.get("runs", 0) or 0)
        wickets  = int(active_inn.get("wickets", 0) or 0)
        overs    = active_inn.get("overs", "0.0")

        try:
            parts    = str(overs).split(".")
            over_no  = int(parts[0])
            ball_no  = int(parts[1]) if len(parts) > 1 else 0
        except Exception:
            over_no, ball_no = 0, 0

        # Bat/bowl info
        bat_info  = active_inn.get("bat", [])
        bowl_info = active_inn.get("bowl", [])

        striker = non_striker = bowler = ""
        for b in bat_info:
            if b.get("live_current_name") == "striker":
                striker = b.get("known_as", b.get("name_full", ""))
            elif b.get("live_current_name") == "non-striker":
                non_striker = b.get("known_as", b.get("name_full", ""))
        for bw in bowl_info:
            if bw.get("live_current_name") == "current":
                bowler = bw.get("known_as", bw.get("name_full", ""))

        return {
            "runs"       : runs,
            "wickets"    : wickets,
            "over"       : over_no,
            "ball"       : ball_no,
            "striker"    : striker,
            "non_striker": non_striker,
            "bowler"     : bowler,
            "fetched_at" : datetime.now().strftime("%H:%M:%S"),
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# RENDER — MAIN TAB FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def render_live_match_tab():
    """
    Call this inside your Streamlit tab:
        with tab_live:
            render_live_match_tab()
    """
    st.markdown("## 🔴 Live In-Match Run Predictor")
    st.caption("Track live score → predict phase totals & final score using ML + batsman SR + bowler economy")

    # Load models
    models, lookups = load_phase_models()
    batter_sr_lookup    = lookups.get("batter_sr", {})
    bowler_eco_lookup   = lookups.get("bowler_eco", {})
    venue_factor_lookup = lookups.get("venue_factor", {})

    models_loaded = bool(models)
    if not models_loaded:
        st.warning(
            "⚠️ Phase models not found. Run `python 7_phase_predictor.py` first.\n"
            "Falling back to run-rate projection."
        )

    # ── SIDEBAR / INPUT PANEL ─────────────────────────────────────────────────
    st.markdown("---")
    col_input, col_live = st.columns([1, 2])

    with col_input:
        st.markdown("### ⚙️ Match Setup")

        batting_team  = st.text_input("Batting Team",  value="India",       key="lt_bat_team")
        bowling_team  = st.text_input("Bowling Team",  value="Australia",   key="lt_bowl_team")
        venue         = st.text_input("Venue",         value="Wankhede Stadium", key="lt_venue")

        st.markdown("---")
        st.markdown("### 🔄 Data Source")
        data_mode = st.radio(
            "Input mode",
            ["Manual Entry", "Auto-Fetch (ESPNcricinfo)"],
            key="lt_data_mode",
        )

        if data_mode == "Auto-Fetch (ESPNcricinfo)":
            match_id_input = st.text_input(
                "ESPNcricinfo Match ID",
                value="",
                help="Found in match URL, e.g. espncricinfo.com/series/xxx/match/1512773/",
                key="lt_match_id",
            )
            auto_refresh = st.toggle("Auto-refresh every 30s", value=False, key="lt_auto_refresh")
        else:
            match_id_input = ""
            auto_refresh   = False

        st.markdown("---")
        st.markdown("### 📊 Current Score")

        total_runs    = st.number_input("Total Runs",    min_value=0, max_value=400, value=30,  step=1, key="lt_runs")
        total_wickets = st.number_input("Wickets Down",  min_value=0, max_value=10,  value=0,   step=1, key="lt_wkts")
        current_over  = st.number_input("Current Over",  min_value=0, max_value=19,  value=2,   step=1, key="lt_over",
                                         help="0 = before over 1, 5 = over 6 in progress")
        current_ball  = st.number_input("Ball in Over",  min_value=0, max_value=6,   value=3,   step=1, key="lt_ball")

        st.markdown("---")
        st.markdown("### 🏏 At Crease")

        striker     = st.text_input("Striker",     value="V Kohli",      key="lt_striker")
        non_striker = st.text_input("Non-Striker", value="R Sharma",     key="lt_non_striker")
        bowler      = st.text_input("Current Bowler", value="",          key="lt_bowler")

        st.markdown("---")
        st.markdown("### 📈 Phase Runs (fill after phase ends)")
        pp_runs    = st.number_input("Powerplay Runs  (1–6)",  min_value=0, max_value=150, value=0, key="lt_pp")
        mid_runs   = st.number_input("Middle Runs    (7–15)",  min_value=0, max_value=150, value=0, key="lt_mid")
        death_runs = st.number_input("Death Runs    (16–20)",  min_value=0, max_value=150, value=0, key="lt_death")

    # ── AUTO-FETCH ────────────────────────────────────────────────────────────
    with col_live:
        if data_mode == "Auto-Fetch (ESPNcricinfo)" and match_id_input:
            if st.button("🔄 Fetch Now", key="lt_fetch_btn") or auto_refresh:
                with st.spinner("Fetching live data..."):
                    live = fetch_live_innings(match_id_input)
                if live and "error" not in live:
                    st.success(f"✅ Fetched at {live['fetched_at']}")
                    # Update state from live data
                    total_runs    = live["runs"]
                    total_wickets = live["wickets"]
                    current_over  = live["over"]
                    current_ball  = live["ball"]
                    striker       = live["striker"]    or striker
                    non_striker   = live["non_striker"] or non_striker
                    bowler        = live["bowler"]      or bowler
                elif live and "error" in live:
                    st.error(f"❌ Fetch failed: {live['error']}")

        # ── COMPUTE EVERYTHING ────────────────────────────────────────────────
        phase_name, (p_start, p_end) = get_phase(current_over), (
            (0,5) if current_over <= 5 else (6,14) if current_over <= 14 else (15,19)
        )
        phase_name = ("Powerplay" if current_over <= 5 else
                      "Middle"    if current_over <= 14 else "Death")
        phase_key  = get_phase_key(phase_name)

        total_phase_balls = (p_end - p_start + 1) * 6
        balls_in_phase    = (current_over - p_start) * 6 + current_ball
        balls_remaining_phase = max(0, total_phase_balls - balls_in_phase)

        # Runs in current phase (approximated from total)
        if phase_name == "Powerplay":
            phase_runs_sofar = total_runs
            wickets_in_phase = total_wickets
        elif phase_name == "Middle":
            phase_runs_sofar = max(0, total_runs - (pp_runs or 0))
            wickets_in_phase = max(0, total_wickets - 0)  # approximate
        else:
            phase_runs_sofar = max(0, total_runs - (pp_runs or 0) - (mid_runs or 0))
            wickets_in_phase = max(0, total_wickets - 0)

        wickets_in_hand = 10 - total_wickets
        batting_strength = sum(
            BATTING_STRENGTH_MAP.get(i, 0.2) for i in range(total_wickets, 10)
        )

        # Lookups
        s_sr  = get_batter_sr(striker,     phase_key, batter_sr_lookup)
        ns_sr = get_batter_sr(non_striker, phase_key, batter_sr_lookup)
        b_eco = get_bowler_eco(bowler,     phase_key, bowler_eco_lookup)
        v_avg_phase, v_avg_total = get_venue_factor(venue, phase_key, venue_factor_lookup)

        # Phase prediction
        rem_pred, rem_lo, rem_hi, phase_total_pred = predict_phase_runs(
            phase_key, models.get(phase_key),
            phase_runs_sofar, wickets_in_phase, balls_in_phase,
            s_sr, ns_sr, b_eco,
            wickets_in_hand, batting_strength, phase_runs_sofar,
            v_avg_phase, v_avg_total,
        )

        # Final score projection
        final_pred = project_final_score(
            current_over, current_ball, total_runs, total_wickets,
            phase_key, models, batter_sr_lookup, bowler_eco_lookup,
            striker, non_striker, bowler, venue, venue_factor_lookup,
            pp_runs=pp_runs or None,
            mid_runs=mid_runs or None,
            death_runs=death_runs or None,
        )

        current_rr = (total_runs / ((current_over * 6 + current_ball) / 6)
                      if current_over * 6 + current_ball >= 6 else 0.0)

        # ── DISPLAY: TOP METRICS ──────────────────────────────────────────────
        st.markdown(f"### 📍 Phase: **{phase_name}**  |  Over {current_over}.{current_ball}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score",         f"{total_runs}/{total_wickets}")
        m2.metric("Run Rate",      f"{current_rr:.2f}")
        m3.metric("Balls Left (Phase)", f"{balls_remaining_phase}")
        m4.metric("Wickets in Hand",    f"{wickets_in_hand}")

        st.markdown("---")

        # ── DISPLAY: BATSMEN & BOWLER ─────────────────────────────────────────
        st.markdown("#### 🏏 At The Crease")
        b1, b2, b3 = st.columns(3)

        with b1:
            st.markdown(f"**🟢 Striker**")
            st.markdown(f"`{striker or 'Unknown'}`")
            st.metric("Career SR (this phase)", f"{s_sr:.1f}")

        with b2:
            st.markdown(f"**🔵 Non-Striker**")
            st.markdown(f"`{non_striker or 'Unknown'}`")
            st.metric("Career SR (this phase)", f"{ns_sr:.1f}")

        with b3:
            st.markdown(f"**🔴 Bowler**")
            st.markdown(f"`{bowler or 'Unknown'}`")
            st.metric("Career Economy (this phase)", f"{b_eco:.1f}")

        st.markdown("---")

        # ── DISPLAY: PHASE PREDICTION ─────────────────────────────────────────
        st.markdown(f"#### 📊 {phase_name} Prediction")

        p1, p2, p3 = st.columns(3)
        p1.metric(f"Runs so far ({phase_name})", f"{phase_runs_sofar}")
        p2.metric("Predicted Remaining", f"{rem_pred}",
                  help=f"Range: {rem_lo}–{rem_hi} runs")
        p3.metric(f"Predicted {phase_name} Total",
                  f"{phase_total_pred}",
                  delta=f"Range: {phase_runs_sofar + rem_lo}–{phase_runs_sofar + rem_hi}")

        # Progress bar
        phase_pct = min(1.0, balls_in_phase / total_phase_balls)
        st.markdown(f"**Phase Progress:** {balls_in_phase}/{total_phase_balls} balls")
        st.progress(phase_pct)

        # ── DISPLAY: FINAL SCORE PROJECTION ──────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🎯 Final Score Projection")

        f1, f2, f3 = st.columns(3)
        margin = max(12, final_pred * 0.08)
        f1.metric("Current Score",       f"{total_runs}/{total_wickets}")
        f2.metric("Projected Final",     f"{final_pred}",
                  delta=f"±{round(margin)} runs")
        f3.metric("Venue Avg (1st inn)", f"{round(v_avg_total)}")

        # Visual bar chart: phase breakdown
        st.markdown("---")
        st.markdown("#### 📉 Projected Score Breakdown")

        # Build phase bars
        if phase_name == "Powerplay":
            pp_bar    = phase_total_pred
            mid_bar   = round((final_pred - pp_bar) * 0.55)
            death_bar = round(final_pred - pp_bar - mid_bar)
        elif phase_name == "Middle":
            pp_bar    = pp_runs or round(v_avg_total * 0.30)
            mid_bar   = phase_total_pred - pp_bar
            death_bar = round(final_pred - pp_bar - mid_bar)
        else:
            pp_bar    = pp_runs   or round(v_avg_total * 0.30)
            mid_bar   = mid_runs  or round(v_avg_total * 0.37)
            death_bar = phase_total_pred - pp_bar - mid_bar

        fig, ax = plt.subplots(figsize=(8, 2.5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")

        phases    = ["Powerplay\n(1–6)", "Middle\n(7–15)", "Death\n(16–20)"]
        vals      = [max(0, pp_bar), max(0, mid_bar), max(0, death_bar)]
        colors    = ["#1976d2", "#ffa726", "#e53935"]
        completed = [phase_name in ["Middle", "Death"], phase_name == "Death", False]
        bar_colors = [
            c if not done else "#555"
            for c, done in zip(colors, [False, False, False])  # always color all
        ]
        actual_flag = [
            "✓ Actual" if (phase_name == "Middle" and i == 0) or
                           (phase_name == "Death"   and i <= 1)
                       else "Projected"
            for i in range(3)
        ]

        bars = ax.barh(phases, vals, color=bar_colors, height=0.5)
        for bar, val, flag in zip(bars, vals, actual_flag):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val}  ({flag})",
                    va="center", color="white", fontsize=9)

        ax.set_xlim(0, max(vals) * 1.35)
        ax.set_xlabel("Runs", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.set_title(
            f"Projected: {max(0,pp_bar)+max(0,mid_bar)+max(0,death_bar)} total",
            color="white", fontsize=11
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── AUTO REFRESH ──────────────────────────────────────────────────────
        if auto_refresh and data_mode == "Auto-Fetch (ESPNcricinfo)":
            st.caption(f"⏱ Auto-refreshing every 30 seconds...")
            time.sleep(30)
            st.rerun()

        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}  "
                   f"| Models: {'✅ Loaded' if models_loaded else '⚠️ Fallback mode'}")


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Run as: streamlit run 4_dashboard.py")
    print("This module should be imported, not run directly.")
    print("\nQuick test — predict_phase_runs (no model, fallback):")
    rem, lo, hi, total = predict_phase_runs(
        "powerplay", None,
        runs_sofar=30, wickets_in_phase=0, balls_completed=18,
        striker_sr=145, non_striker_sr=130, bowler_eco=8.2,
        wickets_in_hand=10, batting_strength=9.5, partnership_runs=30,
        venue_avg_phase=52, venue_avg_total=175,
    )
    print(f"  After 3 overs, 30 runs, 0 wkts:")
    print(f"  Remaining: {rem}  |  Range: {lo}–{hi}  |  PP Total: {total}")