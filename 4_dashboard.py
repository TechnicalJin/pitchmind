"""
PITCHMIND — IPL PREDICTION DASHBOARD
=====================================
Step 4: Streamlit Interactive Dashboard

Run from pitchmind/ directory:
  streamlit run 4_dashboard.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import json as _json
import datetime as _dt

# ── PLAYER FEATURES MODULE ────────────────────────────────────────────────────
try:
    from pitchmind_player_features import (
        load_deliveries as _load_deliveries,
        load_matches    as _load_matches,
        compute_batting_stats,
        compute_bowling_stats,
        compute_h2h,
        search_players,
    )
    PLAYER_MODULE_OK = True
except ImportError:
    # Try loading from same directory as 6_player_features.py
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from six_player_features import (
            load_deliveries as _load_deliveries,
            load_matches    as _load_matches,
            compute_batting_stats,
            compute_bowling_stats,
            compute_h2h,
            search_players,
        )
        PLAYER_MODULE_OK = True
    except ImportError:
        PLAYER_MODULE_OK = False

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchMind — IPL Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .win-box-blue {
        background: linear-gradient(135deg, #0d47a1, #1976d2);
        padding: 28px; border-radius: 14px;
        text-align: center; color: white;
    }
    .win-box-red {
        background: linear-gradient(135deg, #b71c1c, #e53935);
        padding: 28px; border-radius: 14px;
        text-align: center; color: white;
    }
    .section-header {
        font-size: 1.25rem; font-weight: bold; color: #ffa726;
        border-bottom: 2px solid #ffa726;
        padding-bottom: 5px; margin: 18px 0 10px 0;
    }
    .stat-card {
        background: #1e2130; border-radius: 10px;
        padding: 14px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── DATA CLEANING HELPER ──────────────────────────────────────────────────────
def clean_dataframe(df):
    """Replace dash placeholders with None and coerce numeric columns."""
    df = df.replace(["\u2014", "\u2013", "\u2012", "\u2010", "\u2015",
                     "-", "\u2212", ""], None)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


# ── LOAD DATA & MODELS ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join("data", "master_features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_resource
def load_models():
    xgb_path  = os.path.join("models", "ipl_model.pkl")
    rf_path   = os.path.join("models", "ipl_model_rf.pkl")
    feat_path = os.path.join("models", "feature_cols.pkl")
    if not os.path.exists(xgb_path) or not os.path.exists(feat_path):
        return None, None, None
    xgb_model    = joblib.load(xgb_path)
    rf_model     = joblib.load(rf_path) if os.path.exists(rf_path) else None
    feature_cols = joblib.load(feat_path)
    return xgb_model, rf_model, feature_cols


df                                 = load_data()
xgb_model, rf_model, feature_cols = load_models()

@st.cache_data(show_spinner=False)
def load_player_data():
    """Load deliveries + batting/bowling stats (cached)."""
    if not PLAYER_MODULE_OK:
        return None, None, None
    del_df     = _load_deliveries()
    matches_df = _load_matches()
    if del_df is None:
        return None, None, None
    bat_df  = compute_batting_stats(del_df, matches_df)
    bowl_df = compute_bowling_stats(del_df)
    return del_df, bat_df, bowl_df


del_df, bat_df, bowl_df = load_player_data()


# ── HELPER: get team stats from dataset ───────────────────────────────────────
def get_team_stats(team, role, venue, df):
    default = {
        "win_rate": 0.5, "recent_form": 0.5, "nrr": 0.0,
        "avg_runs": 155.0, "powerplay_runs": 45.0, "middle_runs": 55.0,
        "death_runs": 40.0, "boundary_pct": 0.16, "dot_ball_pct": 0.35,
        "run_rate": 8.0, "top3_sr": 117.0,
        "bowling_economy": 8.5, "death_economy": 9.5,
        "pp_wickets": 1.5, "bowling_sr": 20.0,
        "venue_win_rate": 0.5,
    }
    if df is None:
        return default

    mask = df[role] == team
    sub  = df[mask]
    if len(sub) == 0:
        return default

    r = sub.iloc[-1]

    venue_sub = sub[sub["venue"] == venue]
    vwr = venue_sub[f"{role}_venue_win_rate"].mean() if len(venue_sub) > 0 \
          else sub[f"{role}_venue_win_rate"].mean()

    return {
        "win_rate":        r.get(f"{role}_win_rate",        0.5),
        "recent_form":     r.get(f"{role}_recent_form",     0.5),
        "nrr":             r.get(f"{role}_nrr",             0.0),
        "avg_runs":        r.get(f"{role}_avg_runs",        155.0),
        "powerplay_runs":  r.get(f"{role}_powerplay_runs",  45.0),
        "middle_runs":     r.get(f"{role}_middle_runs",     55.0),
        "death_runs":      r.get(f"{role}_death_runs",      40.0),
        "boundary_pct":    r.get(f"{role}_boundary_pct",    0.16),
        "dot_ball_pct":    r.get(f"{role}_dot_ball_pct",    0.35),
        "run_rate":        r.get(f"{role}_run_rate",        8.0),
        "top3_sr":         r.get(f"{role}_top3_sr",         117.0),
        "bowling_economy": r.get(f"{role}_bowling_economy", 8.5),
        "death_economy":   r.get(f"{role}_death_economy",   9.5),
        "pp_wickets":      r.get(f"{role}_pp_wickets",      1.5),
        "bowling_sr":      r.get(f"{role}_bowling_sr",      20.0),
        "venue_win_rate":  float(vwr) if not np.isnan(float(vwr)) else 0.5,
    }


# ── HELPER: build feature vector matching trained model ───────────────────────
def build_feature_vector(team1, team2, venue, toss_winner, toss_decision, df, feature_cols):
    s1 = get_team_stats(team1, "team1", venue, df)
    s2 = get_team_stats(team2, "team2", venue, df)

    toss_win         = 1 if toss_winner == team1 else 0
    toss_field       = 1 if toss_decision == "field" else 0
    toss_team1_field = 1 if (toss_winner == team1 and toss_decision == "field") else 0

    h2h = 0.5
    if df is not None:
        h2h_mask = ((df["team1"] == team1) & (df["team2"] == team2)) | \
                   ((df["team1"] == team2) & (df["team2"] == team1))
        h2h_data = df[h2h_mask]
        if len(h2h_data) > 0:
            t1_wins = (
                ((h2h_data["team1"] == team1) & (h2h_data["target"] == 1)).sum() +
                ((h2h_data["team2"] == team1) & (h2h_data["target"] == 0)).sum()
            )
            h2h = t1_wins / len(h2h_data)

    venue_avg = 160.0
    if df is not None:
        v = df[df["venue"] == venue]["venue_avg_runs"].mean()
        if not np.isnan(v):
            venue_avg = v

    feat = {
        "team1_win_rate":        s1["win_rate"],
        "team2_win_rate":        s2["win_rate"],
        "team1_recent_form":     s1["recent_form"],
        "team2_recent_form":     s2["recent_form"],
        "h2h_win_rate":          h2h,
        "team1_nrr":             s1["nrr"],
        "team2_nrr":             s2["nrr"],
        "team1_avg_runs":        s1["avg_runs"],
        "team2_avg_runs":        s2["avg_runs"],
        "team1_powerplay_runs":  s1["powerplay_runs"],
        "team2_powerplay_runs":  s2["powerplay_runs"],
        "team1_middle_runs":     s1["middle_runs"],
        "team2_middle_runs":     s2["middle_runs"],
        "team1_death_runs":      s1["death_runs"],
        "team2_death_runs":      s2["death_runs"],
        "team1_boundary_pct":    s1["boundary_pct"],
        "team2_boundary_pct":    s2["boundary_pct"],
        "team1_dot_ball_pct":    s1["dot_ball_pct"],
        "team2_dot_ball_pct":    s2["dot_ball_pct"],
        "team1_run_rate":        s1["run_rate"],
        "team2_run_rate":        s2["run_rate"],
        "team1_top3_sr":         s1["top3_sr"],
        "team2_top3_sr":         s2["top3_sr"],
        "team1_bowling_economy": s1["bowling_economy"],
        "team2_bowling_economy": s2["bowling_economy"],
        "team1_death_economy":   s1["death_economy"],
        "team2_death_economy":   s2["death_economy"],
        "team1_pp_wickets":      s1["pp_wickets"],
        "team2_pp_wickets":      s2["pp_wickets"],
        "team1_bowling_sr":      s1["bowling_sr"],
        "team2_bowling_sr":      s2["bowling_sr"],
        "team1_venue_win_rate":  s1["venue_win_rate"],
        "team2_venue_win_rate":  s2["venue_win_rate"],
        "venue_avg_runs":        venue_avg,
        "toss_win":              toss_win,
        "toss_field":            toss_field,
        "toss_team1_field":      toss_team1_field,
        "diff_win_rate":         s1["win_rate"]        - s2["win_rate"],
        "diff_recent_form":      s1["recent_form"]     - s2["recent_form"],
        "diff_avg_runs":         s1["avg_runs"]        - s2["avg_runs"],
        "diff_death_runs":       s1["death_runs"]      - s2["death_runs"],
        "diff_death_economy":    s2["death_economy"]   - s1["death_economy"],
        "diff_bowling_economy":  s2["bowling_economy"] - s1["bowling_economy"],
        "diff_pp_wickets":       s1["pp_wickets"]      - s2["pp_wickets"],
        "diff_run_rate":         s1["run_rate"]        - s2["run_rate"],
        "diff_nrr":              s1["nrr"]             - s2["nrr"],
        "diff_venue_win_rate":   s1["venue_win_rate"]  - s2["venue_win_rate"],
    }

    return pd.DataFrame([feat])[feature_cols], s1, s2, h2h, venue_avg


# ── PREDICT ───────────────────────────────────────────────────────────────────
def predict_winner(X_input):
    if xgb_model is None:
        return 0.5, 0.5

    xgb_prob = xgb_model.predict_proba(X_input)[0][1]

    if rf_model is not None:
        rf_prob = rf_model.predict_proba(X_input)[0][1]
        prob_t1 = (xgb_prob + rf_prob) / 2
    else:
        prob_t1 = xgb_prob

    return round(prob_t1, 4), round(1 - prob_t1, 4)


# ── LIVE DATA HELPERS ─────────────────────────────────────────────────────────
def load_live_match():
    path = os.path.join("data", "live", "todays_match.json")
    if os.path.exists(path):
        with open(path) as f:
            return _json.load(f)
    return None


def save_live_match(data):
    save_path = os.path.join("data", "live", "todays_match.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        _json.dump(data, f, indent=2)


# ── LIVE DATA PANEL ───────────────────────────────────────────────────────────
def render_live_panel(team1_name, team2_name):
    st.markdown("---")
    st.markdown("## 🔴 Live Match Data")

    live = load_live_match()

    # ── Status Banner ─────────────────────────────────────────────────────────
    if live:
        has_xi    = bool(live.get("team1_xi") and live.get("team2_xi"))
        has_toss  = bool(live.get("toss_winner") and live.get("toss_decision"))
        # FIXED: define match_name as a plain variable before using it in string
        match_name = live.get("match_name") or "Today's Match"

        if has_toss and has_xi:
            st.success("🟢 Live data loaded — " + match_name)
        elif has_toss:
            st.warning("🟡 Toss data loaded — Playing XI missing")
        else:
            st.error("🔴 Live JSON found but missing key fields — use manual entry below")
    else:
        st.error("🔴 No live data file found. Run `python 5_live_data_fetch.py` or use manual entry below.")
        live = {}

    # ── Auto-fetched Data Display ─────────────────────────────────────────────
    if live:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📍 Match Info**")
            st.write("**Match:** "  + str(live.get("match_name", "—")))
            st.write("**Venue:** "  + str(live.get("venue", "—")))
            st.write("**Date:** "   + str(live.get("match_date", "—")))
            st.write("**Series:** " + str(live.get("series", "—")))

        with col2:
            st.markdown("**🪙 Toss**")
            st.write("**Won Toss:** "     + str(live.get("toss_winner", "—")))
            st.write("**Decision:** "     + str(live.get("toss_decision", "—")))
            st.write("**Pitch Type:** "   + str(live.get("pitch_type") or "Unknown"))
            st.write("**Dew Expected:** " + str(live.get("dew_expected") or "Unknown"))

        with col3:
            st.markdown("**🌤️ Weather**")
            wx = live.get("weather", {})
            if wx:
                st.write("**Temp:** "        + str(wx.get("temperature_c", "—")) + "°C")
                st.write("**Humidity:** "    + str(wx.get("humidity_pct", "—")) + "%")
                st.write("**Dew Point:** ~"  + str(wx.get("dew_point_approx", "—")) + "°C")
                st.write("**Conditions:** "  + str(wx.get("description", "—")).title())
            else:
                st.write("Weather data not available")

        # Playing XIs
        if live.get("team1_xi") or live.get("team2_xi"):
            st.markdown("**🏏 Playing XIs**")
            xi_col1, xi_col2 = st.columns(2)
            with xi_col1:
                st.markdown("**" + team1_name + "**")
                for i, p in enumerate(live.get("team1_xi", []), 1):
                    st.write(str(i) + ". " + str(p))
            with xi_col2:
                st.markdown("**" + team2_name + "**")
                for i, p in enumerate(live.get("team2_xi", []), 1):
                    st.write(str(i) + ". " + str(p))

        if live.get("expert_notes"):
            st.markdown("**📋 Expert Notes / Pitch Report**")
            st.info(live["expert_notes"])

    st.markdown("---")

    # ── MANUAL ENTRY SECTION ─────────────────────────────────────────────────
    st.markdown("### 📝 Manual Entry (use if auto-fetch is incomplete or wrong)")

    with st.expander("📋 Paste Commentary / Expert Pitch Report / Playing XI"):
        st.markdown(
            "Copy from Cricbuzz, ESPNcricinfo, or TV broadcast and paste here. "
            "This will show alongside the model prediction so you can verify context."
        )
        placeholder_text = (
            "Example:\n"
            + team1_name + " won the toss and elected to bat first.\n"
            + team1_name + " Playing XI: Player 1, Player 2, ...\n"
            + team2_name + " Playing XI: Player 1, Player 2, ...\n"
            + "Pitch Report: Hard pitch, good pace and carry."
        )
        manual_text = st.text_area(
            "Paste any text here:",
            height=200,
            key="manual_raw_text",
            placeholder=placeholder_text,
        )

        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("💾 Save Commentary"):
                if manual_text.strip():
                    live_data = load_live_match() or {}
                    live_data["expert_notes"]      = manual_text
                    live_data["manual_updated_at"] = str(_dt.datetime.now())
                    save_live_match(live_data)
                    st.success("✅ Saved. Refresh page to see update.")
                else:
                    st.warning("Nothing to save — paste text first.")

    with st.expander("⚙️ Manual Field Overrides"):
        st.markdown("Override individual fields if auto-fetch got them wrong:")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            m_toss_winner = st.selectbox(
                "Toss Winner",
                ["(auto)", team1_name, team2_name],
                key="m_toss_winner"
            )
            m_toss_decision = st.selectbox(
                "Toss Decision",
                ["(auto)", "bat", "field"],
                key="m_toss_decision"
            )
            m_pitch = st.selectbox(
                "Pitch Type",
                ["(auto)", "Good/Hard", "Dry/Spin", "Seamer-friendly", "Flat/High-scoring", "Unknown"],
                key="m_pitch"
            )
        with m_col2:
            m_dew = st.selectbox(
                "Dew Expected?",
                ["(auto)", "Yes — heavy", "Yes — light", "No", "Unknown"],
                key="m_dew"
            )
            m_xi1 = st.text_area(
                team1_name + " Playing XI (one per line)",
                height=130,
                key="m_xi1"
            )
            m_xi2 = st.text_area(
                team2_name + " Playing XI (one per line)",
                height=130,
                key="m_xi2"
            )

        if st.button("💾 Save Field Overrides"):
            live_data = load_live_match() or {}
            if m_toss_winner   != "(auto)": live_data["toss_winner"]   = m_toss_winner
            if m_toss_decision != "(auto)": live_data["toss_decision"] = m_toss_decision
            if m_pitch         != "(auto)": live_data["pitch_type"]    = m_pitch
            if m_dew           != "(auto)": live_data["dew_expected"]  = m_dew
            if m_xi1.strip():
                live_data["team1_xi"] = [p.strip() for p in m_xi1.strip().splitlines() if p.strip()]
            if m_xi2.strip():
                live_data["team2_xi"] = [p.strip() for p in m_xi2.strip().splitlines() if p.strip()]
            save_live_match(live_data)
            st.success("✅ Overrides saved! Refresh page to see update.")

    st.markdown("---")




# ══════════════════════════════════════════════════════════════════════════════
# PLAYER SCOUT RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def _fmt(val, suffix="", na="—"):
    """Format a numeric stat safely."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return f"{val}{suffix}"

def _color_metric(val, good_high=True, thresholds=(40, 70)):
    """Return a color based on value and whether higher is better."""
    if val is None:
        return "#888"
    lo, hi = thresholds
    if good_high:
        return "#4caf50" if val >= hi else ("#ffa726" if val >= lo else "#ef5350")
    else:
        return "#4caf50" if val <= lo else ("#ffa726" if val <= hi else "#ef5350")

def _player_batting_card(player, stats):
    """Render a compact batting stat card for one player."""
    if not stats:
        st.markdown(
            f"<div style='background:#1e2130;border-radius:8px;padding:10px;"
            f"margin:4px 0;color:#888;font-size:0.85rem'>"
            f"🏏 <b>{player}</b> — no batting data found</div>",
            unsafe_allow_html=True
        )
        return

    sr   = stats.get("strike_rate", 0)
    avg  = stats.get("batting_avg", 0)
    runs = stats.get("runs", 0)
    inn  = stats.get("innings", 0)

    pp_sr  = stats.get("pp_sr")
    mid_sr = stats.get("middle_sr")
    d_sr   = stats.get("death_sr")
    r_avg  = stats.get("recent_avg", 0)
    r_sr   = stats.get("recent_sr", 0)
    bdry   = stats.get("boundary_pct", 0)
    dot    = stats.get("dot_ball_pct", 0)

    sr_col  = _color_metric(sr,  good_high=True,  thresholds=(110, 145))
    avg_col = _color_metric(avg, good_high=True,  thresholds=(20, 35))

    phase_html = ""
    if pp_sr  is not None: phase_html += f"<span style='color:#90caf9'>PP:{pp_sr:.0f}</span> &nbsp;"
    if mid_sr is not None: phase_html += f"<span style='color:#a5d6a7'>Mid:{mid_sr:.0f}</span> &nbsp;"
    if d_sr   is not None: phase_html += f"<span style='color:#ef9a9a'>Death:{d_sr:.0f}</span>"

    st.markdown(
        f"<div style='background:#1e2130;border-radius:8px;padding:10px 14px;"
        f"margin:4px 0;border-left:3px solid #1976d2'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='color:#e0e0e0;font-weight:600'>🏏 {player}</span>"
        f"<span style='font-size:0.75rem;color:#888'>{inn} innings · {runs} runs</span>"
        f"</div>"
        f"<div style='margin-top:6px;display:flex;gap:14px;flex-wrap:wrap'>"
        f"<span>SR <b style='color:{sr_col}'>{sr:.0f}</b></span>"
        f"<span>Avg <b style='color:{avg_col}'>{avg:.1f}</b></span>"
        f"<span>Bdry <b style='color:#ffa726'>{bdry:.0f}%</b></span>"
        f"<span>Dot <b style='color:#888'>{dot:.0f}%</b></span>"
        f"<span style='font-size:0.8rem;color:#aaa'>Recent: {r_avg:.0f} avg / {r_sr:.0f} SR</span>"
        f"</div>"
        f"<div style='margin-top:4px;font-size:0.78rem'>{phase_html}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

def _player_bowling_card(player, stats):
    """Render a compact bowling stat card for one player."""
    if not stats:
        st.markdown(
            f"<div style='background:#1e2130;border-radius:8px;padding:10px;"
            f"margin:4px 0;color:#888;font-size:0.85rem'>"
            f"🎳 <b>{player}</b> — no bowling data found</div>",
            unsafe_allow_html=True
        )
        return

    econ = stats.get("economy", 0)
    avg  = stats.get("bowling_avg", 0)
    wkts = stats.get("wickets", 0)
    inn  = stats.get("innings", 0)
    sr   = stats.get("bowling_sr", 0)

    pp_e  = stats.get("pp_economy")
    mid_e = stats.get("middle_economy")
    d_e   = stats.get("death_economy")
    r_e   = stats.get("recent_economy", econ)
    r_w   = stats.get("recent_wickets", 0)
    dot   = stats.get("dot_ball_pct", 0)

    econ_col = _color_metric(econ, good_high=False, thresholds=(7.5, 9.5))
    avg_col  = _color_metric(avg,  good_high=False, thresholds=(22, 35))

    phase_html = ""
    if pp_e  is not None: phase_html += f"<span style='color:#90caf9'>PP:{pp_e:.1f}</span> &nbsp;"
    if mid_e is not None: phase_html += f"<span style='color:#a5d6a7'>Mid:{mid_e:.1f}</span> &nbsp;"
    if d_e   is not None: phase_html += f"<span style='color:#ef9a9a'>Death:{d_e:.1f}</span>"

    st.markdown(
        f"<div style='background:#1e2130;border-radius:8px;padding:10px 14px;"
        f"margin:4px 0;border-left:3px solid #e53935'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='color:#e0e0e0;font-weight:600'>🎳 {player}</span>"
        f"<span style='font-size:0.75rem;color:#888'>{inn} matches · {wkts} wkts</span>"
        f"</div>"
        f"<div style='margin-top:6px;display:flex;gap:14px;flex-wrap:wrap'>"
        f"<span>Econ <b style='color:{econ_col}'>{econ:.2f}</b></span>"
        f"<span>Avg <b style='color:{avg_col}'>{avg:.1f}</b></span>"
        f"<span>SR <b style='color:#ffa726'>{sr:.1f}</b></span>"
        f"<span>Dot <b style='color:#888'>{dot:.0f}%</b></span>"
        f"<span style='font-size:0.8rem;color:#aaa'>Recent: {r_e:.1f} econ / {r_w} wkts</span>"
        f"</div>"
        f"<div style='margin-top:4px;font-size:0.78rem'>{phase_html}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

def _render_player_scout(team1_name, team2_name):
    st.markdown("# 🔍 Player Scout")
    st.markdown(f"**{team1_name}** vs **{team2_name}** — individual player stats from historical data")
    st.markdown("---")

    if bat_df is None or bowl_df is None:
        st.error(
            "❌ Player data not available.\n\n"
            "Make sure `deliveries_clean.csv` exists in `data/` directory.\n\n"
            "Run: `python 1_data_cleaning.py` then `python 6_player_features.py`"
        )
        return

    bat_lookup  = bat_df.set_index("player").to_dict("index")  if len(bat_df) > 0 else {}
    bowl_lookup = bowl_df.set_index("player").to_dict("index") if len(bowl_df) > 0 else {}

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 Squad Comparison", "🔎 Player Search", "⚔️ Head to Head"])

    # ══════════════════════════════════════════════════════
    # TAB 1: Squad Comparison — side-by-side XIs
    # ══════════════════════════════════════════════════════
    with tab1:
        # Load playing XIs from live data if available
        live = load_live_match()
        xi1_default = live.get("team1_xi", []) if live else []
        xi2_default = live.get("team2_xi", []) if live else []

        st.markdown("**Enter Playing XIs** (one name per line, or auto-loaded from Live Data)")

        c1, c2 = st.columns(2)
        with c1:
            xi1_raw = st.text_area(
                f"🔵 {team1_name} XI",
                value="\n".join(xi1_default) if xi1_default else "",
                height=220,
                key="scout_xi1",
                placeholder="e.g.\nRohit Sharma\nV Kohli\n..."
            )
        with c2:
            xi2_raw = st.text_area(
                f"🔴 {team2_name} XI",
                value="\n".join(xi2_default) if xi2_default else "",
                height=220,
                key="scout_xi2",
                placeholder="e.g.\nMS Dhoni\nRavindra Jadeja\n..."
            )

        xi1 = [p.strip() for p in xi1_raw.strip().splitlines() if p.strip()]
        xi2 = [p.strip() for p in xi2_raw.strip().splitlines() if p.strip()]

        if not xi1 and not xi2:
            st.info("👆 Enter at least one player name above to see stats. Names must match Cricsheet data exactly (e.g. 'V Kohli', 'RG Sharma').")
            return

        st.markdown("---")

        # ── Batting Section ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">🏏 Batting Stats</div>', unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)

        with bc1:
            st.markdown(f"**{team1_name}**")
            if xi1:
                for p in xi1:
                    _player_batting_card(p, bat_lookup.get(p, {}))
            else:
                st.caption("No players entered")

        with bc2:
            st.markdown(f"**{team2_name}**")
            if xi2:
                for p in xi2:
                    _player_batting_card(p, bat_lookup.get(p, {}))
            else:
                st.caption("No players entered")

        # ── Bowling Section ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🎳 Bowling Stats</div>', unsafe_allow_html=True)
        bwc1, bwc2 = st.columns(2)

        with bwc1:
            st.markdown(f"**{team1_name}**")
            if xi1:
                for p in xi1:
                    _player_bowling_card(p, bowl_lookup.get(p, {}))
            else:
                st.caption("No players entered")

        with bwc2:
            st.markdown(f"**{team2_name}**")
            if xi2:
                for p in xi2:
                    _player_bowling_card(p, bowl_lookup.get(p, {}))
            else:
                st.caption("No players entered")

        # ── Summary Table ────────────────────────────────────────────────────
        if xi1 or xi2:
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Summary Table</div>', unsafe_allow_html=True)

            rows = []
            for team_name, xi in [(team1_name, xi1), (team2_name, xi2)]:
                for p in xi:
                    b = bat_lookup.get(p, {})
                    w = bowl_lookup.get(p, {})
                    rows.append({
                        "Team"   : team_name,
                        "Player" : p,
                        "Bat SR" : f"{b.get('strike_rate', 0):.0f}" if b else None,
                        "Bat Avg": f"{b.get('batting_avg', 0):.1f}" if b else None,
                        "Runs"   : int(b.get("runs", 0)) if b else None,
                        "Bowl Eco": f"{w.get('economy', 0):.2f}" if w else None,
                        "Wkts"   : int(w.get("wickets", 0)) if w else None,
                    })
            if rows:
                _summary_df = clean_dataframe(pd.DataFrame(rows))
                st.dataframe(_summary_df, width="stretch", hide_index=True)

    # ══════════════════════════════════════════════════════
    # TAB 2: Player Search
    # ══════════════════════════════════════════════════════
    with tab2:
        st.markdown("Search any player by name and view their full career stats.")

        search_q = st.text_input("🔎 Search player name", placeholder="e.g. Kohli, Bumrah, Dhoni...")

        if search_q:
            all_batters  = set(bat_df["player"].unique()) if bat_df is not None and len(bat_df) > 0 else set()
            all_bowlers  = set(bowl_df["player"].unique()) if bowl_df is not None and len(bowl_df) > 0 else set()
            all_players  = sorted(all_batters | all_bowlers)
            matches_list = [p for p in all_players if search_q.lower() in p.lower()]

            if not matches_list:
                st.warning(f"No players found matching '{search_q}'. Try a shorter name or different spelling.")
            else:
                selected = st.selectbox("Select player:", matches_list)

                if selected:
                    bat_s  = bat_lookup.get(selected, {})
                    bowl_s = bowl_lookup.get(selected, {})

                    st.markdown(f"### {selected}")
                    pc1, pc2 = st.columns(2)

                    with pc1:
                        st.markdown("**🏏 Batting**")
                        if bat_s:
                            st.metric("Strike Rate",   f"{bat_s.get('strike_rate', 0):.1f}")
                            st.metric("Batting Avg",   f"{bat_s.get('batting_avg', 0):.1f}")
                            st.metric("Total Runs",    str(bat_s.get("runs", 0)))
                            st.metric("Innings",       str(bat_s.get("innings", 0)))
                            st.metric("Boundary %",    f"{bat_s.get('boundary_pct', 0):.1f}%")
                            st.metric("Dot Ball %",    f"{bat_s.get('dot_ball_pct', 0):.1f}%")

                            st.markdown("**Phase Strike Rates**")
                            ph_cols = st.columns(3)
                            with ph_cols[0]:
                                st.metric("Powerplay", f"{bat_s.get('pp_sr') or '—'}")
                            with ph_cols[1]:
                                st.metric("Middle",    f"{bat_s.get('middle_sr') or '—'}")
                            with ph_cols[2]:
                                st.metric("Death",     f"{bat_s.get('death_sr') or '—'}")

                            st.markdown("**Recent Form (last 5 matches)**")
                            rf_cols = st.columns(2)
                            with rf_cols[0]:
                                st.metric("Avg Runs",    f"{bat_s.get('recent_avg', 0):.1f}")
                            with rf_cols[1]:
                                st.metric("Strike Rate", f"{bat_s.get('recent_sr', 0):.1f}")
                        else:
                            st.info("No batting data found for this player.")

                    with pc2:
                        st.markdown("**🎳 Bowling**")
                        if bowl_s:
                            st.metric("Economy",       f"{bowl_s.get('economy', 0):.2f}")
                            st.metric("Bowling Avg",   f"{bowl_s.get('bowling_avg', 0):.1f}")
                            st.metric("Wickets",       str(bowl_s.get("wickets", 0)))
                            st.metric("Bowling SR",    f"{bowl_s.get('bowling_sr', 0):.1f}")
                            st.metric("Dot Ball %",    f"{bowl_s.get('dot_ball_pct', 0):.1f}%")
                            st.metric("Bdry Conceded", f"{bowl_s.get('boundary_pct_given', 0):.1f}%")

                            st.markdown("**Phase Economy**")
                            bph_cols = st.columns(3)
                            with bph_cols[0]:
                                st.metric("Powerplay", f"{bowl_s.get('pp_economy') or '—'}")
                            with bph_cols[1]:
                                st.metric("Middle",    f"{bowl_s.get('middle_economy') or '—'}")
                            with bph_cols[2]:
                                st.metric("Death",     f"{bowl_s.get('death_economy') or '—'}")

                            st.markdown("**Recent Form (last 5 matches)**")
                            rb_cols = st.columns(2)
                            with rb_cols[0]:
                                st.metric("Economy",  f"{bowl_s.get('recent_economy', 0):.2f}")
                            with rb_cols[1]:
                                st.metric("Wickets",  str(bowl_s.get("recent_wickets", 0)))
                        else:
                            st.info("No bowling data found for this player.")
        else:
            # Show top batters and bowlers by default
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**🏏 Top 15 Batters by Strike Rate** (min 30 balls)")
                if bat_df is not None and len(bat_df) > 0:
                    top_bat = bat_df.nlargest(15, "strike_rate")[
                        ["player", "innings", "runs", "strike_rate", "batting_avg", "boundary_pct"]
                    ].rename(columns={
                        "player": "Player", "innings": "Inn", "runs": "Runs",
                        "strike_rate": "SR", "batting_avg": "Avg", "boundary_pct": "Bdry%"
                    })
                    top_bat = clean_dataframe(top_bat)
                    st.dataframe(top_bat, width="stretch", hide_index=True)
            with col_b:
                st.markdown("**🎳 Top 15 Bowlers by Economy** (min 30 balls)")
                if bowl_df is not None and len(bowl_df) > 0:
                    top_bowl = bowl_df.nsmallest(15, "economy")[
                        ["player", "innings", "wickets", "economy", "bowling_avg", "bowling_sr"]
                    ].rename(columns={
                        "player": "Player", "innings": "Inn", "wickets": "Wkts",
                        "economy": "Econ", "bowling_avg": "Avg", "bowling_sr": "SR"
                    })
                    top_bowl = clean_dataframe(top_bowl)
                    st.dataframe(top_bowl, width="stretch", hide_index=True)

    # ══════════════════════════════════════════════════════
    # TAB 3: Head to Head Matchups
    # ══════════════════════════════════════════════════════
    with tab3:
        st.markdown("**Batter vs Bowler matchup stats** — minimum 6 balls faced")
        st.markdown("Enter one batter and one bowler to see their historical matchup.")

        live2 = load_live_match()
        xi1_h = live2.get("team1_xi", []) if live2 else []
        xi2_h = live2.get("team2_xi", []) if live2 else []

        all_bat_names  = sorted(bat_df["player"].unique()) if bat_df is not None and len(bat_df) > 0 else []
        all_bowl_names = sorted(bowl_df["player"].unique()) if bowl_df is not None and len(bowl_df) > 0 else []

        hh_col1, hh_col2 = st.columns(2)
        with hh_col1:
            batter_sel = st.selectbox("Select Batter", options=[""] + all_bat_names, key="h2h_batter")
        with hh_col2:
            bowler_sel = st.selectbox("Select Bowler", options=[""] + all_bowl_names, key="h2h_bowler")

        if batter_sel and bowler_sel and del_df is not None:
            h2h_data = compute_h2h(del_df)
            key = (batter_sel, bowler_sel)
            stats_h2h = h2h_data.get(key, {})

            if stats_h2h:
                st.markdown(f"### {batter_sel} vs {bowler_sel}")
                hmc1, hmc2, hmc3, hmc4 = st.columns(4)
                with hmc1:
                    st.metric("Balls Faced", stats_h2h["balls"])
                with hmc2:
                    st.metric("Runs Scored", stats_h2h["runs"])
                with hmc3:
                    st.metric("Wickets Lost", stats_h2h["wkts"])
                with hmc4:
                    sr_h2h = stats_h2h["sr"]
                    sr_col = _color_metric(sr_h2h, good_high=True, thresholds=(100, 140))
                    st.markdown(
                        f"<div style='text-align:center'>"
                        f"<div style='font-size:0.85rem;color:#888'>Strike Rate</div>"
                        f"<div style='font-size:1.8rem;font-weight:bold;color:{sr_col}'>{sr_h2h}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                adv = "🏏 **Batter advantage**" if sr_h2h >= 130 else (
                      "🎳 **Bowler advantage**" if sr_h2h < 100 else "⚖️ **Even matchup**")
                st.info(adv + f" — {batter_sel} scores at SR {sr_h2h} off {stats_h2h['balls']} balls "
                              f"({stats_h2h['wkts']} dismissal{'s' if stats_h2h['wkts'] != 1 else ''})")
            else:
                st.warning(
                    f"No matchup data found (need ≥6 balls). "
                    f"{batter_sel} and {bowler_sel} may not have faced each other in this dataset."
                )
        elif batter_sel or bowler_sel:
            st.info("Select both a batter and a bowler to see their H2H stats.")
        else:
            st.info("👆 Select a batter and a bowler above to explore their historical matchup.")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏏 PitchMind")
    st.markdown("### IPL Match Predictor")
    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────
    page = st.radio(
        "📌 Navigate",
        ["🎯 Match Predictor", "🔍 Player Scout"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    if df is not None:
        all_teams  = sorted(set(df["team1"].unique()) | set(df["team2"].unique()))
        all_venues = sorted(df["venue"].unique())
    else:
        all_teams = [
            "Chennai Super Kings", "Mumbai Indians",
            "Royal Challengers Bengaluru", "Kolkata Knight Riders",
            "Delhi Capitals", "Punjab Kings", "Rajasthan Royals",
            "Sunrisers Hyderabad", "Lucknow Super Giants", "Gujarat Titans",
        ]
        all_venues = [
            "Wankhede Stadium", "M Chinnaswamy Stadium",
            "Eden Gardens", "MA Chidambaram Stadium",
            "Arun Jaitley Stadium", "Rajiv Gandhi International Stadium",
        ]

    st.markdown("### ⚙️ Match Setup")
    team1 = st.selectbox("🔵 Team 1", all_teams, index=0)
    team2 = st.selectbox(
        "🔴 Team 2",
        [t for t in all_teams if t != team1],
        index=1
    )
    venue         = st.selectbox("🏟️ Venue", all_venues)
    toss_winner   = st.selectbox("🪙 Toss Winner", [team1, team2])
    toss_decision = st.selectbox("Toss Decision", ["field", "bat"])

    st.markdown("---")
    predict_btn = st.button(
        "🎯 PREDICT WINNER",
        width="stretch",
        type="primary"
    )

    if xgb_model is None:
        st.error("⚠️ Model not found.\nRun `python 3_train_model.py` first.")
    else:
        st.success("✅ Model loaded")

    st.markdown("---")
    st.markdown(
        "<small style='color:#666'>PitchMind v1.0 | XGBoost + RF Ensemble</small>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎯 Match Predictor":
    render_live_panel(team1, team2)

    st.markdown("# 🏏 " + team1 + "  **vs**  " + team2)
    st.markdown("📍 **" + venue + "**  &nbsp;|&nbsp;  🪙 **" + toss_winner + "** won toss → chose to **" + toss_decision + "**")
    st.markdown("---")

    # Build prediction
    if feature_cols is not None:
        X_input, s1, s2, h2h, venue_avg = build_feature_vector(
            team1, team2, venue, toss_winner, toss_decision, df, feature_cols
        )
        prob_t1, prob_t2 = predict_winner(X_input)
    else:
        prob_t1, prob_t2 = 0.5, 0.5
        s1        = get_team_stats(team1, "team1", venue, df)
        s2        = get_team_stats(team2, "team2", venue, df)
        h2h       = 0.5
        venue_avg = 160.0

    winner_label = team1 if prob_t1 >= 0.5 else team2
    winner_prob  = max(prob_t1, prob_t2)

    # ── WIN PROBABILITY ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Win Probability</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([5, 2, 5])
    with col1:
        st.markdown(
            '<div class="win-box-blue">'
            '<h2 style="margin:0; font-size:1.3rem">' + team1 + '</h2>'
            '<h1 style="font-size:3.8rem; margin:10px 0">' + f"{prob_t1:.0%}" + '</h1>'
            '<p style="margin:0; opacity:0.8">Win Probability</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; color:#ffa726'>VS</h2>", unsafe_allow_html=True)

    with col3:
        st.markdown(
            '<div class="win-box-red">'
            '<h2 style="margin:0; font-size:1.3rem">' + team2 + '</h2>'
            '<h1 style="font-size:3.8rem; margin:10px 0">' + f"{prob_t2:.0%}" + '</h1>'
            '<p style="margin:0; opacity:0.8">Win Probability</p>'
            '</div>',
            unsafe_allow_html=True
        )

    # Probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 0.7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.barh(0, prob_t1, color="#1976d2", height=0.6)
    ax.barh(0, prob_t2, left=prob_t1, color="#e53935", height=0.6)
    ax.text(prob_t1 / 2,           0, f"{prob_t1:.0%}", ha="center", va="center",
            color="white", fontweight="bold", fontsize=12)
    ax.text(prob_t1 + prob_t2 / 2, 0, f"{prob_t2:.0%}", ha="center", va="center",
            color="white", fontweight="bold", fontsize=12)
    ax.set_xlim(0, 1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig, width="stretch")
    plt.close()

    st.success("🏆 **Predicted Winner: " + winner_label + "** with **" + f"{winner_prob:.0%}" + "** confidence")
    st.markdown("---")


    # ── TEAM COMPARISON ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Team Comparison</div>', unsafe_allow_html=True)

    comp = {
        "Metric": [
            "Overall Win Rate", "Recent Form (last 5)", "Avg Runs Scored",
            "Powerplay Runs", "Middle Overs Runs", "Death Overs Runs",
            "Boundary %", "Dot Ball %", "Run Rate",
            "Bowling Economy", "Death Economy", "PP Wickets Taken",
            "Venue Win Rate"
        ],
        team1: [
            f"{s1['win_rate']:.1%}",
            f"{s1['recent_form']:.1%}",
            f"{s1['avg_runs']:.0f}",
            f"{s1['powerplay_runs']:.0f}",
            f"{s1['middle_runs']:.0f}",
            f"{s1['death_runs']:.0f}",
            f"{s1['boundary_pct']:.1%}",
            f"{s1['dot_ball_pct']:.1%}",
            f"{s1['run_rate']:.1f}",
            f"{s1['bowling_economy']:.1f}",
            f"{s1['death_economy']:.1f}",
            f"{s1['pp_wickets']:.1f}",
            f"{s1['venue_win_rate']:.1%}",
        ],
        team2: [
            f"{s2['win_rate']:.1%}",
            f"{s2['recent_form']:.1%}",
            f"{s2['avg_runs']:.0f}",
            f"{s2['powerplay_runs']:.0f}",
            f"{s2['middle_runs']:.0f}",
            f"{s2['death_runs']:.0f}",
            f"{s2['boundary_pct']:.1%}",
            f"{s2['dot_ball_pct']:.1%}",
            f"{s2['run_rate']:.1f}",
            f"{s2['bowling_economy']:.1f}",
            f"{s2['death_economy']:.1f}",
            f"{s2['pp_wickets']:.1f}",
            f"{s2['venue_win_rate']:.1%}",
        ],
    }
    comp_df = pd.DataFrame(comp).set_index("Metric")
    comp_df = clean_dataframe(comp_df)
    st.dataframe(comp_df, width="stretch")
    st.markdown("---")


    # ── KEY FACTORS ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ Key Match Factors</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if toss_decision == "field":
            st.info("**🪙 Toss**\n\n✅ " + toss_winner + " chose to field — chasing advantage")
        else:
            st.info("**🪙 Toss**\n\n⚠️ " + toss_winner + " chose to bat — sets a target")

    with col2:
        if h2h > 0.5:
            st.info("**📊 Head to Head**\n\n✅ " + team1 + " leads H2H (" + f"{h2h:.0%}" + ")")
        elif h2h < 0.5:
            st.info("**📊 Head to Head**\n\n✅ " + team2 + " leads H2H (" + f"{1 - h2h:.0%}" + ")")
        else:
            st.info("**📊 Head to Head**\n\nEven record between these teams")

    with col3:
        wr_diff = s1["win_rate"] - s2["win_rate"]
        if abs(wr_diff) < 0.05:
            st.info("**📈 Win Rate**\n\nVery evenly matched teams")
        elif wr_diff > 0:
            st.info("**📈 Win Rate**\n\n✅ " + team1 + " has better overall win rate")
        else:
            st.info("**📈 Win Rate**\n\n✅ " + team2 + " has better overall win rate")

    st.markdown("---")


    # ── VENUE ANALYSIS ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏟️ Venue Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Score at Venue", f"{venue_avg:.0f} runs")
    with col2:
        st.metric(team1 + " Venue Win Rate", f"{s1['venue_win_rate']:.0%}")
    with col3:
        st.metric(team2 + " Venue Win Rate", f"{s2['venue_win_rate']:.0%}")

    st.markdown("---")


    # ── BAR CHART: BATTING vs BOWLING ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Batting vs Bowling Comparison</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#1e2130")
        ax.set_facecolor("#1e2130")
        categories = ["Avg Runs", "PP Runs", "Mid Runs", "Death Runs"]
        v1 = [s1["avg_runs"], s1["powerplay_runs"], s1["middle_runs"], s1["death_runs"]]
        v2 = [s2["avg_runs"], s2["powerplay_runs"], s2["middle_runs"], s2["death_runs"]]
        x  = np.arange(len(categories))
        w  = 0.35
        ax.bar(x - w/2, v1, w, label=team1, color="#1976d2")
        ax.bar(x + w/2, v2, w, label=team2, color="#e53935")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.legend(fontsize=8, facecolor="#1e2130", labelcolor="white")
        ax.set_title("Batting Stats", color="white", fontsize=11)
        for spine in ax.spines.values():
            spine.set_color("#444")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#1e2130")
        ax.set_facecolor("#1e2130")
        categories = ["Economy", "Death Econ", "PP Wickets"]
        v1 = [s1["bowling_economy"], s1["death_economy"], s1["pp_wickets"]]
        v2 = [s2["bowling_economy"], s2["death_economy"], s2["pp_wickets"]]
        x  = np.arange(len(categories))
        w  = 0.35
        ax.bar(x - w/2, v1, w, label=team1, color="#1976d2")
        ax.bar(x + w/2, v2, w, label=team2, color="#e53935")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.legend(fontsize=8, facecolor="#1e2130", labelcolor="white")
        ax.set_title("Bowling Stats", color="white", fontsize=11)
        for spine in ax.spines.values():
            spine.set_color("#444")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

    st.markdown("---")


    # ── ABOUT MODEL ───────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        **Model:** XGBoost + Random Forest Ensemble (averaged probabilities)

        **45 Features used:**
        - Team win rates, recent form (last 5), head-to-head record, NRR
        - Avg runs, powerplay runs, **middle overs runs**, death overs runs
        - Run rate, top-3 batsman strike rate, boundary %, **dot ball %**
        - Bowling economy, death economy, PP wickets taken
        - Venue win rates, venue average runs
        - Toss win, toss decision, toss + field combination
        - **10 difference features** (team1 vs team2 relative advantage)

        **Dataset:** IPL 2008–2025 | ~1,169 matches

        **Training:** Time-aware split — trained on older seasons, tested on 2024–2025

        **Limitations:**
        - Does not account for player injuries or playing XI composition
        - Cannot predict rain interruptions or pitch deterioration mid-game
        """)




# ══════════════════════════════════════════════════════════════════════════════
# PLAYER SCOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Player Scout":
    _render_player_scout(team1, team2)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem'>"
    "PitchMind 🏏 | Built with Python · XGBoost · Streamlit | Educational Use Only"
    "</p>",
    unsafe_allow_html=True
)