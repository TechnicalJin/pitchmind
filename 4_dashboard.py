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
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏏 PitchMind")
    st.markdown("### IPL Match Predictor")
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
        use_container_width=True,
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
st.pyplot(fig, use_container_width=True)
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
st.dataframe(comp_df, use_container_width=True)
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
    st.pyplot(fig, use_container_width=True)
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
    st.pyplot(fig, use_container_width=True)
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


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem'>"
    "PitchMind 🏏 | Built with Python · XGBoost · Streamlit | Educational Use Only"
    "</p>",
    unsafe_allow_html=True
)