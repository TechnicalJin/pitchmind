# PitchMind — Comprehensive Project Dependency Map

**Generated:** 2026-03-27
**Project:** IPL Cricket Match Prediction System
**Technology Stack:** Python 3.8+, Pandas, XGBoost, Random Forest, Streamlit, Scikit-learn

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [File-by-File Dependencies](#file-by-file-dependencies)
3. [Dependency Graph (Adjacency List)](#dependency-graph-adjacency-list)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [Module Import Flow (ASCII)](#module-import-flow-ascii)
6. [Circular Dependency Analysis](#circular-dependency-analysis)
7. [Unused Files & Dead Code](#unused-files--dead-code)
8. [Core Entry Points](#core-entry-points)
9. [External Libraries & Import Map](#external-libraries--import-map)

---

## Executive Summary

**Project Structure:**
- **17 Python files** organized in a linear 8-step pipeline
- **Zero circular dependencies** detected
- **4 core utilities** supporting the pipeline
- **Sequential execution** model (each step depends on previous output)

**Data Flow:**
```
Raw JSON Data
    ↓
0_json_to_csv.py (JSON → CSV)
    ↓
1_data_cleaning.py (Clean CSVs)
    ↓
2_feature_engineering.py (Engineer 63+ features)
    ↓
3_train_model.py (Train XGBoost + Random Forest)
    ↓
models/*.pkl (Trained models)
    ↓
4_dashboard.py (Streamlit UI) ← 5_live_data_fetch.py (Live data)
    ↓
6_player_features.py (Player stats)
7_phase_predictor.py (Phase models)
```

---

## File-by-File Dependencies

### **Core Pipeline Files**

#### **0_json_to_csv.py** — JSON to CSV Converter
**Purpose:** Converts Cricsheet IPL JSON files to matches.csv + deliveries.csv
**Input:** `data/Data_Cricsheet/*.json` (1169+ files)
**Output:** `data/matches.csv`, `data/deliveries.csv`
**Direct Dependencies:**
- `utils.py` → `normalize_team()`
- **External:** pandas, pathlib, json, os

**Functions:**
- `parse_match(filepath)` — Parse single match JSON
- `main()` — Orchestrate batch processing

**Used By:** `1_data_cleaning.py`

---

#### **1_data_cleaning.py** — Data Cleaning & Normalization
**Purpose:** Clean matches.csv and deliveries.csv for downstream features
**Input:** `data/matches.csv`, `data/deliveries.csv`
**Output:** `data/matches_clean.csv`, `data/deliveries_clean.csv`
**Direct Dependencies:**
- `utils.py` → `normalize_team()`, `apply_team_normalization()`
- **External:** pandas, numpy, os

**Functions:**
- `clean_matches(filepath)` — Drop null columns, normalize teams
- `clean_deliveries(filepath)` — Convert to numeric, lowercase columns
- `save_cleaned_data(matches, deliveries)` — Save outputs

**Used By:** `2_feature_engineering.py`

---

#### **2_feature_engineering.py** — Feature Engineering (63 Features)
**Purpose:** Engineer temporal, contextual, squad-level features for match prediction
**Input:** `data/matches.csv`, `data/deliveries.csv` (clean versions)
**Output:** `data/master_features.csv`
**Direct Dependencies:**
- `utils.py` → `normalize_team()`, `apply_team_normalization()`, `validate_toss_distribution()`, `validate_team_name_consistency()`
- **External:** pandas, numpy, os, warnings

**Key Functions:**
- `load_data()` — Load & normalize cleaned data
- `compute_rolling_win_rate()` — Expanding win % per team
- `compute_rolling_recent_form()` — EWM-weighted last 5 matches
- `compute_rolling_h2h()` — Head-to-head rolling win rate
- `compute_rolling_nrr()` — Net run rate from deliveries
- `compute_rolling_venue_win_rate()` — Venue-specific win rate
- `compute_rolling_venue_avg_runs()` — Venue first innings average
- `compute_ewm_delivery_features()` — EWM batting/bowling rolling stats
- `compute_rolling_chase_win_rate()` — When batting 2nd
- `compute_rolling_home_win_rate()` — Home ground win rate
- `compute_win_streak()` — Consecutive wins entering match
- `compute_days_rest()` — Days since last match (fatigue proxy)
- `compute_squad_features()` — Player-level XI strength from deliveries
- `compute_season_stage()` — 0=group, 1=playoff
- `build_master_features()` — Orchestrate all feature engineering
- `validate_and_print()` — Data quality checks, null validation, leakage checks

**Features Engineered (63 total):**
- Team strength: win_rate, recent_form, NRR (3)
- Batting: avg_runs, pp/middle/death_runs, run_rate, boundary%, dot%, top3_sr (14)
- Bowling: economy, death_economy, pp_wickets, bowling_sr (8)
- Venue: venue_win_rate, venue_avg_runs (2)
- Toss context: toss_win, toss_field, toss_team1_field (3)
- Chase ability: chase_win_rate x2, home_win_rate x2 (4)
- Momentum: win_streak x2, days_rest x2, season_stage (5)
- Squad strength: squad_bat_sr x2, squad_bowl_econ x2, allrounder x2 (6)
- Difference features: diff_win_rate, diff_recent_form, diff_avg_runs, diff_death_runs, diff_death_economy, diff_bowling_economy, diff_pp_wickets, diff_run_rate, diff_nrr, diff_venue_win_rate, diff_chase_win_rate, diff_squad_bat_sr (12)

**Used By:** `3_train_model.py`

---

#### **3_train_model.py** — Model Training & Calibration
**Purpose:** Train XGBoost + Random Forest ensemble with season weighting, Optuna tuning, probability calibration
**Input:** `data/master_features.csv`
**Output:**
- `models/ipl_model.pkl` (XGBoost, Optuna params)
- `models/ipl_model_rf.pkl` (Random Forest)
- `models/ipl_model_calibrated.pkl` (Calibrated ensemble)
- `models/optuna_best_params.json`
- `models/feature_cols.pkl`

**Direct Dependencies:**
- **External:** pandas, numpy, joblib, optuna, sklearn (TimeSeriesSplit, calibration), xgboost

**Key Functions:**
- `load_data()` — Load master_features.csv
- `prepare_data()` — Extract features & target
- `compute_season_weights()` — Exponential weighting (recent seasons ~15x heavier)
- `split_data()` — 80/20 time-aware train/test
- `run_optuna_tuning()` — Bayesian hyperparameter search (30 trials)
- `run_cross_validation()` — TimeSeriesSplit 5-fold CV
- `train_random_forest()` — RF with season weights
- `train_xgboost()` — XGB with best Optuna params + season weights
- `build_raw_ensemble()` — Average RF & XGB probabilities
- `calibrate_models()` — Isotonic regression calibration
- `print_feature_importance()` — Top 15 features from RF & XGB
- `save_artifacts()` — Save all models & metadata

**Model Specifications:**
- **XGBoost:** n_estimators∈[200,700], max_depth∈[3,6], learning_rate∈[0.01,0.15], subsample/colsample∈[0.6,1.0]
- **Random Forest:** n_estimators=500, max_depth=8, min_samples_leaf=5, max_features='sqrt'
- **Calibration:** Isotonic regression on test set

**Used By:** `4_dashboard.py`

---

#### **4_dashboard.py** — Main Streamlit Dashboard
**Purpose:** Interactive IPL match prediction UI with tabs: Live Match, Match Predictor, Player Scout
**Input:** `data/master_features.csv`, `models/*.pkl`, live data JSON
**Output:** Web UI + live predictions
**Direct Dependencies:**
- `live_match_tab.py` → `render_live_match_tab()`
- `pitchmind_player_features.py` → `get_all_player_stats()`, `compute_batting_stats()`, `compute_bowling_stats()`, `compute_h2h()`, `search_players()`
- `name_resolver.py` → `resolve_name()`, `get_resolution_stats()`
- **External:** streamlit, pandas, numpy, joblib, matplotlib, shap (optional), requests, json, datetime

**Key Sections:**
1. **Data Loading:**
   - `load_data()` → Load master_features.csv
   - `load_models()` → Load XGBoost + RF + feature_cols + SHAP explainer

2. **Player Data:**
   - `load_player_data()` → Load batting/bowling stats via `get_all_player_stats()`

3. **Prediction Engine:**
   - `get_team_stats(team, role, venue, df)` → Extract 23 features per team
   - `build_feature_vector()` → Assemble 63-feature input for model
   - `predict_winner()` → Ensemble probability (avg RF & XGB)

4. **Tabs:**
   - **🔴 Live Match:** Real-time score input → phase predictions → final innings projection
   - **🎯 Match Predictor:** Match setup → prediction + SHAP explanation + team comparison
   - **🔍 Player Scout:** Squad comparison, player search, H2H matchups

5. **SHAP Explainability:**
   - `_render_shap_explanation()` → Waterfall chart + natural language summary

**Used By:** End user (Streamlit app)

---

#### **5_live_data_fetch.py** — Live Match Data Fetcher
**Purpose:** Auto-fetch toss, playing XI, weather from ESPNcricinfo ~1hr before match
**Input:** ESPNcricinfo match ID (config)
**Output:** `data/live/todays_match.json`
**Direct Dependencies:**
- **External:** requests, json, os, datetime

**Key Functions:**
- `fetch_espncricinfo(match_id)` → Get toss + XI from ESPNcricinfo JSON API
- `fetch_weather(venue, api_key)` → Get temp, humidity, dew point from OpenWeatherMap
- `build_live_json()` → Merge auto-fetch + manual overrides
- `main()` — Orchestrate fetch + save

**Data Provided:**
- Match name, date, series, venue
- Toss winner, decision, pitch type, dew forecast
- Playing XIs (team1, team2)
- Weather (temp, humidity, dew point, description)
- Expert notes / commentary (manual paste from dashboard)

**Used By:** `4_dashboard.py`, `live_match_tab.py`

---

#### **6_player_features.py** & **pitchmind_player_features.py** — Player Stats Engine
**Purpose:** Compute per-player batting & bowling career stats from deliveries
**Input:** `data/deliveries_clean.csv`, `data/matches_clean.csv`
**Output:**
- `data/player_batting_stats.csv`
- `data/player_bowling_stats.csv`

**Direct Dependencies:**
- `name_resolver.py` → `resolve_name()`, `get_fallback_stats()` (optional, for new players)
- **External:** pandas, numpy, os, warnings

**Key Functions:**
- `load_deliveries()` → Load deliveries CSV
- `load_matches()` → Load matches CSV
- `compute_batting_stats()` — Per-batter: innings, runs, strike_rate, batting_avg, boundary%, dot%, phase SRs, recent form
- `compute_bowling_stats()` — Per-bowler: innings, wickets, economy, bowling_avg, bowling_sr, boundary%, phase economies, recent form
- `compute_h2h()` → Dict of (batter, bowler): {balls, runs, wickets, sr} (min 6 balls)
- `get_all_player_stats()` → Entry point: returns (bat_df, bowl_df)
- `get_player_stats()` → Single-player lookup
- `get_squad_stats()` → Full squad stats for 2 XIs
- `search_players()` → Partial name search

**Batting Stats (per player):**
- innings, runs, balls_faced, strike_rate, batting_avg, dismissals, not_outs
- fours, sixes, boundary%, dot_ball%
- pp_sr, middle_sr, death_sr (phase-wise)
- recent_avg, recent_sr (last 5 matches)

**Bowling Stats (per player):**
- innings, wickets, balls_bowled, economy, bowling_avg, bowling_sr
- dot_ball%, boundary% conceded
- pp_economy, middle_economy, death_economy
- pp_wickets, death_wickets
- recent_economy, recent_wickets

**Used By:** `4_dashboard.py` (Player Scout tab)

---

#### **7_phase_predictor.py** — In-Match Phase Run Predictor
**Purpose:** Train phase-wise run prediction models (Powerplay / Middle / Death)
**Input:** `data/deliveries_clean.csv`, `data/matches_clean.csv`
**Output:**
- `models/phase_pp_model.pkl`
- `models/phase_mid_model.pkl`
- `models/phase_death_model.pkl`
- `models/phase_batter_sr.pkl`
- `models/phase_bowler_eco.pkl`
- `models/phase_venue_factor.pkl`
- `models/phase_feature_cols.pkl`

**Direct Dependencies:**
- **External:** pandas, numpy, joblib, xgboost, sklearn.metrics

**Key Workflow:**
1. **Build LookUp Tables:**
   - `build_batter_sr_lookup()` → Career SR per player per phase
   - `build_bowler_eco_lookup()` → Career economy per bowler per phase
   - `build_venue_factor_lookup()` → Venue avg runs per phase

2. **Build Training Samples:**
   - `build_phase_samples()` → Create mid-phase snapshots (every ball, target = remaining runs in phase)
   - Features: runs_sofar, wickets_in_phase, balls_completed, phase_rr, striker_sr, non_striker_sr, bowler_economy, wickets_in_hand, batting_strength, partnership_runs, venue_avg_phase, venue_avg_total

3. **Train Models:**
   - `train_phase_model()` → XGBRegressor per phase (80/20 time-aware split)
   - MAE (Mean Absolute Error) ~8-12 runs

4. **Evaluation:**
   - `predict_phase()` → Predict remaining runs + confidence interval

**Used By:** `live_match_tab.py` (real-time phase predictions)

---

### **Support/Utility Files**

#### **utils.py** — Team Name Normalization & Validation
**Purpose:** Centralized team name handling, toss distribution validation
**Direct Dependencies:** pandas (for DataFrame operations)
**Functions:**
- `normalize_team(name: str) → str` — Convert to canonical team names
  - Handles: "Bangalore"→"Bengaluru", "Delhi Daredevils"→"Delhi Capitals", etc.
  - Removes extra spaces, converts to lowercase

- `apply_team_normalization(df, columns: list)` — In-place normalization of columns

- `validate_toss_distribution(df) → bool` — Check toss_win ≈ 0.50 (balanced)

- `validate_team_name_consistency(df, team_columns: list) → bool` — Ensure consistent team names across columns (team1, team2, toss_winner, winner)

- `compute_team1_toss_win(df)` — Compute team1_toss_win feature, detect bias

- `validate_toss_winner_in_teams(df) → int` — Verify toss_winner ∈ {team1, team2}

**Used By:** Files 0, 1, 2

**Canonical Teams:**
```
{
  "chennai super kings", "delhi capitals", "gujarat titans",
  "kolkata knight riders", "lucknow super giants", "mumbai indians",
  "punjab kings", "rajasthan royals", "royal challengers bengaluru",
  "sunrisers hyderabad",
  # Defunct:
  "rising pune supergiant", "pune warriors", "kochi tuskers kerala", "gujarat lions"
}
```

---

#### **name_resolver.py** — Player Name Resolution
**Purpose:** Resolve player names between formats (full "Jasprit Bumrah" ↔ cricsheet "JJ Bumrah")
**Input:**
- Cricsheet deliveries data (auto-detected player names)
- Manual mapping dict (200+ known mappings)
- Optional squad markdown file

**Output:** `data/name_map.json`
**Direct Dependencies:**
- **External:** pandas, rapidfuzz (optional, for fuzzy matching), json, os, re

**Key Functions:**
- `normalize_name(name: str)` — Lowercase, strip, remove dots/hyphens, collapse spaces
- `extract_surname(name: str)` → Last word
- `extract_initials(name: str)` → First letters of all words except surname
- `load_cricsheet_players()` → All unique players from deliveries
- `resolve_name(raw_name: str, use_cache=True)` — Resolution algorithm:
  1. Check cache
  2. Check exact match in cricsheet
  3. Check manual mapping (both directions)
  4. Try surname+initial match
  5. Try fuzzy match (rapidfuzz)
  6. Return None (new player)

- `resolve_squad_names(squad: List[str])` → Batch resolution
- `build_name_mapping()` → Build complete mapping from squad file
- `get_resolution_stats()` → Report: total players, matched, unmatched

**Manual Mappings (200+ entries):**
- Top stars: "jj bumrah"→"jasprit bumrah", "rg sharma"→"rohit sharma", etc.
- Covers all 10 IPL teams + international players

**Used By:**
- `4_dashboard.py` (Player Scout name resolution)
- `pitchmind_player_features.py` (squad stats lookup)
- `live_match_tab.py` (live XI name resolution)

---

#### **live_match_tab.py** — Live Match Prediction UI Component
**Purpose:** Streamlit tab for real-time in-match run predictions
**Input:**
- Manual score entry OR ESPNcricinfo auto-fetch
- Phase prediction models + lookup tables
- Player name resolution

**Output:** Streamlit UI element (called from `4_dashboard.py`)
**Direct Dependencies:**
- `name_resolver.py` → `resolve_name()`
- `models/phase_*.pkl` (XGBoost phase models + lookups)
- **External:** streamlit, pandas, numpy, matplotlib, requests, json, datetime

**Key UI Sections:**
1. **Match Setup:** Team selection, venue, data source
2. **Score Input:** Current runs, wickets, over, ball
3. **Crease:** Striker, non-striker, bowler (auto-resolved to cricsheet names)
4. **Phase Runs:** Historical runs in each phase (filled after phase ends)
5. **Live Metrics:** Score, run rate, balls left, wickets in hand
6. **Phase Prediction:** Remaining runs + confidence interval + phase total
7. **Final Projection:** Projected first innings total
8. **Breakdowns:** Phase-wise score bars (actual vs projected)
9. **Auto-Refresh:** Every 30s from ESPNcricinfo (optional)

**Used By:** `4_dashboard.py` (imported and called in Live Match tab)

---

#### **pitchmind_player_features.py** — Player Stats Entry Point (v2)
**Purpose:** Wrapper/alias for `6_player_features.py` with caching optimizations
**Direct Dependencies:** Same as `6_player_features.py`
**Difference from 6_player_features.py:**
- Module-level cache: compute stats once, lookup instantly in subsequent calls
- Optimized for Streamlit `@st.cache_data` decorator
- `get_all_player_stats()` entry point for dashboard

**Used By:** `4_dashboard.py`

---

#### **build_master_players.py** — (Detected, Content Unknown)
**Status:** File exists in git status but not examined
**Purpose:** Likely helper for master player list construction
**Estimated Usage:** Optional/deprecated support script

---

### **Web Scraper Module (Directory)**

#### **espn_live_scraper/** — Live Match Scraper
**Contents:**
- `app.py` — Flask app for scraping
- `scraper.py` — Scrape logic

**Purpose:** Alternative live data source (complements `5_live_data_fetch.py`)
**Status:** Separate from main pipeline, optional enhancement

---

## Dependency Graph (Adjacency List)

### Direct Dependencies

```
0_json_to_csv.py
├── utils.py (normalize_team)
└── pandas, pathlib, json, os

1_data_cleaning.py
├── utils.py (normalize_team, apply_team_normalization)
└── pandas, numpy, os

2_feature_engineering.py
├── utils.py (normalize_team, apply_team_normalization, validate_toss_distribution, validate_team_name_consistency)
└── pandas, numpy, os, warnings

3_train_model.py
├── pandas, numpy, joblib
├── optuna
├── sklearn (TimeSeriesSplit, CalibratedClassifierCV, FrozenEstimator)
└── xgboost

4_dashboard.py
├── live_match_tab.py (render_live_match_tab)
├── pitchmind_player_features.py (get_all_player_stats, compute_batting_stats, etc.)
├── name_resolver.py (resolve_name, get_resolution_stats)
├── Models: ipl_model.pkl, ipl_model_rf.pkl, feature_cols.pkl, optuna_best_params.json
└── External: streamlit, pandas, numpy, joblib, matplotlib, shap (optional), requests, json, datetime

5_live_data_fetch.py
├── requests, json, os, datetime
└── ESPNcricinfo API (external)

6_player_features.py / pitchmind_player_features.py
├── name_resolver.py (resolve_name, get_fallback_stats - optional)
└── pandas, numpy, os, warnings

7_phase_predictor.py
├── pandas, numpy, joblib, xgboost
└── sklearn.metrics

live_match_tab.py
├── name_resolver.py (resolve_name)
├── Models: phase_pp/mid/death_model.pkl, phase_batter_sr.pkl, phase_bowler_eco.pkl, phase_venue_factor.pkl
└── External: streamlit, pandas, numpy, matplotlib, requests, json, datetime

name_resolver.py
├── pandas, rapidfuzz (optional), json, os, re
└── Cricsheet player data (auto-detected)

utils.py
└── pandas

build_master_players.py
└── (Unknown - not examined)
```

---

## Data Flow Pipeline

### Sequential Pipeline (Linear Dependency Chain)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RAW DATA INPUTS                                 │
├─────────────────────────────────────────────────────────────────────┤
│  • IPL Cricsheet JSON (1169+ matches): data/Data_Cricsheet/*.json   │
│  • ESPNcricinfo API: https://espncricinfo.com/matches/engine/...    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                      ↓ Step 0
                   ┌──────────────────┐
                   │ 0_json_to_csv.py │
                   │  JSON → CSV      │
                   └──────────────────┘
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
      matches.csv          deliveries.csv
              │                         │
              └────────────┬────────────┘
                           │
                      ↓ Step 1
                  ┌────────────────────┐
                  │ 1_data_cleaning.py │
                  │  Normalize Teams   │
                  │  Clean NULLs       │
                  └────────────────────┘
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
  matches_clean.csv         deliveries_clean.csv
              │                         │
              └────────────┬────────────┘
                           │
                      ↓ Step 2
              ┌──────────────────────────┐
              │ 2_feature_engineering.py │
              │  Engineer 63 Features    │
              │  • Rolling Stats         │
              │  • EWM Weighting         │
              │  • Team/Venue/Toss       │
              │  • Squad Strength        │
              └──────────────────────────┘
                           │
                    ↓ master_features.csv
                           │
                      ↓ Step 3
              ┌──────────────────────────┐
              │  3_train_model.py        │
              │  • Optuna Tuning         │
              │  • Season Weights        │
              │  • Cross-Validation      │
              │  • Calibration           │
              └──────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   ipl_model.pkl    ipl_model_rf.pkl    ipl_model_calib.pkl
   feature_cols.pkl    optuna_params.json
        │                                    │
        └────────────┬───────────────────────┘
                     │
               ↓ Step 4 (Dashboard)
         ┌────────────────────────┐
         │  4_dashboard.py        │
         │  Streamlit UI          │
         │  • Match Predictor     │
         │  • Live Match Tab      │
         │  • Player Scout        │
         └────────────────────────┘
                  /│\
                 / │ \
        ┌───────┘  │  └───────┐
        │          │          │
    ↓ Step 5    ↓ Step 6    ↓ Step 7
    Live Data   Player      Phase
    Fetcher     Features    Predictor
```

### Parallel Branches (Non-Blocking)

```
Main Pipeline (Steps 0-4):
├─ Steps 0-3: Sequential (each depends on previous output)
└─ Step 4: Dashboard UI
    └─ Consumes outputs from:
       ├─ Step 3 (models)
       ├─ Step 5 (live data) [optional, external]
       ├─ Step 6 (player stats)
       └─ Step 7 (phase models)

Step 5 (5_live_data_fetch.py):
├─ Can run 1 hour before match
├─ Fetches from ESPNcricinfo API
└─ Populates data/live/todays_match.json

Step 6 (6_player_features.py):
├─ Depends on: deliveries_clean.csv (Step 1 output)
├─ Can run any time after Step 1
└─ Generates player career stats

Step 7 (7_phase_predictor.py):
├─ Depends on: deliveries_clean.csv + matches_clean.csv (Step 1 outputs)
├─ Can run any time after Step 1
└─ Generates phase-specific prediction models
```

---

## Module Import Flow (ASCII)

```
Dashboard Execution Tree
═════════════════════════════════════════════════════════════════════

4_dashboard.py (RUNTIME)
│
├─ SYNC LOAD: models/ipl_model.pkl (XGBoost)
│  └─ 3_train_model.py output
│
├─ SYNC LOAD: models/ipl_model_rf.pkl (Random Forest)
│  └─ 3_train_model.py output
│
├─ SYNC LOAD: models/feature_cols.pkl
│  └─ Feature column manifest (XGBoost booster names)
│
├─ SYNC LOAD: data/master_features.csv
│  └─ 2_feature_engineering.py output
│
├─ CACHE LOAD: Player data via pitchmind_player_features.py
│  │
│  ├─ Load deliveries_clean.csv (Step 1 output)
│  ├─ Load matches_clean.csv (Step 1 output)
│  ├─ compute_batting_stats(deliveries, matches)
│  ├─ compute_bowling_stats(deliveries)
│  └─ compute_h2h(deliveries)
│
├─ OPTIONAL: name_resolver.py
│  │
│  ├─ Load deliveries for player names
│  ├─ Load name_map.json (if exists)
│  └─ Cache-based name resolution
│
├─ DYNAMIC: live_match_tab.py (on demand)
│  │
│  ├─ Load models/phase_pp/mid/death_model.pkl
│  ├─ Load models/phase_batter_sr.pkl
│  ├─ Load models/phase_bowler_eco.pkl
│  ├─ Load models/phase_venue_factor.pkl
│  ├─ Load data/live/todays_match.json (optional)
│  └─ Call name_resolver.py for XI name resolution
│
└─ OPTIONAL: SHAP TreeExplainer (on demand)
   └─ shap.TreeExplainer(xgb_model)
```

---

## Circular Dependency Analysis

### Finding

**✅ ZERO circular dependencies detected**

### Analysis Process

1. **Direct Import Chains Examined:**
   - 0 → 1 → 2 → 3 → 4 (linear, no backlinks)
   - 4 imports: live_match_tab, pitchmind_player_features, name_resolver
   - None of these import 4 or its dependencies

2. **Utility Files (Utilities Pattern):**
   - `utils.py` → imported by 0, 1, 2 (no imports of these)
   - `name_resolver.py` → imported by 4, 6, live_match_tab (no imports of these)
   - `pitchmind_player_features.py` → imported by 4 (no imports of 4)

3. **Model Output Dependencies:**
   - Models are generated by 3, consumed by 4, 5 (no feedback to consumers)
   - Data flows uni-directionally

### Conclusion

**Pattern:** Strict hierarchical dependency tree with no cycles. Safe for:
- Sequential execution (Steps 0-4)
- Parallel execution (Steps 5-7 can run simultaneously)
- Caching and idempotence
- Module reloading without infinite recursion

---

## Unused Files & Dead Code

### Identified Unused/Deprecated Files

**`build_master_players.py`**
- **Status:** File exists in repository, not imported anywhere
- **Purpose:** Likely helper for constructing master player list (unknown without inspection)
- **Recommendation:**
  - If unused: remove to reduce project clutter
  - If used externally: document its purpose and re-integrate if needed

**`espn_live_scraper/` (Directory)**
- **Status:** Separate module, exists parallel to main pipeline
- **Purpose:** Alternative live scraper (Flask-based)
- **Note:** Main pipeline uses `5_live_data_fetch.py` for ESPNcricinfo fetching
- **Status:** Appears to be experimental/alternative implementation

---

## Core Entry Points

### User-Facing Entry Points

#### **1. Data Pipeline (Sequential Execution)**
```bash
# Step-by-step pipeline execution (in order):
python 0_json_to_csv.py              # JSON → raw CSVs
python 1_data_cleaning.py             # Clean & normalize
python 2_feature_engineering.py       # Engineer features
python 3_train_model.py               # Train models
```

**Requirements:**
- Cricsheet JSON files must be in `data/Data_Cricsheet/*.json`
- Output: models saved to `models/`

---

#### **2. Supporting Scripts (Can run after Step 0-1)**
```bash
python 5_live_data_fetch.py           # Fetch live data (1hr before match)
python 6_player_features.py           # Compute player stats
python 7_phase_predictor.py           # Train phase models
```

**Requirements:**
- Step 1 outputs (deliveries_clean.csv, matches_clean.csv) must exist
- Can run in parallel with Step 3

---

#### **3. Interactive Dashboard (Post-Training)**
```bash
streamlit run 4_dashboard.py
```

**Requirements:**
- All models saved to `models/`
- master_features.csv exists
- Accesses `live_match_tab.py`, `pitchmind_player_features.py`, `name_resolver.py`

**Features:**
- 🔴 Live Match: Real-time score tracking + phase predictions
- 🎯 Match Predictor: Pre-match probability + explanations
- 🔍 Player Scout: Squad analysis + head-to-head matchups

---

### Developer-Facing Entry Points

#### **`2_feature_engineering.py`**
- **Invoked by:** Step 3
- **Direct invocation:** `python 2_feature_engineering.py` (standalone testing)
- **Purpose:** Validate feature engineering independently

#### **`6_player_features.py`** & **`pitchmind_player_features.py`**
- **Invoked by:** Dashboard + Live Match tab
- **Direct invocation:** `python 6_player_features.py` (compute + save stats)
- **Purpose:** Standalone player stats generation or development testing

#### **`name_resolver.py`**
- **Invoked by:** Dashboard, Player Scout, Live Match tab
- **Direct invocation:** `python name_resolver.py [squad_file]` (build name mapping)
- **Purpose:** Generate/update `data/name_map.json` mapping

---

## External Libraries & Import Map

### Core Data Processing
| Library | Version | Usage | Files |
|---------|---------|-------|-------|
| `pandas` | 1.0+ | CSV loading, filtering, groupby, merging | All pipeline files |
| `numpy` | 1.0+ | Numeric operations, exp, quantiles | All pipeline files |
| `os`, `pathlib` | Built-in | File I/O, path operations | 0,1,2,5,7 |
| `json` | Built-in | JSON parsing, config save/load | 0,5,6,4 |

### Machine Learning
| Library | Version | Usage | Files |
|---------|---------|-------|-------|
| `xgboost` | 1.2+ | XGBoost classifier + regressor | 3, 7 |
| `sklearn` | 0.22+ | Random Forest, Time Series Split, Calibration | 3 |
| `optuna` | 2.0+ | Hyperparameter tuning (Bayesian) | 3 |

### UI & Visualization
| Library | Version | Usage | Files |
|---------|---------|-------|-------|
| `streamlit` | 1.0+ | Dashboard UI, caching, widgets | 4, live_match_tab |
| `matplotlib` | 3.0+ | Plotting (barh, line charts) | 4, live_match_tab |
| `shap` | 0.35+ | SHAP explainability (optional) | 4 |

### Utilities
| Library | Version | Usage | Files |
|---------|---------|-------|-------|
| `joblib` | 1.0+ | Model serialization (.pkl) | 3, 4, 7 |
| `requests` | 2.25+ | HTTP requests (ESPNcricinfo) | 5, live_match_tab |
| `rapidfuzz` | 1.0+ | Fuzzy string matching (optional) | name_resolver |
| `datetime` | Built-in | Timestamp handling | 4, 5, live_match_tab |
| `warnings` | Built-in | Suppress warnings | 2, 3, 6, 7 |

### Optional Dependencies
- **`shap`** → SHAP TreeExplainer in dashboard (graceful fallback if missing)
- **`rapidfuzz`** → Fuzzy name matching in resolver (falls back to surname+initial)

---

## Summary Table: Files & Their Roles

| File | Type | Role | Dependencies | Produces |
|------|------|------|--------------|----------|
| 0_json_to_csv.py | Pipeline | JSON → CSV | utils | matches.csv, deliveries.csv |
| 1_data_cleaning.py | Pipeline | Normalize & clean | utils | matches_clean.csv, deliveries_clean.csv |
| 2_feature_engineering.py | Pipeline | Feature engineering | utils | master_features.csv |
| 3_train_model.py | Pipeline | Model training | pandas, sklearn, xgboost, optuna | *.pkl models |
| 4_dashboard.py | UI | Main Streamlit app | live_match_tab, pitchmind_player_features, name_resolver | Web UI |
| 5_live_data_fetch.py | Support | Fetch live data | requests, json | todays_match.json |
| 6_player_features.py | Support | Player stats | name_resolver (opt) | player_*.csv |
| pitchmind_player_features.py | Support | Player stats (cached) | name_resolver (opt) | player_*.csv |
| 7_phase_predictor.py | Support | Phase models | pandas, xgboost | phase_*.pkl |
| utils.py | Utility | Team normalization | pandas | (functions only) |
| name_resolver.py | Utility | Name resolution | pandas, rapidfuzz (opt) | name_map.json |
| live_match_tab.py | Component | Live match UI | name_resolver | Streamlit widget |
| build_master_players.py | Unknown | Unknown | ? | ? |
| espn_live_scraper/ | Module | Alternative scraper | ? | ? |

---

## Execution Recommendations

### For Fresh Installation

**Order of Execution:**
```
1. 0_json_to_csv.py         (5 min: 1169 JSON files → 2 CSVs)
2. 1_data_cleaning.py        (2 min: clean data)
3. 2_feature_engineering.py  (10 min: engineer 63 features)
4. 3_train_model.py          (15 min: train models + calibration)
5-7. In parallel or sequential:
     - 5_live_data_fetch.py  (1 min: fetch live data, only 1hr before match)
     - 6_player_features.py  (3 min: compute player stats)
     - 7_phase_predictor.py  (5 min: train phase models)
8. streamlit run 4_dashboard.py (interactive dashboard)
```

**Total Time:** ~40 minutes (one-time setup)

### For Daily Match Day

```
1. 5_live_data_fetch.py      (1 min: ~1 hour before match)
   - Fetches toss, XI, weather
2. streamlit run 4_dashboard.py
   - Live score tracking + predictions
```

### For Development/Debugging

```
# Test individual pipeline steps:
python 2_feature_engineering.py  # Quick feature engineering test
python 6_player_features.py      # Player stats test
python name_resolver.py squad.md # Update name mappings

# Run dashboard in development:
streamlit run 4_dashboard.py --logger.level=debug
```

---

## Conclusion

**PitchMind** follows a **linear, hierarchical pipeline architecture** with:
- ✅ **Zero circular dependencies**
- ✅ **Clear data flow:** Raw JSON → Features → Models → Dashboard
- ✅ **Modular design:** Steps 5-7 can run independently after Step 1
- ✅ **Caching optimizations:** Player stats cached at module level
- ✅ **Graceful degradation:** SHAP and fuzzy matching are optional

**Strengths:**
- Easy to debug (linear flow)
- Parallelizable (branches 5-7)
- Production-ready (no circular imports)

**Potential Improvements:**
- Document `build_master_players.py` or remove if unused
- Consider consolidating `6_player_features.py` and `pitchmind_player_features.py`
- Add dependency injection for testing
