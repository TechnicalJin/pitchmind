# PitchMind — IPL Match Prediction System

> **An end-to-end machine learning pipeline that predicts IPL (Indian Premier League) cricket match winners using historical match and ball-by-ball delivery data, served through an interactive Streamlit dashboard.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Dataset Information](#4-dataset-information)
5. [Data Cleaning Pipeline](#5-data-cleaning-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [Machine Learning / Prediction Model](#7-machine-learning--prediction-model)
8. [Scripts Explanation](#8-scripts-explanation)
9. [Dependencies](#9-dependencies)
10. [Project Workflow](#10-project-workflow)
11. [Outputs](#11-outputs)
12. [How to Run the Project](#12-how-to-run-the-project)

---

## 1. Project Overview

### What This Project Does

PitchMind is a cricket analytics and prediction system that forecasts the winner of an IPL match **before it begins**. Users select two teams, a venue, and toss details, and the system returns a win probability for each team along with a detailed statistical comparison.

### Main Objective

Build a reliable, data-driven IPL match outcome predictor using historical data (2008–2020) with **zero data leakage** — every feature used for prediction is computed exclusively from matches that occurred *before* the match being predicted.

### Problem It Solves

Cricket match prediction is notoriously difficult due to the sport's inherent randomness. PitchMind addresses this by:

- Aggregating 33 engineered features across team strength, batting, bowling, venue, and toss dimensions.
- Using an ensemble of Random Forest and XGBoost classifiers to capture both linear and non-linear patterns.
- Providing an interactive dashboard so users can explore predictions and compare team statistics visually.

---

## 2. Tech Stack

| Category                | Technology                                         |
|-------------------------|----------------------------------------------------|
| **Programming Language** | Python 3.x                                        |
| **Data Processing**      | pandas, NumPy                                     |
| **Machine Learning**     | scikit-learn (RandomForestClassifier), XGBoost (XGBClassifier) |
| **Model Serialization**  | joblib                                            |
| **Visualization**        | Matplotlib, Streamlit (built-in charts)           |
| **Web Dashboard**        | Streamlit                                         |
| **Development Tools**    | Python venv (virtual environment), pip            |

---

## 3. Project Structure

```
pitchmind/
├── data/
│   ├── matches.csv              # Raw match-level IPL dataset (2008–2020)
│   ├── deliveries.csv           # Raw ball-by-ball delivery dataset
│   ├── matches_clean.csv        # Cleaned match data (output of Step 1)
│   ├── deliveries_clean.csv     # Cleaned delivery data (output of Step 1)
│   └── master_features.csv      # Engineered feature matrix (output of Step 2)
├── models/
│   ├── ipl_model.pkl            # Trained XGBoost model (primary)
│   ├── ipl_model_rf.pkl         # Trained Random Forest model (ensemble partner)
│   ├── feature_cols.pkl         # Ordered list of 33 feature column names
│   └── training_results.txt     # Accuracy summary from training
├── venv/                        # Python virtual environment (not tracked)
├── 1_data_cleaning.py           # Step 1: Cleans raw CSV files
├── 2_feature_engineering.py     # Step 2: Engineers 33 ML features
├── 3_train_model.py             # Step 3: Trains RF + XGBoost ensemble
├── 4_dashboard.py               # Step 4: Streamlit interactive dashboard
└── README.md                    # Project documentation (this file)
```

### File & Folder Descriptions

| Path | Purpose |
|------|---------|
| `data/` | Holds all raw and processed datasets. |
| `data/matches.csv` | Raw match metadata from Kaggle IPL dataset. |
| `data/deliveries.csv` | Raw ball-by-ball data from Kaggle IPL dataset. |
| `data/matches_clean.csv` | Cleaned matches — umpire columns dropped, missing values handled, team names standardized. |
| `data/deliveries_clean.csv` | Cleaned deliveries — nulls removed, dtypes enforced. |
| `data/master_features.csv` | Final ML-ready feature matrix (1090 rows × 40 columns). |
| `models/` | Stores trained model artifacts and metadata. |
| `models/ipl_model.pkl` | Serialized XGBoost classifier. |
| `models/ipl_model_rf.pkl` | Serialized Random Forest classifier. |
| `models/feature_cols.pkl` | Pickled list of feature column names (ensures column order consistency at inference). |
| `models/training_results.txt` | Plain-text accuracy report. |
| `1_data_cleaning.py` | Pipeline step 1 — data cleaning script. |
| `2_feature_engineering.py` | Pipeline step 2 — feature engineering script (v2, all leakage bugs fixed). |
| `3_train_model.py` | Pipeline step 3 — model training script. |
| `4_dashboard.py` | Pipeline step 4 — Streamlit prediction dashboard. |
| `venv/` | Python virtual environment directory. |

---

## 4. Dataset Information

### 4.1 `data/matches.csv` — Raw Match Data

| Property | Value |
|----------|-------|
| **Rows** | 1,095 |
| **Columns** | 20 |
| **Source** | Kaggle IPL Dataset (2008–2020) |
| **Type** | Raw data |

| Column | Dtype | Description |
|--------|-------|-------------|
| `id` | int64 | Unique match identifier |
| `season` | object | IPL season (e.g., "2007/08", "2019") |
| `city` | object | City where match was played (51 nulls) |
| `date` | object | Date of match (YYYY-MM-DD) |
| `match_type` | object | Type of match (League, Qualifier, Final, etc.) |
| `player_of_match` | object | Player of the Match awardee (5 nulls) |
| `venue` | object | Stadium name |
| `team1` | object | First team listed |
| `team2` | object | Second team listed |
| `toss_winner` | object | Team that won the toss |
| `toss_decision` | object | Toss decision — "bat" or "field" |
| `winner` | object | Match winner (5 nulls = no result) |
| `result` | object | Result type — "runs", "wickets", or "no result" |
| `result_margin` | float64 | Margin of victory (runs or wickets) |
| `target_runs` | float64 | Target set for chasing team |
| `target_overs` | float64 | Overs available for chasing team |
| `super_over` | object | Whether match went to super over ("Y"/"N") |
| `method` | object | D/L method applied (mostly null — 1,074 nulls) |
| `umpire1` | object | First on-field umpire |
| `umpire2` | object | Second on-field umpire |

### 4.2 `data/deliveries.csv` — Raw Ball-by-Ball Data

| Property | Value |
|----------|-------|
| **Rows** | 260,920 |
| **Columns** | 17 |
| **Source** | Kaggle IPL Dataset (2008–2020) |
| **Type** | Raw data |

| Column | Dtype | Description |
|--------|-------|-------------|
| `match_id` | int64 | Foreign key to `matches.id` |
| `inning` | int64 | Innings number (1, 2, or 3+ for super overs) |
| `batting_team` | object | Team batting |
| `bowling_team` | object | Team bowling |
| `over` | int64 | Over number (0-indexed) |
| `ball` | int64 | Ball number within the over |
| `batter` | object | Batsman on strike |
| `bowler` | object | Bowler |
| `non_striker` | object | Non-striker batsman |
| `batsman_runs` | int64 | Runs scored off the bat |
| `extra_runs` | int64 | Extra runs (wides, no-balls, byes, leg-byes) |
| `total_runs` | int64 | Total runs from that delivery |
| `extras_type` | object | Type of extra (mostly null — 246,795 nulls) |
| `is_wicket` | int64 | Whether a wicket fell (0 or 1) |
| `player_dismissed` | object | Player dismissed (null if no wicket) |
| `dismissal_kind` | object | Mode of dismissal (caught, bowled, lbw, etc.) |
| `fielder` | object | Fielder involved in dismissal |

### 4.3 `data/matches_clean.csv` — Cleaned Match Data

| Property | Value |
|----------|-------|
| **Rows** | 1,095 |
| **Columns** | 18 |
| **Source** | Output of `1_data_cleaning.py` |
| **Type** | Cleaned data |

**Preprocessing applied:**
- Dropped `umpire1`, `umpire2`, `umpire3` columns (not useful for prediction).
- Filled missing `winner` values with "No Result".
- Filled missing `city` values with "Unknown".
- Filled missing `player_of_match` values with "Unknown".
- Standardized team names (e.g., "Delhi Daredevils" → "Delhi Capitals", "Kings XI Punjab" → "Punjab Kings").

### 4.4 `data/deliveries_clean.csv` — Cleaned Delivery Data

| Property | Value |
|----------|-------|
| **Rows** | 260,920 |
| **Columns** | 17 |
| **Source** | Output of `1_data_cleaning.py` |
| **Type** | Cleaned data |

**Preprocessing applied:**
- Lowercased all column names.
- Dropped rows with null `match_id`.
- Cast `over`, `ball`, `total_runs`, `is_wicket` to int.

### 4.5 `data/master_features.csv` — Engineered Feature Matrix

| Property | Value |
|----------|-------|
| **Rows** | 1,090 |
| **Columns** | 40 (6 identifiers + 33 features + 1 target) |
| **Source** | Output of `2_feature_engineering.py` |
| **Type** | Processed / ML-ready data |
| **Null Values** | None |

5 matches were excluded (no result / incomplete data), reducing from 1,095 to 1,090 rows.

Columns: `match_id`, `date`, `season`, `team1`, `team2`, `venue`, 33 feature columns (see [Section 6](#6-feature-engineering)), and `target` (1 = team1 wins, 0 = team2 wins).

---

## 5. Data Cleaning Pipeline

### `1_data_cleaning.py`

| Step | Action | Details |
|------|--------|---------|
| 1 | **Load raw CSVs** | Reads `matches.csv` and `deliveries.csv` from `data/`. |
| 2 | **Drop umpire columns** | Removes `umpire1`, `umpire2`, `umpire3` (no predictive value). |
| 3 | **Handle missing values** | `winner` → "No Result", `city` → "Unknown", `player_of_match` → "Unknown". |
| 4 | **Standardize team names** | Maps legacy names to current franchise names using `TEAM_NAME_MAP`. |
| 5 | **Lowercase column names** | Deliveries column names converted to lowercase for consistency. |
| 6 | **Drop null match_ids** | Removes delivery rows without a valid `match_id`. |
| 7 | **Enforce dtypes** | Casts `over`, `ball`, `total_runs`, `is_wicket` to `int`. |
| 8 | **Save cleaned files** | Outputs `matches_clean.csv` and `deliveries_clean.csv`. |

### Team Name Standardization Map

| Old Name | New Name |
|----------|----------|
| Delhi Daredevils | Delhi Capitals |
| Kings XI Punjab | Punjab Kings |
| Rising Pune Supergiants | Rising Pune Supergiant |

The feature engineering script (`2_feature_engineering.py`) applies an **extended map** that also includes:

| Old Name | New Name |
|----------|----------|
| Deccan Chargers | Sunrisers Hyderabad |
| Royal Challengers Bangalore | Royal Challengers Bengaluru |

### Venue Name Standardization

| Old Name | New Name |
|----------|----------|
| Feroz Shah Kotla | Arun Jaitley Stadium |
| Wankhede Stadium, Mumbai | Wankhede Stadium |
| MA Chidambaram Stadium, Chepauk, Chennai | MA Chidambaram Stadium, Chepauk |
| Rajiv Gandhi International Stadium, Uppal | Rajiv Gandhi International Stadium |

---

## 6. Feature Engineering

### Overview

`2_feature_engineering.py` computes **33 features** organized into 5 categories. All features use **rolling historical averages** with `shift(1)` to prevent data leakage — only data from matches *before* the current match is used.

### Feature Catalog

#### Team Strength Features (7)

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `team1_win_rate` | Expanding mean of wins for team1 across all prior matches (shift(1)) | Overall historical strength of team1 |
| `team2_win_rate` | Same as above for team2 | Overall historical strength of team2 |
| `team1_recent_form` | Rolling mean of wins over last 5 matches (shift(1)) | Short-term momentum |
| `team2_recent_form` | Same as above for team2 | Short-term momentum |
| `h2h_win_rate` | Cumulative win rate of team1 vs team2 in all prior head-to-head matches | Historical dominance between the two specific teams |
| `team1_nrr` | Rolling `(avg run rate scored) − (avg run rate conceded)` from deliveries (shift(1)) | Net Run Rate — overall performance quality |
| `team2_nrr` | Same as above for team2 | Net Run Rate |

#### Batting Features (12)

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `team1_avg_runs` | Rolling historical mean of total runs scored per innings (shift(1)) | Batting strength baseline |
| `team2_avg_runs` | Same for team2 | Batting strength baseline |
| `team1_powerplay_runs` | Rolling avg runs scored in overs 0–5 (shift(1)) | Ability to score in the powerplay |
| `team2_powerplay_runs` | Same for team2 | Powerplay scoring |
| `team1_death_runs` | Rolling avg runs scored in overs 15–19 (shift(1)) | Death-over finishing ability |
| `team2_death_runs` | Same for team2 | Death-over finishing |
| `team1_boundary_pct` | Rolling avg fraction of deliveries hit for 4 or 6 (shift(1)) | Aggressive batting tendency |
| `team2_boundary_pct` | Same for team2 | Aggressive batting |
| `team1_run_rate` | Rolling avg run rate (runs/overs) (shift(1)) | Scoring tempo |
| `team2_run_rate` | Same for team2 | Scoring tempo |
| `team1_top3_sr` | Rolling avg strike rate of the top-3 batting order players (shift(1)) | Quality of the top order |
| `team2_top3_sr` | Same for team2 | Top-order quality |

#### Bowling Features (8)

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `team1_bowling_economy` | Rolling avg economy rate conceded while bowling (shift(1)) | Overall bowling restrictiveness |
| `team2_bowling_economy` | Same for team2 | Bowling economy |
| `team1_death_economy` | Rolling avg economy in overs 15–19 while bowling (shift(1)) | Death bowling discipline |
| `team2_death_economy` | Same for team2 | Death bowling |
| `team1_pp_wickets` | Rolling avg wickets taken in powerplay overs 0–5 (shift(1)) | Ability to take early wickets |
| `team2_pp_wickets` | Same for team2 | Powerplay wicket-taking |
| `team1_bowling_sr` | Rolling avg bowling strike rate (balls per wicket) (shift(1)) | Overall wicket-taking frequency |
| `team2_bowling_sr` | Same for team2 | Bowling strike rate |

#### Venue Features (3)

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `team1_venue_win_rate` | Rolling win rate of team1 at this specific venue (shift(1)), falling back to overall win rate | Home/venue advantage |
| `team2_venue_win_rate` | Same for team2 | Venue advantage |
| `venue_avg_runs` | Rolling avg first-innings total at this venue (shift(1)) | Venue scoring conditions (high vs low scoring ground) |

#### Context Features (3)

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `toss_win` | 1 if team1 won the toss, else 0 | Toss advantage signal |
| `toss_field` | 1 if the toss winner chose to field, else 0 | Chasing preference signal |
| `toss_team1_field` | 1 if team1 won toss AND chose to field, else 0 | Combined toss-field advantage |

### Data Leakage Fixes (v2)

The script documents and fixes **7 data leakage bugs** from a prior version:

| Bug | Issue | Fix |
|-----|-------|-----|
| BUG 1 | `team1_avg_runs` used current match score | Rolling historical average with shift(1) |
| BUG 2 | Multiple batting/bowling stats from current match | All converted to rolling historical averages |
| BUG 3 | team1 vs team2 avg_runs had ~7 run asymmetry | Historical avg regardless of innings |
| BUG 4 | NRR used `target_runs` column incorrectly | Computed from deliveries as run_rate_scored − run_rate_conceded |
| BUG 5 | `is_day_night` hardcoded to 1 for all matches | Removed (zero predictive value) |
| BUG 6 | `h2h_win_rate` used all data including future matches | Rolling cumulative H2H before each match date |
| BUG 7 | `team_win_rate` and `venue_win_rate` used global data | Rolling cumulative up to each match date |

---

## 7. Machine Learning / Prediction Model

### Models Used

| Model | Library | Role |
|-------|---------|------|
| **Random Forest** | scikit-learn `RandomForestClassifier` | Ensemble member |
| **XGBoost** | xgboost `XGBClassifier` | Ensemble member (primary saved model) |
| **Ensemble** | Manual probability averaging | Final prediction (average of RF + XGBoost probabilities) |

### Target Variable

- **`target`**: Binary (1 = team1 wins, 0 = team2 wins).

### Input Features

33 features (see [Section 6](#6-feature-engineering)). Identifier columns (`match_id`, `date`, `team1`, `team2`, `venue`, `season`, `target`) are excluded from training.

### Training Process

1. Load `master_features.csv`.
2. Separate features (33 columns) from target.
3. Fill any remaining NaN values with column medians.
4. Split into 80% train / 20% test (`random_state=42`).
5. Train Random Forest (300 trees, max_depth=10).
6. Train XGBoost (400 estimators, max_depth=6, learning_rate=0.05, subsample=0.9).
7. Ensemble: Average predicted probabilities from both models, threshold at 0.5.

### Hyperparameters

| Parameter | Random Forest | XGBoost |
|-----------|--------------|---------|
| `n_estimators` | 300 | 400 |
| `max_depth` | 10 | 6 |
| `learning_rate` | N/A | 0.05 |
| `subsample` | N/A | 0.9 |
| `colsample_bytree` | N/A | 0.9 |
| `random_state` | 42 | 42 |
| `n_jobs` | -1 | N/A |

### Evaluation Metrics

- **Accuracy** (primary metric)
- **Classification Report** (precision, recall, F1-score for both classes)
- **Feature Importance** (top 10 from Random Forest)

### Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | 55.05% |
| XGBoost | 53.21% |
| **Ensemble (RF + XGBoost)** | **55.96%** |
| Features Used | 33 |

> **Note:** ~56% accuracy is reasonable for cricket match prediction — the sport is inherently unpredictable. A 50% baseline (coin flip) is the lower bound. The model is designed to capture subtle statistical edges rather than guarantee outcomes.

---

## 8. Scripts Explanation

### `1_data_cleaning.py`

| Property | Details |
|----------|---------|
| **Purpose** | Cleans raw match and delivery datasets for downstream processing. |
| **Input Files** | `data/matches.csv`, `data/deliveries.csv` |
| **Output Files** | `data/matches_clean.csv`, `data/deliveries_clean.csv` |
| **Main Functions** | `clean_matches(filepath)` — Drops umpire columns, handles missing values, standardizes team names. |
| | `clean_deliveries(filepath)` — Lowercases columns, drops null match_ids, enforces integer dtypes. |
| | `save_cleaned_data(matches, deliveries)` — Writes cleaned DataFrames to CSV. |

### `2_feature_engineering.py`

| Property | Details |
|----------|---------|
| **Purpose** | Engineers 33 ML features from raw data using rolling historical statistics with no data leakage. |
| **Input Files** | `data/matches.csv`, `data/deliveries.csv` (reads raw data directly with its own cleaning) |
| **Output Files** | `data/master_features.csv` |
| **Main Functions** | `load_data()` — Loads and cleans raw datasets, standardizes team/venue names. |
| | `compute_rolling_win_rate(matches)` — Expanding cumulative win rate per team (shift(1)). |
| | `compute_rolling_recent_form(matches, window=5)` — Rolling 5-match win rate (shift(1)). |
| | `compute_rolling_h2h(matches)` — Rolling head-to-head win rate between two teams. |
| | `compute_rolling_nrr(matches, deliveries)` — Net Run Rate from run rate scored minus conceded. |
| | `compute_rolling_venue_win_rate(matches)` — Per-team per-venue rolling win rate. |
| | `compute_rolling_venue_avg_runs(deliveries, matches)` — Per-venue rolling first-innings average. |
| | `compute_rolling_delivery_features(deliveries, matches)` — All batting/bowling stats (12+8 features). |
| | `build_master_features(matches, deliveries)` — Assembles all features into one DataFrame. |
| | `validate_and_print(master_df)` — Prints summary stats, null checks, and leakage validation. |

### `3_train_model.py`

| Property | Details |
|----------|---------|
| **Purpose** | Trains Random Forest + XGBoost models and saves artifacts. |
| **Input Files** | `data/master_features.csv` |
| **Output Files** | `models/ipl_model.pkl`, `models/ipl_model_rf.pkl`, `models/feature_cols.pkl`, `models/training_results.txt` |
| **Main Functions** | `load_data()` — Reads master_features.csv. |
| | `prepare_data(df)` — Separates feature columns from target, fills NaNs with medians. |
| | `split_data(X, y)` — 80/20 train-test split (random_state=42). |
| | `train_random_forest(...)` — Trains RF (300 trees, max_depth=10). |
| | `train_xgboost(...)` — Trains XGBoost (400 estimators, LR=0.05). |
| | `train_ensemble(rf, xgb, ...)` — Averages probabilities from both models. |
| | `print_feature_importance(rf, feature_cols)` — Shows top 10 RF importances. |
| | `print_report(y_test, final_pred)` — Prints classification report. |
| | `save_artifacts(...)` — Saves models, feature list, and results to `models/`. |

### `4_dashboard.py`

| Property | Details |
|----------|---------|
| **Purpose** | Interactive Streamlit web dashboard for match prediction and team comparison. |
| **Input Files** | `data/master_features.csv`, `models/ipl_model.pkl`, `models/ipl_model_rf.pkl`, `models/feature_cols.pkl` |
| **Output Files** | None (real-time web UI) |
| **Main Functions** | `load_data()` — Cached loading of master feature dataset. |
| | `load_models()` — Cached loading of XGBoost, RF models, and feature column list. |
| | `get_team_stats(team, role, venue, df)` — Retrieves latest historical stats for a team. |
| | `build_feature_vector(...)` — Constructs a 1-row DataFrame matching the trained model's feature schema. |
| | `predict_winner(X_input)` — Returns ensemble win probability (averaged RF + XGBoost probabilities). |

**Dashboard Sections:**
- **Sidebar** — Team selection (Team 1, Team 2), venue, toss winner, toss decision, and a "Predict Winner" button.
- **Win Probability** — Large probability display with visual bar chart.
- **Team Comparison** — Table comparing 11 metrics (win rate, form, runs, economy, etc.).
- **Key Match Factors** — Toss impact, head-to-head record, and win rate analysis.
- **Venue Analysis** — Average score at venue and per-team venue win rates.
- **Batting vs Bowling Comparison** — Side-by-side bar charts of batting and bowling stats.
- **About This Model** — Expandable section with model details and limitations.

---

## 9. Dependencies

### Required Python Packages

| Package | Version (tested) | Used For |
|---------|-----------------|----------|
| `pandas` | 2.3.3 | Data loading, manipulation, and feature computation |
| `numpy` | 2.1.1 | Numerical operations |
| `scikit-learn` | 1.8.0 | Random Forest classifier, train-test split, metrics |
| `xgboost` | 3.2.0 | XGBoost gradient boosting classifier |
| `joblib` | 1.5.1 | Model serialization (save/load `.pkl` files) |
| `matplotlib` | 3.9.2 | Bar chart visualizations in the dashboard |
| `streamlit` | 1.55.0 | Interactive web dashboard framework |

### Python Standard Library Imports

`os`, `warnings`

---

## 10. Project Workflow

```
┌─────────────────────┐
│   Raw Data          │
│   matches.csv       │
│   deliveries.csv    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌───────────────────────────────┐
│ Step 1: Data Clean  │────▶│ matches_clean.csv             │
│ 1_data_cleaning.py  │     │ deliveries_clean.csv          │
└────────┬────────────┘     └───────────────────────────────┘
         │
         ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│ Step 2: Feature Eng.     │────▶│ master_features.csv      │
│ 2_feature_engineering.py │     │ (1090 rows × 40 cols)    │
└────────┬─────────────────┘     └──────────────────────────┘
         │
         ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│ Step 3: Model Training   │────▶│ ipl_model.pkl (XGBoost)  │
│ 3_train_model.py         │     │ ipl_model_rf.pkl (RF)    │
│                          │     │ feature_cols.pkl          │
│                          │     │ training_results.txt      │
└────────┬─────────────────┘     └──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Step 4: Dashboard        │
│ 4_dashboard.py           │──────▶  Interactive Web UI
│ (Streamlit)              │         (localhost:8501)
└──────────────────────────┘
```

**Pipeline Summary:**

```
Raw Data → Data Cleaning → Feature Engineering → Model Training → Interactive Dashboard
```

---

## 11. Outputs

### Model Files

| File | Format | Description |
|------|--------|-------------|
| `models/ipl_model.pkl` | Pickle (joblib) | Trained XGBoost classifier — primary model |
| `models/ipl_model_rf.pkl` | Pickle (joblib) | Trained Random Forest classifier — ensemble partner |
| `models/feature_cols.pkl` | Pickle (joblib) | Ordered list of 33 feature column names |

### Data Files

| File | Format | Description |
|------|--------|-------------|
| `data/matches_clean.csv` | CSV | Cleaned match data (1,095 rows × 18 columns) |
| `data/deliveries_clean.csv` | CSV | Cleaned delivery data (260,920 rows × 17 columns) |
| `data/master_features.csv` | CSV | Engineered feature matrix (1,090 rows × 40 columns) |

### Reports

| File | Format | Description |
|------|--------|-------------|
| `models/training_results.txt` | Text | Model accuracy summary (RF: 55.05%, XGBoost: 53.21%, Ensemble: 55.96%) |

### Dashboard

| Output | Type | Description |
|--------|------|-------------|
| Streamlit Web App | Real-time UI | Win probability display, team comparison table, bar charts (batting/bowling), venue analysis, key match factors |

---

## 12. How to Run the Project

### Prerequisites

- Python 3.9+ installed
- pip package manager

### Step 1: Clone / Open the Project

```bash
cd pitchmind
```

### Step 2: Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib streamlit
```

### Step 4: Run the Pipeline

Run scripts in order — each step depends on the output of the previous step:

```bash
# Step 1: Clean raw data
python 1_data_cleaning.py

# Step 2: Engineer features
python 2_feature_engineering.py

# Step 3: Train models
python 3_train_model.py

# Step 4: Launch dashboard
streamlit run 4_dashboard.py
```

The dashboard will open in your browser at **http://localhost:8501**.

### Quick Start (if models already exist)

If the `models/` directory already contains trained models, you can skip steps 1–3 and launch the dashboard directly:

```bash
streamlit run 4_dashboard.py
```

---

## License

This project is for **educational use only**. The IPL dataset is sourced from publicly available Kaggle datasets.
