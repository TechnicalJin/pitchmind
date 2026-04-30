"""
IPL PREDICTION MODEL — STEP 1: DATA CLEANING  (v3 — 2023+ Era Only)
=====================================================================
KEY CHANGE vs v2:
  HARD CUTOFF: Only 2023+ data is used.
  Reason: Post-2022 IPL is the true modern era.
    - Impact Player rule (introduced 2022) now fully embedded by 2023
    - 190+ scores are the new normal (not just 175)
    - Death over hitting completely transformed (strike rates 170+)
    - Powerplay strategies changed drastically — aggressive openers standard
    - Average first-innings score jumped ~15–20 runs vs 2019–2022
    - Bowlers concede more; economy standards recalibrated
    - Training on 2019–2022 DILUTES modern patterns

  RESULT: Model learns only from the current IPL meta (2023, 2024, 2025).

Run:
  python 1_data_cleaning.py

Output:
  data/matches_clean.csv      ← 2023+ matches only
  data/deliveries_clean.csv   ← deliveries for 2023+ matches only
"""

import pandas as pd
import numpy as np
import os
from utils import normalize_team, apply_team_normalization

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
MATCHES_FILE    = os.path.join(DATA_DIR, "matches.csv")
DELIVERIES_FILE = os.path.join(DATA_DIR, "deliveries.csv")

# ── CRITICAL: Only use data from this season onwards ─────────────────────────
# Pre-2023 IPL = different game entirely.
# 2019-2022: avg score ~165-172, Impact Player rule not fully embedded,
# different death-over tactics, bowlers had more control.
# 2023+: true modern IPL — 185-195 avg, Impact Player fully in play,
# aggressive batting at every phase, completely different team compositions.
CUTOFF_SEASON = 2023


# ══════════════════════════════════════════════════════════════════════════════
# MATCHES CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def clean_matches(filepath):
    matches = pd.read_csv(filepath)
    print(f"Matches raw shape   : {matches.shape}")

    # Drop unused columns
    matches = matches.drop(columns=["umpire1", "umpire2", "umpire3"], errors="ignore")

    # Fill missing values
    matches["winner"]          = matches["winner"].fillna("No Result")
    matches["city"]            = matches["city"].fillna("Unknown")
    matches["player_of_match"] = matches["player_of_match"].fillna("Unknown")

    # Standardize team names
    apply_team_normalization(matches, ["team1", "team2", "toss_winner", "winner"])

    # ── APPLY CUTOFF ─────────────────────────────────────────────────────────
    matches["season_int"] = matches["season"].astype(str).str[:4].astype(int)
    before = len(matches)
    matches = matches[matches["season_int"] >= CUTOFF_SEASON].copy()
    after   = len(matches)

    print(f"\n  ✂️  Hard cutoff: season >= {CUTOFF_SEASON}")
    print(f"  Removed {before - after} pre-{CUTOFF_SEASON} matches ({before} → {after})")
    print(f"  Seasons kept: {sorted(matches['season_int'].unique())}")
    print(f"\nMatches clean shape : {matches.shape}")
    return matches


# ══════════════════════════════════════════════════════════════════════════════
# DELIVERIES CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def clean_deliveries(filepath, valid_match_ids):
    deliveries = pd.read_csv(filepath, low_memory=False)
    print(f"\nDeliveries raw shape: {deliveries.shape}")

    # Lowercase column names
    deliveries.columns = deliveries.columns.str.lower()

    # Drop rows with null match_id
    deliveries = deliveries.dropna(subset=["match_id"])

    # ── FILTER to only 2023+ match IDs ───────────────────────────────────────
    deliveries["match_id"] = deliveries["match_id"].astype(str)
    valid_ids_str = set(str(m) for m in valid_match_ids)
    before = len(deliveries)
    deliveries = deliveries[deliveries["match_id"].isin(valid_ids_str)].copy()
    after = len(deliveries)
    print(f"  Filtered deliveries to 2023+ matches ({before:,} → {after:,} rows)")

    # Type coercions
    for col in ["over", "ball", "total_runs", "is_wicket"]:
        if col in deliveries.columns:
            deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce").fillna(0).astype(int)

    print(f"Deliveries clean shape: {deliveries.shape}")
    return deliveries


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
def save_cleaned_data(matches, deliveries):
    out_matches    = os.path.join(DATA_DIR, "matches_clean.csv")
    out_deliveries = os.path.join(DATA_DIR, "deliveries_clean.csv")

    matches.to_csv(out_matches, index=False)
    deliveries.to_csv(out_deliveries, index=False)

    print(f"\n✅ Saved: {out_matches}    ({len(matches):,} matches)")
    print(f"✅ Saved: {out_deliveries}  ({len(deliveries):,} deliveries)")

    # Print season breakdown
    print("\n  Matches per season:")
    for season, grp in matches.groupby("season_int"):
        print(f"    {season}: {len(grp)} matches")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(f"  PITCHMIND — DATA CLEANING v3 (2023+ only)")
    print(f"  Cutoff: {CUTOFF_SEASON}  |  Pre-{CUTOFF_SEASON} data excluded")
    print("=" * 60 + "\n")

    matches    = clean_matches(MATCHES_FILE)
    valid_ids  = set(matches["id"].astype(str).tolist())
    deliveries = clean_deliveries(DELIVERIES_FILE, valid_ids)
    save_cleaned_data(matches, deliveries)

    print("\n" + "=" * 60)
    print("  DATA CLEANING COMPLETE")
    print("=" * 60)
    print("\n  WHY 2023+ ONLY:")
    print("  ❌ 2019-2022: avg score ~165-172, Impact Player not yet normalised")
    print("  ✅ 2023+    : Impact Player fully embedded, 185-195 avg is new normal")
    print("  ✅ 2024     : 190+ consistently, aggressive openers, new death tactics")
    print("  ✅ 2025     : Most recent data — gold standard for current predictions")
    print("\n  Next step → python 2_feature_engineering.py")