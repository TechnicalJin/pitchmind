"""
IPL PREDICTION MODEL — STEP 1: DATA CLEANING
=============================================
Cleans matches.csv and deliveries.csv for downstream feature engineering.

Run:
  python 1_data_cleaning.py
"""

import pandas as pd
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
MATCHES_FILE = os.path.join(DATA_DIR, "matches.csv")
DELIVERIES_FILE = os.path.join(DATA_DIR, "deliveries.csv")

# Team name replacements (old → current)
TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}


# ── MATCHES CLEANING ──────────────────────────────────────────────────────────
def clean_matches(filepath):
    """Load and clean the matches dataset."""
    matches = pd.read_csv(filepath)
    print(f"Matches raw shape: {matches.shape}")

    # Drop columns not useful for prediction
    matches = matches.drop(columns=["umpire1", "umpire2", "umpire3"], errors="ignore")

    # Handle missing values
    matches["winner"] = matches["winner"].fillna("No Result")
    matches["city"] = matches["city"].fillna("Unknown")
    matches["player_of_match"] = matches["player_of_match"].fillna("Unknown")

    # Standardize team names
    for col in ["team1", "team2", "toss_winner", "winner"]:
        matches[col] = matches[col].replace(TEAM_NAME_MAP)

    print(f"Matches clean shape: {matches.shape}")
    print("Matches dataset cleaned")
    return matches


# ── DELIVERIES CLEANING ───────────────────────────────────────────────────────
def clean_deliveries(filepath):
    """Load and clean the deliveries dataset."""
    deliveries = pd.read_csv(filepath)
    print(f"\nDeliveries raw shape: {deliveries.shape}")

    # Ensure column names are lowercase
    deliveries.columns = deliveries.columns.str.lower()

    # Remove rows with null match_id
    deliveries = deliveries.dropna(subset=["match_id"])

    # Convert columns to int
    deliveries["over"] = deliveries["over"].astype(int)
    deliveries["ball"] = deliveries["ball"].astype(int)
    deliveries["total_runs"] = deliveries["total_runs"].astype(int)
    deliveries["is_wicket"] = deliveries["is_wicket"].astype(int)

    print(f"Deliveries clean shape: {deliveries.shape}")
    print("Deliveries dataset cleaned")
    return deliveries


# ── SAVE CLEANED FILES ────────────────────────────────────────────────────────
def save_cleaned_data(matches, deliveries):
    """Save cleaned DataFrames to CSV."""
    out_matches = os.path.join(DATA_DIR, "matches_clean.csv")
    out_deliveries = os.path.join(DATA_DIR, "deliveries_clean.csv")

    matches.to_csv(out_matches, index=False)
    deliveries.to_csv(out_deliveries, index=False)

    print(f"\nSaved: {out_matches}")
    print(f"Saved: {out_deliveries}")
    print("Cleaned files saved successfully")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    matches = clean_matches(MATCHES_FILE)
    deliveries = clean_deliveries(DELIVERIES_FILE)
    save_cleaned_data(matches, deliveries)
