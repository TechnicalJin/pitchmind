"""
DIAGNOSTIC: TOSS WINNER BIAS
=============================
Identifies why team1_toss_win distribution is ~37% instead of ~50%

Usage:
  python diagnostic_toss_bias.py
"""

import pandas as pd
import os

DATA_DIR = "data"
MATCHES_FILE = os.path.join(DATA_DIR, "matches.csv")
MATCHES_CLEAN = os.path.join(DATA_DIR, "matches_clean.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "master_features.csv")

def diagnose_toss_bias():
    """Find the root cause of toss bias."""
    
    print("=" * 80)
    print("DIAGNOSTIC: TOSS WINNER BIAS")
    print("=" * 80)
    
    # Load all versions
    try:
        matches_raw = pd.read_csv(MATCHES_FILE)
        print(f"✅ Loaded matches.csv: {len(matches_raw)} rows")
    except FileNotFoundError:
        print(f"❌ File not found: {MATCHES_FILE}")
        return
    
    try:
        matches_clean = pd.read_csv(MATCHES_CLEAN)
        print(f"✅ Loaded matches_clean.csv: {len(matches_clean)} rows")
    except FileNotFoundError:
        print(f"❌ File not found: {MATCHES_CLEAN}")
        matches_clean = None
    
    try:
        features = pd.read_csv(FEATURES_FILE)
        print(f"✅ Loaded master_features.csv: {len(features)} rows")
    except FileNotFoundError:
        print(f"❌ File not found: {FEATURES_FILE}")
        features = None
    
    print("\n" + "=" * 80)
    print("SECTION 1: RAW MATCHES ANALYSIS")
    print("=" * 80)
    
    # Check raw team names
    print(f"\nTeam columns present: {[c for c in matches_raw.columns if 'team' in c.lower()]}")
    print(f"Unique teams in team1: {matches_raw['team1'].nunique()}")
    print(f"Unique teams in team2: {matches_raw['team2'].nunique()}")
    print(f"Unique teams in toss_winner: {matches_raw['toss_winner'].nunique()}")
    
    # Sample of raw data
    print("\nFirst 10 matches (raw):")
    print(matches_raw[['id', 'team1', 'team2', 'toss_winner']].head(10).to_string())
    
    # Check for case inconsistencies
    print("\n\nChecking for case inconsistencies in toss_winner:")
    toss_unique = matches_raw['toss_winner'].unique()
    print(f"Sample toss_winner values (first 10):")
    for i, t in enumerate(toss_unique[:10]):
        print(f"  {i+1}. '{t}'")
    
    # Manually compute toss win rate (raw)
    raw_toss_wins = (matches_raw['team1'] == matches_raw['toss_winner']).sum()
    raw_toss_rate = raw_toss_wins / len(matches_raw)
    print(f"\nRaw team1_toss_win rate: {raw_toss_wins}/{len(matches_raw)} = {raw_toss_rate:.4f}")
    
    if raw_toss_rate < 0.45:
        print(f"⚠️  BIAS DETECTED in raw data: {raw_toss_rate:.2%}")
    
    # Find mismatches
    print("\n\nFinding team1 vs toss_winner MISMATCHES:")
    mismatches = matches_raw[matches_raw['team1'] != matches_raw['toss_winner']]
    print(f"Mismatches: {len(mismatches)} out of {len(matches_raw)}")
    
    print("\nFirst 10 mismatches:")
    for idx, row in mismatches.head(10).iterrows():
        print(f"  {row['id']}: team1='{row['team1']}' != toss_winner='{row['toss_winner']}'")
    
    # Check for duplicate team names with different cases
    print("\n\nChecking for CASE SENSITIVITY issues:")
    team1_lower = matches_raw['team1'].str.lower().unique()
    toss_lower = matches_raw['toss_winner'].str.lower().unique()
    
    team1_not_in_toss = set(team1_lower) - set(toss_lower)
    toss_not_in_team1 = set(toss_lower) - set(team1_lower)
    
    if team1_not_in_toss:
        print(f"Teams in team1 but not in toss_winner (lowercase): {team1_not_in_toss}")
    if toss_not_in_team1:
        print(f"Teams in toss_winner but not in team1 (lowercase): {toss_not_in_team1}")
    
    if not team1_not_in_toss and not toss_not_in_team1:
        print("✅ No case sensitivity issues detected")
    
    if matches_clean is not None:
        print("\n" + "=" * 80)
        print("SECTION 2: CLEANED MATCHES ANALYSIS")
        print("=" * 80)
        
        # Check normalized team names
        print(f"\nAfter normalization:")
        print(f"Unique teams in team1: {matches_clean['team1'].nunique()}")
        print(f"Unique teams in team2: {matches_clean['team2'].nunique()}")
        print(f"Unique teams in toss_winner: {matches_clean['toss_winner'].nunique()}")
        
        print("\nAll unique teams in team1:")
        for t in sorted(matches_clean['team1'].unique()):
            print(f"  '{t}'")
        
        print("\nAll unique teams in toss_winner:")
        for t in sorted(matches_clean['toss_winner'].unique()):
            print(f"  '{t}'")
        
        # Compute toss win rate (clean)
        clean_toss_wins = (matches_clean['team1'] == matches_clean['toss_winner']).sum()
        clean_toss_rate = clean_toss_wins / len(matches_clean)
        print(f"\nCleaned team1_toss_win rate: {clean_toss_wins}/{len(matches_clean)} = {clean_toss_rate:.4f}")
        
        if clean_toss_rate < 0.45:
            print(f"⚠️  BIAS STILL PRESENT: {clean_toss_rate:.2%}")
        elif clean_toss_rate > 0.55:
            print(f"⚠️  BIAS IN OPPOSITE DIRECTION: {clean_toss_rate:.2%}")
        else:
            print(f"✅ BIAS RESOLVED: {clean_toss_rate:.2%}")
    
    if features is not None:
        print("\n" + "=" * 80)
        print("SECTION 3: FEATURES ANALYSIS")
        print("=" * 80)
        
        if 'team1_toss_win' in features.columns:
            feature_toss_mean = features['team1_toss_win'].mean()
            print(f"\nteam1_toss_win mean in master_features: {feature_toss_mean:.4f}")
            
            if feature_toss_mean < 0.45:
                print(f"❌ BIAS DETECTED: {feature_toss_mean:.2%} (expected ~50%)")
            elif feature_toss_mean > 0.55:
                print(f"⚠️  OPPOSITE BIAS: {feature_toss_mean:.2%}")
            else:
                print(f"✅ ACCEPTABLE: {feature_toss_mean:.2%}")
        else:
            print("❌ 'team1_toss_win' column not found in features")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    print("\nRECOMMENDED NEXT STEPS:")
    print("1. If bias detected in raw data → fix 0_json_to_csv.py normalization")
    print("2. If bias persists after cleaning → fix utils.py normalize_team()")
    print("3. If bias only in features → fix feature engineering team comparison")
    print("4. Run entire pipeline again after fixes")

if __name__ == "__main__":
    diagnose_toss_bias()