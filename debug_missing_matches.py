"""
DIAGNOSTIC: MISSING MATCHES
=============================
Identifies which 23 matches are dropped between steps and why

Usage:
  python debug_missing_matches.py
"""

import pandas as pd
import os

DATA_DIR = "data"
MATCHES_RAW = os.path.join(DATA_DIR, "matches.csv")
MATCHES_CLEAN = os.path.join(DATA_DIR, "matches_clean.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "master_features.csv")
DELIVERIES_CLEAN = os.path.join(DATA_DIR, "deliveries_clean.csv")

def debug_missing_matches():
    """Identify which matches are dropped and where."""
    
    print("=" * 80)
    print("DIAGNOSTIC: MISSING MATCHES")
    print("=" * 80)
    
    # Load datasets
    try:
        matches_raw = pd.read_csv(MATCHES_RAW)
        print(f"✅ matches.csv: {len(matches_raw)} rows")
    except FileNotFoundError:
        print(f"❌ {MATCHES_RAW} not found")
        return
    
    try:
        matches_clean = pd.read_csv(MATCHES_CLEAN)
        print(f"✅ matches_clean.csv: {len(matches_clean)} rows")
    except FileNotFoundError:
        print(f"⚠️  {MATCHES_CLEAN} not found (checking features only)")
        matches_clean = None
    
    try:
        features = pd.read_csv(FEATURES_FILE)
        print(f"✅ master_features.csv: {len(features)} rows")
    except FileNotFoundError:
        print(f"⚠️  {FEATURES_FILE} not found")
        features = None
    
    try:
        deliveries = pd.read_csv(DELIVERIES_CLEAN)
        print(f"✅ deliveries_clean.csv: {len(deliveries)} rows")
    except FileNotFoundError:
        print(f"⚠️  {DELIVERIES_CLEAN} not found")
        deliveries = None
    
    print("\n" + "=" * 80)
    print("SECTION 1: MATCHES DROPPED IN STEP 1 (Cleaning)")
    print("=" * 80)
    
    if matches_clean is not None:
        dropped_step1 = set(matches_raw['id']) - set(matches_clean['id'])
        print(f"\nMatches dropped: {len(dropped_step1)}")
        
        if dropped_step1:
            print("\nDropped match IDs:")
            for mid in sorted(dropped_step1):
                match = matches_raw[matches_raw['id'] == mid].iloc[0]
                print(f"\nID: {mid}")
                print(f"  Date: {match['date']}")
                print(f"  team1: {match['team1']}")
                print(f"  team2: {match['team2']}")
                print(f"  toss_winner: {match['toss_winner']}")
                print(f"  winner: {match['winner']}")
                
                # Check for None/NaN values
                null_cols = [col for col in match.index if pd.isna(match[col])]
                if null_cols:
                    print(f"  ⚠️  NULL values: {null_cols}")
        else:
            print("✅ No matches dropped in step 1")
    
    print("\n" + "=" * 80)
    print("SECTION 2: MATCHES DROPPED IN STEP 2 (Feature Engineering)")
    print("=" * 80)
    
    if matches_clean is not None and features is not None:
        dropped_step2 = set(matches_clean['id']) - set(features['team1'])
        
        # Note: features might use team1/team2 as index instead of match_id
        # Try to count unique matches in features
        n_feat_matches = len(features)
        
        print(f"\nMatches in features: {n_feat_matches}")
        print(f"Matches in clean: {len(matches_clean)}")
        print(f"Difference: {len(matches_clean) - n_feat_matches}")
        
        if len(matches_clean) - n_feat_matches > 0:
            # Try to identify which ones
            if 'team1' in features.columns and 'team2' in features.columns:
                # Features don't have match_id, so we can't directly map
                print("\n⚠️  Cannot directly map features to matches (no match_id in features)")
                print("   Need to check feature engineering code for filtering logic")
            
            print("\nPossible reasons for drop:")
            print("  1. Matches with NaN in critical columns (team1, team2, toss_winner)")
            print("  2. Matches where toss_winner ≠ team1 AND ≠ team2")
            print("  3. Matches with no deliveries in deliveries.csv")
            print("  4. Feature engineering validation filters")
    else:
        print("❌ Cannot analyze step 2 (missing clean or feature files)")
    
    print("\n" + "=" * 80)
    print("SECTION 3: DELIVERIES ORPHANED MATCHES")
    print("=" * 80)
    
    if deliveries is not None and matches_clean is not None:
        deliveries_matches = set(deliveries['match_id'].unique())
        matches_clean_ids = set(matches_clean['id'].astype(str))
        
        print(f"\nUnique matches in deliveries: {len(deliveries_matches)}")
        print(f"Total matches in clean: {len(matches_clean)}")
        
        orphaned = deliveries_matches - matches_clean_ids
        if orphaned:
            print(f"\n⚠️  ORPHANED DELIVERIES: {len(orphaned)} matches have deliveries but are not in matches_clean")
            print(f"   Deliveries for these matches: {len(deliveries[deliveries['match_id'].isin(orphaned)])}")
            
            # Check if these are the 23 dropped matches
            if len(orphaned) == 23:
                print("   ✅ These are exactly the 23 dropped matches!")
        else:
            print("✅ No orphaned deliveries (all delivery matches are in clean matches)")
    
    print("\n" + "=" * 80)
    print("SECTION 4: DATA CONSISTENCY CHECKS")
    print("=" * 80)
    
    # Check toss_winner consistency
    print("\nToss winner validation:")
    if matches_clean is not None:
        valid_toss = matches_clean['toss_winner'].isin(
            set(matches_clean['team1']) | set(matches_clean['team2'])
        )
        invalid_toss = (~valid_toss).sum()
        print(f"  Toss winners that don't match team1 or team2: {invalid_toss}")
        
        if invalid_toss > 0:
            print("\n  Invalid toss winner entries:")
            for idx, row in matches_clean[~valid_toss].iterrows():
                print(f"    {row['id']}: toss_winner='{row['toss_winner']}' not in ['{row['team1']}', '{row['team2']}']")
    
    # Check for NULL values
    print("\nNULL value distribution:")
    if matches_clean is not None:
        null_counts = matches_clean.isnull().sum()
        critical_cols = ['id', 'date', 'team1', 'team2', 'toss_winner', 'winner']
        for col in critical_cols:
            if col in matches_clean.columns:
                count = null_counts[col]
                if count > 0:
                    print(f"  ⚠️  {col}: {count} NULLs")
                else:
                    print(f"  ✅ {col}: 0 NULLs")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    print("\nRECOMMENDATION:")
    print("1. Identify the 23 dropped matches (list above)")
    print("2. Determine reason: NULL values, inconsistent team names, or validation filters")
    print("3. Decision:")
    print("   - Option A: Fix the 23 matches and re-run pipeline")
    print("   - Option B: Document as invalid and exclude from analysis")
    print("4. Update data dictionary with reasoning")

if __name__ == "__main__":
    debug_missing_matches()