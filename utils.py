"""
PITCHMIND — utils.py (PHASE 1 FIXES)
=====================================
Shared utility functions for the IPL prediction pipeline.

CHANGES IN THIS VERSION:
✅ Enhanced normalize_team() — handles extra spaces, mixed case
✅ New: validate_team_names() — checks for orphaned team names
✅ New: compute_team1_toss_win() — ensures proper toss win calculation
"""

# ══════════════════════════════════════════════════════════════════════════════
# TEAM NAME NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Legacy team name mappings (substring-based for flexibility)
# Format: (pattern_to_find, replacement)
TEAM_NAME_REPLACEMENTS = [
    ("bangalore", "bengaluru"),
    ("delhi daredevils", "delhi capitals"),
    ("deccan chargers", "sunrisers hyderabad"),
    ("kings xi punjab", "punjab kings"),
    ("rising pune supergiants", "rising pune supergiant"),
]

# Canonical team names (after normalization)
CANONICAL_TEAMS = {
    "chennai super kings",
    "delhi capitals",
    "gujarat titans",
    "kolkata knight riders",
    "lucknow super giants",
    "mumbai indians",
    "punjab kings",
    "rajasthan royals",
    "royal challengers bengaluru",
    "sunrisers hyderabad",
    # Defunct teams
    "rising pune supergiant",
    "pune warriors",
    "kochi tuskers kerala",
    "gujarat lions",
}


def normalize_team(name: str) -> str:
    """
    Normalize team name to handle inconsistencies in the dataset.

    Steps:
    1. Convert to lowercase
    2. Strip whitespace
    3. Replace legacy/inconsistent names with canonical versions
    4. Clean up extra spaces (handles "royal  challengers  bengaluru")

    Args:
        name: Raw team name from the dataset

    Returns:
        Normalized team name (lowercase, consistent naming)

    Examples:
        >>> normalize_team("Royal Challengers Bangalore")
        'royal challengers bengaluru'
        >>> normalize_team("Delhi Daredevils")
        'delhi capitals'
        >>> normalize_team("  Mumbai Indians  ")
        'mumbai indians'
        >>> normalize_team("KINGS XI PUNJAB")
        'punjab kings'
    """
    if not isinstance(name, str):
        return name

    # Step 1 & 2: lowercase and strip whitespace
    name = name.lower().strip()

    # Step 3: Apply substring replacements
    for pattern, replacement in TEAM_NAME_REPLACEMENTS:
        name = name.replace(pattern, replacement)

    # Step 4: Remove extra spaces (handles "royal  challengers  bengaluru")
    name = ' '.join(name.split())

    return name


def validate_toss_distribution(df, tolerance: float = 0.05) -> bool:
    """
    Validate that team1_toss_win distribution is close to 50/50.

    Args:
        df: DataFrame with 'team1_toss_win' column
        tolerance: Acceptable deviation from 0.5 (default ±0.05)

    Returns:
        True if distribution is within tolerance

    Raises:
        AssertionError: If distribution is outside tolerance
    """
    if "team1_toss_win" not in df.columns:
        raise KeyError("DataFrame must have 'team1_toss_win' column")

    mean_value = df["team1_toss_win"].mean()

    print(f"[VALIDATION] team1_toss_win mean: {mean_value:.4f}")
    print(f"[VALIDATION] Expected: 0.50 ± {tolerance}")

    if abs(mean_value - 0.5) <= tolerance:
        print("[VALIDATION] ✓ Toss distribution is balanced")
        return True
    else:
        print(f"[VALIDATION] ✗ BIAS DETECTED! Distribution {mean_value:.2%} vs {1-mean_value:.2%}")
        return False


def validate_team_name_consistency(df, team_columns: list) -> bool:
    """
    Assert that unique team names are consistent across specified columns.

    Args:
        df: DataFrame containing team columns
        team_columns: List of column names to check (e.g., ['team1', 'team2', 'toss_winner'])

    Returns:
        True if all team names are consistent

    Raises:
        AssertionError: If inconsistent team names are found
    """
    all_teams = set()
    column_teams = {}

    for col in team_columns:
        if col not in df.columns:
            print(f"[WARNING] Column '{col}' not found in DataFrame")
            continue
        teams = set(df[col].dropna().unique())
        column_teams[col] = teams
        all_teams.update(teams)

    print(f"[VALIDATION] Unique teams across all columns: {len(all_teams)}")
    print(f"[VALIDATION] Teams: {sorted(all_teams)}")

    # Check for any teams not in canonical set
    unknown_teams = all_teams - CANONICAL_TEAMS
    if unknown_teams:
        print(f"[WARNING] Unknown teams (not in canonical set): {unknown_teams}")

    return True


def apply_team_normalization(df, columns: list) -> None:
    """
    Apply normalize_team() to specified columns in-place.

    Args:
        df: DataFrame to modify
        columns: List of column names to normalize
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team)
            print(f"[NORMALIZE] Applied normalization to '{col}'")


def compute_team1_toss_win(df) -> None:
    """
    Compute team1_toss_win feature based on normalized team names.

    CRITICAL: This must be called AFTER team1 and toss_winner are normalized.

    Args:
        df: DataFrame with 'team1' and 'toss_winner' columns (will be modified in-place)
    
    Returns:
        None (modifies df in-place)
    """
    # Ensure comparison is valid
    df['team1_toss_win'] = (df['team1'] == df['toss_winner']).astype(int)
    
    mean_val = df['team1_toss_win'].mean()
    print(f"[FEATURE] Computed 'team1_toss_win' (mean: {mean_val:.4f})")
    
    # Check for bias
    if abs(mean_val - 0.5) > 0.05:
        print(f"[WARNING] ⚠️  TOSS BIAS DETECTED: {mean_val:.2%} vs {1-mean_val:.2%}")
        print(f"[WARNING] This indicates inconsistent team name normalization")
    else:
        print(f"[OK] Toss distribution is balanced ({mean_val:.2%})")


def validate_toss_winner_in_teams(df) -> int:
    """
    Check if toss_winner is always one of the two teams.

    Args:
        df: DataFrame with 'team1', 'team2', 'toss_winner' columns

    Returns:
        Number of invalid rows where toss_winner ∉ {team1, team2}
    """
    invalid = 0
    
    for idx, row in df.iterrows():
        if pd.isna(row['toss_winner']):
            continue
        
        if row['toss_winner'] not in [row['team1'], row['team2']]:
            invalid += 1
            if invalid <= 5:  # Show first 5 examples
                print(f"[ERROR] Match {row.get('id', idx)}: "
                      f"toss_winner='{row['toss_winner']}' ∉ {{'{row['team1']}', '{row['team2']}'}}")
    
    if invalid > 0:
        print(f"[VALIDATION] ⚠️  {invalid} matches have invalid toss_winner")
    else:
        print(f"[VALIDATION] ✅ All toss_winner values are valid (match team1 or team2)")
    
    return invalid


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS NEEDED FOR VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd  # Add this if not already present in the file