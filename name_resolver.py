"""
PITCHMIND — PLAYER NAME RESOLUTION SYSTEM
==========================================
Resolves player name mismatches between:
- Squad data (full names): "Jasprit Bumrah", "Virat Kohli"
- Cricsheet data (initials): "JJ Bumrah", "V Kohli"

Usage:
    from name_resolver import resolve_name, get_cricsheet_name, build_name_mapping

    # Resolve a squad name to cricsheet name
    cricsheet_name = resolve_name("Jasprit Bumrah")  # -> "JJ Bumrah"

    # Or use the lookup directly
    cricsheet_name = get_cricsheet_name("Virat Kohli")  # -> "V Kohli"
"""

import os
import json
import re
import pandas as pd
from typing import Optional, Dict, Tuple, Set, List

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = "data"
NAME_MAP_PATH = os.path.join(DATA_DIR, "name_map.json")
MASTER_PLAYERS_PATH = os.path.join(DATA_DIR, "master_players.csv")

# Fuzzy matching threshold (0-100)
FUZZY_THRESHOLD = 85

# ══════════════════════════════════════════════════════════════════════════════
# MANUAL NAME MAPPING (CRITICAL — known initials to full names)
# ══════════════════════════════════════════════════════════════════════════════

# Format: "normalized_cricsheet_name" -> "normalized_full_name"
# This handles the JJ Bumrah -> Jasprit Bumrah type conversions
MANUAL_CRICSHEET_TO_FULL = {
    # Top Stars - Mumbai Indians
    "jj bumrah": "jasprit bumrah",
    "rg sharma": "rohit sharma",
    "hh pandya": "hardik pandya",
    "sk yadav": "suryakumar yadav",
    "sa yadav": "suryakumar yadav",
    "ty david": "tim david",
    "qj de kock": "quinton de kock",
    "q de kock": "quinton de kock",

    # CSK
    "ms dhoni": "ms dhoni",  # exact match preserved
    "n jagadeesan": "narayan jagadeesan",
    "ra jadeja": "ravindra jadeja",
    "dj bravo": "dwayne bravo",
    "sm curran": "sam curran",
    "dp conway": "devon conway",
    "rn gaikwad": "ruturaj gaikwad",
    "sd dube": "shivam dube",
    "m theekshana": "maheesh theekshana",
    "ts deshpande": "tushar deshpande",

    # RCB
    "v kohli": "virat kohli",
    "kh pandya": "krunal pandya",
    "fa allen": "finn allen",
    "rjw topley": "reece topley",
    "mw short": "matthew short",
    "hv patel": "harshal patel",
    "mk lomror": "mahipal lomror",
    "ss iyer": "shreyas iyer",
    "kd karthik": "dinesh karthik",
    "pw hasaranga": "wanindu hasaranga",
    "jr hazlewood": "josh hazlewood",
    "js sidhu": "jitesh sharma",

    # KKR
    "am rahane": "ajinkya rahane",
    "sp narine": "sunil narine",
    "ad russell": "andre russell",
    "jj roy": "jason roy",
    "ss iyer": "shreyas iyer",
    "vr iyer": "venkatesh iyer",
    "n rana": "nitish rana",
    "r singh": "rinku singh",
    "t seifert": "tim seifert",
    "lh ferguson": "lockie ferguson",
    "um malik": "umran malik",
    "cv varun": "varun chakravarthy",
    "m pathirana": "matheesha pathirana",

    # SRH
    "tw head": "travis head",
    "hc brook": "harry brook",
    "ag patel": "axar patel",
    "b cummins": "pat cummins",
    "pj cummins": "pat cummins",
    "t natarajan": "t natarajan",
    "nk reddy": "nitish kumar reddy",
    "h klaasen": "heinrich klaasen",
    "i kishan": "ishan kishan",

    # DC
    "kl rahul": "kl rahul",
    "da warner": "david warner",
    "mr marsh": "mitchell marsh",
    "pp shaw": "prithvi shaw",
    "ar patel": "axar patel",
    "km jadhav": "kedar jadhav",
    "kk nair": "karun nair",
    "m starc": "mitchell starc",
    "k yadav": "kuldeep yadav",
    "l ngidi": "lungi ngidi",

    # PBKS
    "ls livingstone": "liam livingstone",
    "sm hardie": "sam hardie",
    "jm sharma": "jitesh sharma",
    "arshdeep singh": "arshdeep singh",
    "k rabada": "kagiso rabada",
    "yj chahal": "yuzvendra chahal",
    "yu chahal": "yuzvendra chahal",
    "m jansen": "marco jansen",
    "m stoinis": "marcus stoinis",

    # GT
    "ss gill": "shubman gill",
    "wc saha": "wriddhiman saha",
    "r tewatia": "rahul tewatia",
    "r khan": "rashid khan",
    "da miller": "david miller",
    "jc buttler": "jos buttler",
    "ws sundar": "washington sundar",
    "mj siraj": "mohammed siraj",
    "pk krishna": "prasidh krishna",
    "i sharma": "ishant sharma",

    # LSG
    "r pant": "rishabh pant",
    "kl rahul": "kl rahul",
    "n pooran": "nicholas pooran",
    "a markram": "aiden markram",
    "ma wood": "mark wood",
    "a nortje": "anrich nortje",
    "m shami": "mohammed shami",
    "avesh khan": "avesh khan",
    "mohsin khan": "mohsin khan",
    "mp stoinis": "marcus stoinis",
    "k gowtham": "krishnappa gowtham",

    # RR
    "yb jaiswal": "yashasvi jaiswal",
    "sv samson": "sanju samson",
    "jc archer": "jofra archer",
    "jr buttler": "jos buttler",
    "r parag": "riyan parag",
    "s hetmyer": "shimron hetmyer",
    "r bishnoi": "ravi bishnoi",
    "s sharma": "sandeep sharma",
    "d jurel": "dhruv jurel",

    # Common international players
    "ab de villiers": "ab de villiers",
    "ch gayle": "chris gayle",
    "kc sangakkara": "kumar sangakkara",
    "dpmd jayawardene": "mahela jayawardene",
    "kp pietersen": "kevin pietersen",
    "ac gilchrist": "adam gilchrist",
    "bb mccullum": "brendon mccullum",
    "sr watson": "shane watson",
    "mj guptill": "martin guptill",
    "tm dilshan": "tillakaratne dilshan",
    "se marsh": "shaun marsh",
    "jp faulkner": "james faulkner",
    "nj reardon": "nathan reardon",
    "gr maxwell": "glenn maxwell",
    "aj finch": "aaron finch",
    "mp stoinis": "marcus stoinis",
    "dj malan": "dawid malan",
    "t banton": "tom banton",
    "jm bairstow": "jonny bairstow",
    "je root": "joe root",
    "ba stokes": "ben stokes",
    "jc buttler": "jos buttler",
    "mm ali": "moeen ali",
    "tc bruce": "tom bruce",
    "gh phillips": "glenn phillips",
    "tm southee": "tim southee",
    "ta boult": "trent boult",
    "mj santner": "mitchell santner",
    "is sodhi": "ish sodhi",
    "fh edwards": "fidel edwards",
    "dj sammy": "darren sammy",

    # Additional common players
    "kr sharma": "kieron sharma",
    "a badoni": "ayush badoni",
    "b sai sudharsan": "sai sudharsan",
    "at rayudu": "ambati rayudu",
    "ka pollard": "kieron pollard",
    "lmp simmons": "lendl simmons",
}

# Reverse mapping: full_name -> cricsheet_name
FULL_TO_CRICSHEET = {v: k for k, v in MANUAL_CRICSHEET_TO_FULL.items()}

# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    """
    Normalize a player name for consistent comparison.

    Steps:
    1. Convert to lowercase
    2. Strip whitespace
    3. Remove dots and hyphens
    4. Collapse multiple spaces

    Args:
        name: Raw player name

    Returns:
        Normalized name string

    Examples:
        >>> normalize_name("JJ Bumrah")
        'jj bumrah'
        >>> normalize_name("M.S. Dhoni")
        'ms dhoni'
        >>> normalize_name("de Kock")
        'de kock'
    """
    if not isinstance(name, str):
        return str(name).lower().strip() if name is not None else ""

    name = name.lower().strip()
    name = name.replace(".", "")
    name = name.replace("-", " ")
    name = name.replace("'", "")
    name = " ".join(name.split())  # collapse multiple spaces

    return name


def extract_surname(name: str) -> str:
    """Extract surname (last word) from a name."""
    parts = normalize_name(name).split()
    return parts[-1] if parts else ""


def extract_initials(name: str) -> str:
    """Extract initials from first/middle names."""
    parts = normalize_name(name).split()
    if len(parts) <= 1:
        return ""
    return "".join([p[0] for p in parts[:-1]])

# ══════════════════════════════════════════════════════════════════════════════
# NAME MAPPING CACHE
# ══════════════════════════════════════════════════════════════════════════════

_name_map_cache: Dict[str, Optional[str]] = {}
_cricsheet_players: Set[str] = set()
_master_players: Set[str] = set()


def load_cricsheet_players() -> Set[str]:
    """Load all unique player names from cricsheet data (deliveries.csv)."""
    global _cricsheet_players

    if _cricsheet_players:
        return _cricsheet_players

    for path in ["data/deliveries_clean.csv", "data/deliveries.csv"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False, usecols=["batter", "bowler"])
                batters = set(df["batter"].dropna().unique())
                bowlers = set(df["bowler"].dropna().unique())
                _cricsheet_players = batters | bowlers
                return _cricsheet_players
            except Exception as e:
                print(f"[WARNING] Could not load cricsheet players: {e}")

    return set()


def load_name_map() -> Dict[str, Optional[str]]:
    """Load name mapping from JSON file if it exists."""
    global _name_map_cache

    if _name_map_cache:
        return _name_map_cache

    if os.path.exists(NAME_MAP_PATH):
        try:
            with open(NAME_MAP_PATH, "r", encoding="utf-8") as f:
                _name_map_cache = json.load(f)
            print(f"[NAME_RESOLVER] Loaded {len(_name_map_cache)} mappings from {NAME_MAP_PATH}")
        except Exception as e:
            print(f"[WARNING] Could not load name map: {e}")
            _name_map_cache = {}

    return _name_map_cache


def save_name_map(name_map: Dict[str, Optional[str]] = None) -> None:
    """Save name mapping to JSON file."""
    global _name_map_cache

    if name_map is None:
        name_map = _name_map_cache

    os.makedirs(os.path.dirname(NAME_MAP_PATH) or ".", exist_ok=True)

    with open(NAME_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=2, ensure_ascii=False)

    print(f"[NAME_RESOLVER] Saved {len(name_map)} mappings to {NAME_MAP_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# FUZZY MATCHING
# ══════════════════════════════════════════════════════════════════════════════

def fuzzy_match(name: str, candidates: Set[str], threshold: int = FUZZY_THRESHOLD) -> Tuple[Optional[str], int]:
    """
    Find best fuzzy match for a name among candidates.

    Args:
        name: Name to match
        candidates: Set of candidate names to match against
        threshold: Minimum match score (0-100)

    Returns:
        Tuple of (matched_name, score) or (None, 0) if no match
    """
    try:
        from rapidfuzz import process, fuzz

        if not candidates:
            return None, 0

        name_norm = normalize_name(name)
        candidates_list = list(candidates)

        # Try exact surname match first (more reliable)
        surname = extract_surname(name)
        surname_matches = [c for c in candidates_list if extract_surname(c) == surname]

        if surname_matches:
            # Among surname matches, find best overall match
            result = process.extractOne(
                name_norm,
                [normalize_name(c) for c in surname_matches],
                scorer=fuzz.WRatio
            )
            if result and result[1] >= threshold:
                # Return original name (not normalized)
                idx = [normalize_name(c) for c in surname_matches].index(result[0])
                return surname_matches[idx], result[1]

        # Fallback to general fuzzy match
        result = process.extractOne(
            name_norm,
            [normalize_name(c) for c in candidates_list],
            scorer=fuzz.WRatio
        )

        if result and result[1] >= threshold:
            idx = [normalize_name(c) for c in candidates_list].index(result[0])
            return candidates_list[idx], result[1]

        return None, 0

    except ImportError:
        print("[WARNING] rapidfuzz not installed. Fuzzy matching disabled.")
        return None, 0


def surname_initial_match(full_name: str, cricsheet_players: Set[str]) -> Optional[str]:
    """
    Match using surname + initial pattern.

    Example: "Jasprit Bumrah" matches "JJ Bumrah" (same surname, initials start with J)
    """
    full_norm = normalize_name(full_name)
    parts = full_norm.split()

    if len(parts) < 2:
        return None

    surname = parts[-1]
    first_initial = parts[0][0] if parts[0] else ""

    for cs_name in cricsheet_players:
        cs_norm = normalize_name(cs_name)
        cs_parts = cs_norm.split()

        if len(cs_parts) < 2:
            continue

        cs_surname = cs_parts[-1]
        cs_initials = cs_parts[0]

        # Check same surname and initials start with same letter
        if cs_surname == surname and cs_initials and cs_initials[0] == first_initial:
            return cs_name

    return None

# ══════════════════════════════════════════════════════════════════════════════
# CORE RESOLUTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_name(raw_name: str, use_cache: bool = True) -> Optional[str]:
    """
    Resolve any name format to cricsheet name.

    Resolution order:
    1. Check cache
    2. Check manual mapping (both directions)
    3. Check exact match in cricsheet data
    4. Try surname+initial match
    5. Try fuzzy match
    6. Return None (new player)

    Args:
        raw_name: Player name in any format
        use_cache: Whether to use/update cache

    Returns:
        Cricsheet name or None if not found
    """
    global _name_map_cache

    if not raw_name or not isinstance(raw_name, str):
        return None

    raw_name = raw_name.strip()
    norm_name = normalize_name(raw_name)

    # 1. Check cache
    if use_cache:
        load_name_map()
        if raw_name in _name_map_cache:
            return _name_map_cache[raw_name]
        if norm_name in _name_map_cache:
            return _name_map_cache[norm_name]

    # Load cricsheet players
    cricsheet_players = load_cricsheet_players()
    cricsheet_norm = {normalize_name(p): p for p in cricsheet_players}

    # 2. Check exact match in cricsheet (normalized)
    if norm_name in cricsheet_norm:
        result = cricsheet_norm[norm_name]
        if use_cache:
            _name_map_cache[raw_name] = result
        return result

    # 3. Check manual mapping (full name -> cricsheet format)
    if norm_name in FULL_TO_CRICSHEET:
        cs_norm = FULL_TO_CRICSHEET[norm_name]
        if cs_norm in cricsheet_norm:
            result = cricsheet_norm[cs_norm]
            if use_cache:
                _name_map_cache[raw_name] = result
            return result

    # 4. Check manual mapping (cricsheet -> full, reverse lookup)
    if norm_name in MANUAL_CRICSHEET_TO_FULL:
        # The input is already in cricsheet format
        if norm_name in cricsheet_norm:
            result = cricsheet_norm[norm_name]
            if use_cache:
                _name_map_cache[raw_name] = result
            return result

    # 5. Try surname + initial match
    surname_match = surname_initial_match(raw_name, cricsheet_players)
    if surname_match:
        if use_cache:
            _name_map_cache[raw_name] = surname_match
        return surname_match

    # 6. Try fuzzy match
    fuzzy_result, score = fuzzy_match(raw_name, cricsheet_players)
    if fuzzy_result:
        if use_cache:
            _name_map_cache[raw_name] = fuzzy_result
        return fuzzy_result

    # 7. Not found (new player)
    if use_cache:
        _name_map_cache[raw_name] = None
    return None


def get_cricsheet_name(full_name: str) -> Optional[str]:
    """
    Get cricsheet name for a full name (convenience wrapper).

    Args:
        full_name: Full player name (e.g., "Jasprit Bumrah")

    Returns:
        Cricsheet name (e.g., "JJ Bumrah") or None
    """
    return resolve_name(full_name)


def resolve_squad_names(squad: List[str]) -> Dict[str, Optional[str]]:
    """
    Resolve all names in a squad to cricsheet format.

    Args:
        squad: List of player names

    Returns:
        Dict mapping original names to cricsheet names
    """
    return {name: resolve_name(name) for name in squad}

# ══════════════════════════════════════════════════════════════════════════════
# BUILD COMPLETE MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def build_name_mapping(squad_file: str = None, save: bool = True) -> Dict[str, Optional[str]]:
    """
    Build complete name mapping from squad data.

    Args:
        squad_file: Path to squad markdown file (optional)
        save: Whether to save mapping to JSON

    Returns:
        Complete name mapping dict
    """
    global _name_map_cache

    print("\n" + "=" * 60)
    print("  PITCHMIND — Building Player Name Mapping")
    print("=" * 60)

    # Load cricsheet players
    cricsheet_players = load_cricsheet_players()
    print(f"\n[1] Loaded {len(cricsheet_players)} players from Cricsheet data")

    # Parse squad file if provided
    squad_names = []
    if squad_file and os.path.exists(squad_file):
        try:
            with open(squad_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract player names from markdown table format
            # Pattern: | Player Name | Role | ...
            lines = content.split("\n")
            for line in lines:
                if line.startswith("|") and "Player Name" not in line and "---" not in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 2 and parts[1]:
                        name = parts[1].strip()
                        if name and not name.startswith("**"):
                            squad_names.append(name)

            print(f"[2] Parsed {len(squad_names)} players from squad file")

        except Exception as e:
            print(f"[WARNING] Could not parse squad file: {e}")

    # Build mapping
    mapping = {}
    exact_matches = 0
    manual_matches = 0
    fuzzy_matches = 0
    unmatched = []

    for name in squad_names:
        result = resolve_name(name, use_cache=False)
        mapping[name] = result

        if result:
            norm = normalize_name(name)
            cs_norm = normalize_name(result)

            if norm == cs_norm:
                exact_matches += 1
            elif norm in FULL_TO_CRICSHEET or norm in MANUAL_CRICSHEET_TO_FULL:
                manual_matches += 1
            else:
                fuzzy_matches += 1
        else:
            unmatched.append(name)

    # Update cache
    _name_map_cache.update(mapping)

    # Print stats
    print(f"\n[3] Resolution Results:")
    print(f"    - Exact matches:  {exact_matches}")
    print(f"    - Manual mapped:  {manual_matches}")
    print(f"    - Fuzzy matched:  {fuzzy_matches}")
    print(f"    - Total matched:  {exact_matches + manual_matches + fuzzy_matches}")
    print(f"    - Unmatched (new): {len(unmatched)}")

    if unmatched and len(unmatched) <= 20:
        print(f"\n[4] Unmatched players (no history):")
        for name in unmatched[:20]:
            print(f"    - {name}")

    # Save if requested
    if save:
        save_name_map(_name_map_cache)

    return mapping


def get_resolution_stats() -> Dict:
    """Get statistics about name resolution."""
    cricsheet = load_cricsheet_players()
    name_map = load_name_map()

    matched = sum(1 for v in name_map.values() if v is not None)
    unmatched = sum(1 for v in name_map.values() if v is None)

    return {
        "cricsheet_players": len(cricsheet),
        "mapped_names": len(name_map),
        "matched": matched,
        "unmatched": unmatched,
        "match_rate": f"{matched / max(len(name_map), 1) * 100:.1f}%"
    }

# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK STATS FOR NEW PLAYERS
# ══════════════════════════════════════════════════════════════════════════════

def get_fallback_stats(role: str, team: str, bat_df: pd.DataFrame = None, bowl_df: pd.DataFrame = None) -> Dict:
    """
    Get fallback stats for new players without historical data.
    Uses team/role averages.

    Args:
        role: Player role (e.g., "Batter", "Bowler", "Allrounder")
        team: Team name
        bat_df: Batting stats DataFrame
        bowl_df: Bowling stats DataFrame

    Returns:
        Dict with fallback batting and bowling stats
    """
    fallback = {"batting": {}, "bowling": {}}

    role_lower = role.lower() if role else ""

    # Default league averages (when no team data)
    default_batting = {
        "strike_rate": 130.0,
        "batting_avg": 25.0,
        "boundary_pct": 15.0,
    }

    default_bowling = {
        "economy": 8.5,
        "bowling_avg": 28.0,
        "bowling_sr": 20.0,
    }

    # If we have dataframes, compute team averages
    if bat_df is not None and len(bat_df) > 0:
        # Use overall averages (team-specific would need team column in stats)
        fallback["batting"] = {
            "strike_rate": round(bat_df["strike_rate"].mean(), 1),
            "batting_avg": round(bat_df["batting_avg"].mean(), 1),
            "boundary_pct": round(bat_df["boundary_pct"].mean(), 1),
            "is_fallback": True,
        }
    else:
        fallback["batting"] = {**default_batting, "is_fallback": True}

    if bowl_df is not None and len(bowl_df) > 0:
        fallback["bowling"] = {
            "economy": round(bowl_df["economy"].mean(), 2),
            "bowling_avg": round(bowl_df["bowling_avg"].mean(), 1),
            "bowling_sr": round(bowl_df["bowling_sr"].mean(), 1),
            "is_fallback": True,
        }
    else:
        fallback["bowling"] = {**default_bowling, "is_fallback": True}

    # Adjust based on role
    if "bowler" in role_lower and "all" not in role_lower:
        # Pure bowlers get minimal batting stats
        fallback["batting"]["strike_rate"] = 90.0
        fallback["batting"]["batting_avg"] = 8.0
    elif "batter" in role_lower and "all" not in role_lower:
        # Pure batters get minimal bowling stats
        fallback["bowling"]["economy"] = 10.0
        fallback["bowling"]["bowling_avg"] = 50.0

    return fallback

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Build mapping when run directly
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Default squad file path
    squad_file = "Squad_Data/IPL 2026 Squads — Quick Navigation 3263781c88e08017814dd5af05018b2e.md"

    if len(sys.argv) > 1:
        squad_file = sys.argv[1]

    if os.path.exists(squad_file):
        build_name_mapping(squad_file, save=True)
    else:
        print(f"[ERROR] Squad file not found: {squad_file}")
        print("        Run with: python name_resolver.py <path_to_squad_md>")

        # Still build basic mapping from manual mappings
        print("\n[INFO] Building mapping from manual mappings only...")
        cricsheet = load_cricsheet_players()
        print(f"Loaded {len(cricsheet)} cricsheet players")

        # Test a few names
        test_names = ["Jasprit Bumrah", "Virat Kohli", "MS Dhoni", "Rohit Sharma", "Hardik Pandya"]
        print("\n[TEST] Sample resolutions:")
        for name in test_names:
            cs_name = resolve_name(name)
            print(f"    {name:20} -> {cs_name}")

        save_name_map()

    # Print stats
    stats = get_resolution_stats()
    print(f"\n[STATS] Name Resolution Summary:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
