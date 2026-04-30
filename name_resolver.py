"""
PITCHMIND — PLAYER NAME RESOLUTION SYSTEM  (v4 — Deliveries-First)
====================================================================
Key change vs v3:
  - resolve_name() NOW RETURNS the CRICSHEET name (as stored in deliveries_clean.csv)
    not the full squad name. This is what bat_lookup / bowl_lookup keys use.
  - v3 was returning "Virat Kohli" but the lookup needed "V Kohli" → always missed.
  - The deliveries CSV is the single source of truth for player keys.
  - Squad JSON is used only to confirm a player is in 2026 squads.
  - Auto-generates full bidirectional map from deliveries data on first run.
  - Saves to data/name_map.json for instant reuse.

Resolution order for any input name:
  1. Exact match in deliveries (already a cricsheet name) → return as-is
  2. name_map.json cache
  3. FULL_TO_CRICSHEET manual map  (e.g. "Virat Kohli" → "V Kohli")
  4. CRICSHEET_TO_CRICSHEET aliases (e.g. "jj bumrah" → "JJ Bumrah")
  5. Surname + initial heuristic   (e.g. "Jasprit Bumrah" → "JJ Bumrah")
  6. Fuzzy match against deliveries players
  7. None

Usage:
    from name_resolver import resolve_name, get_team_squad

    resolve_name("V Kohli")        # → "V Kohli"   (already cricsheet)
    resolve_name("Virat Kohli")    # → "V Kohli"   (full → cricsheet)
    resolve_name("Jasprit Bumrah") # → "JJ Bumrah" (full → cricsheet)
    resolve_name("jj bumrah")      # → "JJ Bumrah" (case fix)
"""

import os
import json
import re
import pandas as pd
from typing import Optional, Dict, Set, List, Tuple

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR         = "data"
SQUADS_JSON_PATH = os.path.join("Squad_Data", "ipl_2026_team_squads.json")
NAME_MAP_PATH    = os.path.join(DATA_DIR, "name_map.json")
FUZZY_THRESHOLD  = 80

# ══════════════════════════════════════════════════════════════════════════════
# FULL NAME → CRICSHEET NAME  (manually verified, covers all 2026 squads)
# Key   = any input format (full name, CricAPI name, broadcast name)
# Value = exact key as it appears in deliveries_clean.csv
# ══════════════════════════════════════════════════════════════════════════════
FULL_TO_CRICSHEET: Dict[str, str] = {

    # ── Mumbai Indians ────────────────────────────────────────────────────────
    "Jasprit Bumrah"       : "JJ Bumrah",
    "Rohit Sharma"         : "RG Sharma",
    "Hardik Pandya"        : "HH Pandya",
    "Suryakumar Yadav"     : "SA Yadav",
    "Tilak Varma"          : "Tilak Varma",
    "Tim David"            : "TH David",
    "Quinton de Kock"      : "Q de Kock",
    "Trent Boult"          : "TA Boult",
    "Mitchell Santner"     : "MJ Santner",
    "Will Jacks"           : "WG Jacks",
    "Deepak Chahar"        : "DL Chahar",
    "Shardul Thakur"       : "SN Thakur",
    "Naman Dhir"           : "Naman Dhir",
    "Ryan Rickelton"       : "RD Rickelton",
    "Reece Topley"         : "RJW Topley",

    # ── Chennai Super Kings ───────────────────────────────────────────────────
    "MS Dhoni"             : "MS Dhoni",
    "Ruturaj Gaikwad"      : "RD Gaikwad",
    "Shivam Dube"          : "S Dube",
    "Matthew Short"        : "MW Short",
    "Noor Ahmad"           : "Noor Ahmad",
    "Khaleel Ahmed"        : "KK Ahmed",
    "Rahul Chahar"         : "RD Chahar",
    "Devon Conway"         : "DP Conway",
    "Ravindra Jadeja"      : "RA Jadeja",
    "Ambati Rayudu"        : "AT Rayudu",
    "Moeen Ali"            : "MM Ali",
    "Rachin Ravindra"      : "R Ravindra",
    "Shaik Rasheed"        : "SK Rasheed",

    # ── Royal Challengers Bengaluru ───────────────────────────────────────────
    "Virat Kohli"          : "V Kohli",
    "Josh Hazlewood"       : "JR Hazlewood",
    "Krunal Pandya"        : "KH Pandya",
    "Bhuvneshwar Kumar"    : "B Kumar",
    "Jacob Bethell"        : "JG Bethell",
    "Phil Salt"            : "PD Salt",
    "Romario Shepherd"     : "R Shepherd",
    "Yash Dayal"           : "Yash Dayal",
    "Venkatesh Iyer"       : "VR Iyer",
    "Vicky Ostwal"         : "VC Ostwal",
    "Devdutt Padikkal"     : "D Padikkal",
    "Rajat Patidar"        : "RM Patidar",
    "Suyash Sharma"        : "Suyash Sharma",
    "Jitesh Sharma"        : "JM Sharma",
    "Rasikh Dar"           : "Rasikh Salam",
    "Rasikh Salam Dar"     : "Rasikh Salam",
    "Tim Seifert"          : "TL Seifert",
    "Swapnil Singh"        : "Swapnil Singh",
    "Abhinandan Singh"     : "Abhinandan Singh",

    # ── Kolkata Knight Riders ─────────────────────────────────────────────────
    "Ajinkya Rahane"       : "AM Rahane",
    "Sunil Narine"         : "SP Narine",
    "Rinku Singh"          : "RK Singh",
    "Varun Chakravarthy"   : "CV Varun",
    "Matheesha Pathirana"  : "M Pathirana",
    "Angkrish Raghuvanshi" : "A Raghuvanshi",
    "Manish Pandey"        : "MK Pandey",
    "Finn Allen"           : "FH Allen",
    "Rovman Powell"        : "R Powell",
    "Mitchell Starc"       : "MA Starc",
    "Cameron Green"        : "C Green",
    "Harshit Rana"         : "Harshit Rana",
    "Umran Malik"          : "Umran Malik",
    "Anrich Nortje"        : "A Nortje",
    "Lhuan-dre Pretorius"  : "LG Pretorius",
    "Kwena Maphaka"        : "KT Maphaka",

    # ── Sunrisers Hyderabad ───────────────────────────────────────────────────
    "Travis Head"          : "TM Head",
    "Pat Cummins"          : "PJ Cummins",
    "Heinrich Klaasen"     : "H Klaasen",
    "Ishan Kishan"         : "Ishan Kishan",
    "Nitish Kumar Reddy"   : "Nithish Kumar Reddy",
    "Abhishek Sharma"      : "Abhishek Sharma",
    "Brydon Carse"         : "BM Carse",
    "Harshal Patel"        : "HV Patel",
    "Liam Livingstone"     : "LS Livingstone",
    "Kamindu Mendis"       : "PHKD Mendis",
    "Zeeshan Ansari"       : "Zeeshan Ansari",
    "Simarjeet Singh"      : "Simarjeet Singh",
    "Atharva Taide"        : "Atharva Taide",
    "Mohammed Shami"       : "Mohammed Shami",

    # ── Delhi Capitals ────────────────────────────────────────────────────────
    "Axar Patel"           : "AR Patel",
    "KL Rahul"             : "KL Rahul",
    "Kuldeep Yadav"        : "K Yadav",
    "Lungi Ngidi"          : "L Ngidi",
    "Karun Nair"           : "KK Nair",
    "Nitish Rana"          : "N Rana",
    "Prithvi Shaw"         : "PP Shaw",
    "Tristan Stubbs"       : "T Stubbs",
    "Ben Duckett"          : "BA Duckett",
    "David Miller"         : "DA Miller",
    "Kyle Jamieson"        : "KA Jamieson",
    "T Natarajan"          : "T Natarajan",
    "Faf du Plessis"       : "F du Plessis",
    "Jake Fraser-McGurk"   : "J Fraser-McGurk",
    "Mohit Rathee"         : "Mohit Rathee",
    "Ashutosh Sharma"      : "Ashutosh Sharma",
    "Sameer Rizvi"         : "Sameer Rizvi",
    "Abishek Porel"        : "Abishek Porel",
    "Daryl Mitchell"       : "DJ Mitchell",

    # ── Punjab Kings ─────────────────────────────────────────────────────────
    "Shreyas Iyer"         : "SS Iyer",
    "Arshdeep Singh"       : "Arshdeep Singh",
    "Yuzvendra Chahal"     : "YS Chahal",
    "Marco Jansen"         : "M Jansen",
    "Marcus Stoinis"       : "MP Stoinis",
    "Lockie Ferguson"      : "LH Ferguson",
    "Kagiso Rabada"        : "K Rabada",
    "Priyansh Arya"        : "Priyansh Arya",
    "Musheer Khan"         : "Musheer Khan",
    "Shashank Singh"       : "Shashank Singh",
    "Prabhsimran Singh"    : "P Simran Singh",
    "Glenn Maxwell"        : "GJ Maxwell",
    "Rilee Rossouw"        : "RR Rossouw",
    "Harnoor Brar"         : "Harpreet Brar",
    "Harpreet Brar"        : "Harpreet Brar",
    "Vishwanath Vyshak"    : "Vijaykumar Vyshak",

    # ── Gujarat Titans ────────────────────────────────────────────────────────
    "Shubman Gill"         : "Shubman Gill",
    "Jos Buttler"          : "JC Buttler",
    "Rahul Tewatia"        : "R Tewatia",
    "Rashid Khan"          : "Rashid Khan",
    "Washington Sundar"    : "Washington Sundar",
    "Mohammed Siraj"       : "Mohammed Siraj",
    "Prasidh Krishna"      : "M Prasidh Krishna",
    "Glenn Phillips"       : "GD Phillips",
    "Jason Holder"         : "JO Holder",
    "Sai Sudharsan"        : "B Sai Sudharsan",
    "Shahrukh Khan"        : "M Shahrukh Khan",
    "Manav Suthar"         : "MJ Suthar",
    "Anuj Rawat"           : "Anuj Rawat",
    "Gerald Coetzee"       : "G Coetzee",
    "Nandre Burger"        : "N Burger",
    "Kumar Kushagra"       : "Kumar Kushagra",
    "Karim Janat"          : "Karim Janat",

    # ── Lucknow Super Giants ──────────────────────────────────────────────────
    "Rishabh Pant"         : "RR Pant",
    "Nicholas Pooran"      : "N Pooran",
    "Aiden Markram"        : "AK Markram",
    "Avesh Khan"           : "Avesh Khan",
    "Mohsin Khan"          : "Mohsin Khan",
    "Mitchell Marsh"       : "MR Marsh",
    "Wanindu Hasaranga"    : "PW Hasaranga",
    "Mayank Yadav"         : "MP Yadav",
    "Ayush Badoni"         : "A Badoni",
    "Josh Inglis"          : "JP Inglis",
    "Abdul Samad"          : "Abdul Samad",
    "Akash Deep"           : "Akash Deep",
    "Ravi Bishnoi"         : "Ravi Bishnoi",
    "David Warner"         : "DA Warner",
    "Marcus Stoinis"       : "MP Stoinis",
    "Deepak Hooda"         : "DJ Hooda",
    "Piyush Chawla"        : "PP Chawla",

    # ── Rajasthan Royals ─────────────────────────────────────────────────────
    "Yashasvi Jaiswal"     : "YBK Jaiswal",
    "Riyan Parag"          : "R Parag",
    "Jofra Archer"         : "JC Archer",
    "Shimron Hetmyer"      : "SO Hetmyer",
    "Dhruv Jurel"          : "Dhruv Jurel",
    "Vaibhav Suryavanshi"  : "V Suryavanshi",
    "Sandeep Sharma"       : "Sandeep Sharma",
    "Tushar Deshpande"     : "TU Deshpande",
    "Sanju Samson"         : "SV Samson",
    "Jos Buttler"          : "JC Buttler",
    "Trent Boult"          : "TA Boult",
    "Shimran Hetmyer"      : "SO Hetmyer",
    "Navdeep Saini"        : "NA Saini",
    "Kuldeep Sen"          : "KR Sen",
    "Tanush Kotian"        : "Tanush Kotian",
    "Yudhvir Singh"        : "Yudhvir Singh",
}

# ── ALSO build reverse: cricsheet exact → cricsheet exact (normalize casing) ──
# Handles "v kohli" → "V Kohli", "jj bumrah" → "JJ Bumrah"
# Built at module load from the deliveries file

_CRICSHEET_NORM_TO_EXACT: Dict[str, str] = {}   # "v kohli" -> "V Kohli"
_FULL_NORM_TO_CRICSHEET: Dict[str, str] = {}     # "virat kohli" -> "V Kohli"
_deliveries_players: Set[str] = set()
_squad_cache: Dict[str, List[str]] = {}
_all_squad_players: Set[str] = set()
_name_map_cache: Dict[str, Optional[str]] = {}


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION — build lookup tables from deliveries on first import
# ══════════════════════════════════════════════════════════════════════════════

def _load_deliveries_players() -> Set[str]:
    """Load all unique player names from deliveries_clean.csv (the ground truth)."""
    global _deliveries_players, _CRICSHEET_NORM_TO_EXACT
    if _deliveries_players:
        return _deliveries_players

    for path in [
        os.path.join(DATA_DIR, "deliveries_clean.csv"),
        os.path.join(DATA_DIR, "deliveries.csv"),
        "data/deliveries_clean.csv",
        "data/deliveries.csv",
    ]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False, usecols=["batter", "bowler"])
                batters = set(df["batter"].dropna().unique())
                bowlers = set(df["bowler"].dropna().unique())
                _deliveries_players = batters | bowlers
                # Build normalized → exact lookup
                _CRICSHEET_NORM_TO_EXACT = {
                    _norm(p): p for p in _deliveries_players
                }
                print(f"[NAME_RESOLVER] Loaded {len(_deliveries_players)} players "
                      f"from {path}")
                return _deliveries_players
            except Exception as e:
                print(f"[NAME_RESOLVER][WARN] Could not load {path}: {e}")

    print("[NAME_RESOLVER][WARN] No deliveries file found — name resolution limited.")
    return set()


def _build_full_norm_map() -> None:
    """Build normalized full-name → cricsheet-name lookup from FULL_TO_CRICSHEET."""
    global _FULL_NORM_TO_CRICSHEET
    _FULL_NORM_TO_CRICSHEET = {
        _norm(full): cs for full, cs in FULL_TO_CRICSHEET.items()
    }


def _norm(name: str) -> str:
    """Normalize: lowercase, strip dots and extra spaces."""
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    n = n.replace(".", "").replace("-", " ").replace("'", "")
    return " ".join(n.split())


def _load_squads() -> Dict[str, List[str]]:
    global _squad_cache, _all_squad_players
    if _squad_cache:
        return _squad_cache

    search_paths = [
        SQUADS_JSON_PATH,
        os.path.join(os.path.dirname(__file__), "ipl_2026_team_squads.json"),
        os.path.join(DATA_DIR, "ipl_2026_team_squads.json"),
        "ipl_2026_team_squads.json",
    ]
    for path in search_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for team, players in raw.items():
                clean = [re.sub(r"\s*\(c\)\s*$", "", p).strip() for p in players]
                _squad_cache[team] = clean
                _all_squad_players.update(clean)
            print(f"[NAME_RESOLVER] Loaded {len(_all_squad_players)} 2026 squad "
                  f"players from {path}")
            return _squad_cache
    return {}


def _load_name_map() -> Dict[str, Optional[str]]:
    global _name_map_cache
    if _name_map_cache:
        return _name_map_cache
    if os.path.exists(NAME_MAP_PATH):
        try:
            with open(NAME_MAP_PATH, "r", encoding="utf-8") as f:
                _name_map_cache = json.load(f)
            print(f"[NAME_RESOLVER] Loaded {len(_name_map_cache)} cached mappings.")
        except Exception:
            _name_map_cache = {}
    return _name_map_cache


def save_name_map() -> None:
    os.makedirs(os.path.dirname(NAME_MAP_PATH) or ".", exist_ok=True)
    with open(NAME_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(_name_map_cache, f, indent=2, ensure_ascii=False)


# Initialise on import
_load_deliveries_players()
_build_full_norm_map()
_load_name_map()
_load_squads()


# ══════════════════════════════════════════════════════════════════════════════
# CORE: resolve_name
# Returns the CRICSHEET name (deliveries key), not the full squad name.
# ══════════════════════════════════════════════════════════════════════════════

def resolve_name(raw_name: str, use_cache: bool = True) -> Optional[str]:
    """
    Resolve any player name to the exact string used in deliveries_clean.csv.

    Returns:
        The cricsheet-format name as stored in deliveries (e.g. "V Kohli")
        or None if the player is not found anywhere in the data.

    Examples:
        resolve_name("V Kohli")        → "V Kohli"    (exact match)
        resolve_name("Virat Kohli")    → "V Kohli"    (full → cricsheet)
        resolve_name("v kohli")        → "V Kohli"    (case fix)
        resolve_name("Jasprit Bumrah") → "JJ Bumrah"  (full → cricsheet)
        resolve_name("JJ Bumrah")      → "JJ Bumrah"  (exact match)
        resolve_name("MS Dhoni")       → "MS Dhoni"   (exact match)
    """
    if not raw_name or not isinstance(raw_name, str):
        return None

    raw_name = raw_name.strip()
    n = _norm(raw_name)

    # ── 1. Exact match in deliveries ─────────────────────────────────────────
    if raw_name in _deliveries_players:
        return raw_name

    # ── 2. Cache ──────────────────────────────────────────────────────────────
    if use_cache and raw_name in _name_map_cache:
        return _name_map_cache[raw_name]

    result = _resolve_uncached(raw_name, n)

    if use_cache:
        _name_map_cache[raw_name] = result

    return result


def _resolve_uncached(raw_name: str, n: str) -> Optional[str]:
    """Internal: try all resolution strategies in order."""

    # ── 3. Normalized exact match in deliveries ───────────────────────────────
    if n in _CRICSHEET_NORM_TO_EXACT:
        return _CRICSHEET_NORM_TO_EXACT[n]

    # ── 4. FULL_TO_CRICSHEET manual map ───────────────────────────────────────
    if n in _FULL_NORM_TO_CRICSHEET:
        cs_name = _FULL_NORM_TO_CRICSHEET[n]
        # Verify this cricsheet name actually exists in deliveries
        if cs_name in _deliveries_players:
            return cs_name
        # Try normalized lookup in case of slight casing difference
        cs_norm = _norm(cs_name)
        if cs_norm in _CRICSHEET_NORM_TO_EXACT:
            return _CRICSHEET_NORM_TO_EXACT[cs_norm]
        # Return the mapped name even if not in deliveries (new player)
        return cs_name

    # ── 5. Surname + first initial heuristic ─────────────────────────────────
    # "Jasprit Bumrah" → look for any "X Bumrah" or "XX Bumrah" in deliveries
    si_result = _surname_initial_heuristic(raw_name, n)
    if si_result:
        return si_result

    # ── 6. Single surname match (only if unique) ──────────────────────────────
    surname_result = _unique_surname_match(n)
    if surname_result:
        return surname_result

    # ── 7. Fuzzy match ────────────────────────────────────────────────────────
    fuzzy_result = _fuzzy_match(raw_name, _deliveries_players)
    if fuzzy_result:
        return fuzzy_result

    return None


def _surname_initial_heuristic(raw_name: str, n: str) -> Optional[str]:
    """
    For "Jasprit Bumrah" try to find "JJ Bumrah" or "J Bumrah" in deliveries.
    Matches on: same surname AND same first initial.
    """
    parts = n.split()
    if len(parts) < 2:
        return None

    surname      = parts[-1]
    first_initial = parts[0][0] if parts[0] else ""

    candidates = []
    for cs_name in _deliveries_players:
        cs_parts = _norm(cs_name).split()
        if not cs_parts:
            continue
        cs_surname = cs_parts[-1]
        if cs_surname != surname:
            continue
        # Initials block: e.g. "JJ" or "J" from "JJ Bumrah"
        if len(cs_parts) >= 2:
            cs_initials = cs_parts[0]  # "jj" from "jj bumrah"
            if cs_initials and cs_initials[0] == first_initial:
                candidates.append(cs_name)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer the one whose initials length matches input
        input_first = parts[0]
        for c in candidates:
            c_parts = _norm(c).split()
            if c_parts[0] == input_first:
                return c
        # Return shortest initials match (most common format)
        return sorted(candidates, key=lambda x: len(x))[0]

    return None


def _unique_surname_match(n: str) -> Optional[str]:
    """
    If input is just a surname or the surname uniquely identifies one player,
    return that player. E.g. "Bumrah" → "JJ Bumrah".
    """
    parts = n.split()
    surname = parts[-1]

    matches = [
        p for p in _deliveries_players
        if _norm(p).split()[-1] == surname
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _fuzzy_match(name: str, candidates: Set[str],
                 threshold: int = FUZZY_THRESHOLD) -> Optional[str]:
    try:
        from rapidfuzz import process, fuzz
        if not candidates:
            return None
        name_n     = _norm(name)
        cand_list  = list(candidates)
        cand_norms = [_norm(c) for c in cand_list]

        # Surname-first pass
        parts   = name_n.split()
        surname = parts[-1] if parts else ""
        sub_idx = [i for i, cn in enumerate(cand_norms)
                   if cn.split()[-1] == surname] if surname else []

        if sub_idx:
            sub_norms = [cand_norms[i] for i in sub_idx]
            sub_orig  = [cand_list[i]  for i in sub_idx]
            res = process.extractOne(name_n, sub_norms, scorer=fuzz.WRatio)
            if res and res[1] >= threshold:
                return sub_orig[sub_norms.index(res[0])]

        res = process.extractOne(name_n, cand_norms, scorer=fuzz.WRatio)
        if res and res[1] >= threshold:
            return cand_list[cand_norms.index(res[0])]
    except ImportError:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_team_squad(team: str) -> List[str]:
    """Return squad list for a team from ipl_2026_team_squads.json."""
    squads = _load_squads()
    if team in squads:
        return squads[team]
    tl = team.strip().lower()
    for k, v in squads.items():
        if k.lower() == tl:
            return v
    return []


def get_all_2026_players() -> Set[str]:
    _load_squads()
    return set(_all_squad_players)


def get_cricsheet_name(full_name: str) -> Optional[str]:
    """Alias for resolve_name — backward compatibility."""
    return resolve_name(full_name)


def resolve_squad_names(squad: List[str]) -> Dict[str, Optional[str]]:
    """Resolve every name in a list. Returns {original: cricsheet_or_None}."""
    return {name: resolve_name(name) for name in squad}


def is_2026_player(name: str) -> bool:
    return resolve_name(name) is not None


def get_resolution_stats() -> Dict:
    players = _load_deliveries_players()
    return {
        "deliveries_players" : len(players),
        "squad_2026_players" : len(get_all_2026_players()),
        "manual_mappings"    : len(FULL_TO_CRICSHEET),
        "cached_resolutions" : len(_name_map_cache),
        "matched"            : sum(1 for v in _name_map_cache.values() if v),
        "unmatched"          : sum(1 for v in _name_map_cache.values() if not v),
    }


def build_name_mapping(save: bool = True) -> Dict[str, Optional[str]]:
    """
    Build complete mapping for all 2026 squad players.
    Saves to data/name_map.json.
    """
    global _name_map_cache
    print("\n" + "=" * 60)
    print("  PITCHMIND — Building Name Mapping (v4 Deliveries-First)")
    print("=" * 60)

    squads = _load_squads()
    if not squads:
        print("[ERROR] No squads loaded.")
        return {}

    players = _load_deliveries_players()
    print(f"  Deliveries players : {len(players)}")
    print(f"  2026 squad players : {len(_all_squad_players)}")
    print()

    exact, manual, heuristic, fuzzy_ct, unmatched_list = 0, 0, 0, 0, []

    for team, squad in squads.items():
        for player in squad:
            n = _norm(player)
            cs = None

            # Exact in deliveries
            if player in players:
                cs = player; exact += 1
            elif n in _CRICSHEET_NORM_TO_EXACT:
                cs = _CRICSHEET_NORM_TO_EXACT[n]; exact += 1
            # Manual map
            elif n in _FULL_NORM_TO_CRICSHEET:
                cs = _FULL_NORM_TO_CRICSHEET[n]; manual += 1
            # Heuristic
            else:
                cs = _surname_initial_heuristic(player, n)
                if cs:
                    heuristic += 1
                else:
                    cs = _fuzzy_match(player, players)
                    if cs:
                        fuzzy_ct += 1
                    else:
                        unmatched_list.append((team, player))

            _name_map_cache[player] = cs

    total = exact + manual + heuristic + fuzzy_ct
    print(f"  Exact matches      : {exact}")
    print(f"  Manual mapped      : {manual}")
    print(f"  Heuristic matched  : {heuristic}")
    print(f"  Fuzzy matched      : {fuzzy_ct}")
    print(f"  ──────────────────────")
    print(f"  Total resolved     : {total}")
    print(f"  No data (new/uncap): {len(unmatched_list)}")
    if unmatched_list:
        print(f"\n  Players with no historical data:")
        for team, p in unmatched_list:
            print(f"    [{team}]  {p}")

    if save:
        save_name_map()

    return _name_map_cache


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK STATS (for new / uncapped players)
# ══════════════════════════════════════════════════════════════════════════════

def get_fallback_stats(role: str = "", bat_df=None, bowl_df=None) -> Dict:
    role_lower = role.lower()
    bat = (
        {"strike_rate": float(bat_df["strike_rate"].mean()),
         "batting_avg": float(bat_df["batting_avg"].mean()),
         "boundary_pct": float(bat_df["boundary_pct"].mean()),
         "is_fallback": True}
        if bat_df is not None and len(bat_df) > 0
        else {"strike_rate": 130.0, "batting_avg": 25.0,
              "boundary_pct": 15.0, "is_fallback": True}
    )
    bowl = (
        {"economy": float(bowl_df["economy"].mean()),
         "bowling_avg": float(bowl_df["bowling_avg"].mean()),
         "bowling_sr": float(bowl_df["bowling_sr"].mean()),
         "is_fallback": True}
        if bowl_df is not None and len(bowl_df) > 0
        else {"economy": 9.8, "bowling_avg": 28.0,
              "bowling_sr": 20.0, "is_fallback": True}
    )
    if "bowler" in role_lower and "all" not in role_lower:
        bat.update({"strike_rate": 90.0, "batting_avg": 8.0})
    elif "batter" in role_lower and "all" not in role_lower:
        bowl.update({"economy": 10.0, "bowling_avg": 50.0})

    return {"batting": bat, "bowling": bowl}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run to build / test mapping
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    build_name_mapping(save=True)

    print("\n[TEST] Sample resolutions (should all return cricsheet format):")
    tests = [
        # Full name → should return cricsheet key
        ("Virat Kohli",         "V Kohli"),
        ("Jasprit Bumrah",      "JJ Bumrah"),
        ("Rohit Sharma",        "RG Sharma"),
        ("Shubman Gill",        "Shubman Gill"),
        ("MS Dhoni",            "MS Dhoni"),
        ("Travis Head",         "TM Head"),
        ("Yashasvi Jaiswal",    "YBK Jaiswal"),
        ("Devdutt Padikkal",    "D Padikkal"),
        ("Josh Hazlewood",      "JR Hazlewood"),
        ("Krunal Pandya",       "KH Pandya"),
        ("Suyash Sharma",       "Suyash Sharma"),
        ("Rasikh Dar",          "Rasikh Salam"),
        # Already cricsheet → return as-is
        ("V Kohli",             "V Kohli"),
        ("JJ Bumrah",           "JJ Bumrah"),
        ("RG Sharma",           "RG Sharma"),
        # Lowercase input → should still resolve
        ("v kohli",             "V Kohli"),
        ("ms dhoni",            "MS Dhoni"),
        # Unknown player → None
        ("Random Player 9999",  None),
    ]

    passed = 0
    for name, expected in tests:
        result = resolve_name(name)
        ok = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        print(f"  {ok}  {name:28s} → {str(result):20s}  (expected: {expected})")

    print(f"\n  {passed}/{len(tests)} tests passed")
    print()
    stats = get_resolution_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")