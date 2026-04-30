"""
PITCHMIND — Cricdata Live Match Fetcher (Final Fixed Version)
==============================================================
- Prioritizes local cache (ipl_2026_matches.json)
- Falls back to CricAPI
- Ensures team1, team2, venue, match_name are ALWAYS populated
- Improved merge_manual_fields to protect manual overrides
"""

from __future__ import annotations

import os
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

CRICAPI_BASE = "https://api.cricapi.com/v1"
_CACHED_API_KEY: Optional[str] = None

# Local cache path
LOCAL_MATCHES_CACHE = os.path.join(_REPO_ROOT, "espn_live_scraper", "cache", "ipl_2026_matches.json")


def get_api_key() -> str:
    global _CACHED_API_KEY
    if _CACHED_API_KEY is not None:
        return _CACHED_API_KEY

    k = os.environ.get("CRICAPI_KEY", "").strip()
    if k:
        _CACHED_API_KEY = k
        return k

    try:
        from espn_live_scraper.app import API_KEY as app_key
        if app_key:
            _CACHED_API_KEY = str(app_key).strip()
            return _CACHED_API_KEY
    except Exception:
        pass

    _CACHED_API_KEY = ""
    return ""


def load_local_match_cache() -> List[dict]:
    """Load all matches from local cache."""
    if not os.path.exists(LOCAL_MATCHES_CACHE):
        return []
    try:
        with open(LOCAL_MATCHES_CACHE, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get("data", [])
    except Exception:
        return []


def find_match_in_local_cache(match_id: str) -> Optional[dict]:
    """Find match by ID in local cache."""
    matches = load_local_match_cache()
    for m in matches:
        if m.get("id") == match_id:
            return m
    return None


def _request(endpoint: str, api_key: str, params: Optional[dict] = None) -> dict:
    p = dict(params or {})
    p["apikey"] = api_key
    try:
        r = requests.get(f"{CRICAPI_BASE}/{endpoint}", params=p, timeout=15)
        if r.status_code == 429:
            return {"status": "failure", "reason": "Rate limit (100/day)"}
        if r.status_code == 401:
            return {"status": "failure", "reason": "Invalid API key"}
        if r.status_code != 200:
            return {"status": "failure", "reason": f"HTTP {r.status_code}"}
        return r.json()
    except Exception as e:
        return {"status": "failure", "reason": str(e)}


# ==================== HELPER FUNCTIONS ====================

def normalize_team_to_dataset(api_name: str, candidates: List[str]) -> str:
    if not api_name:
        return ""
    n = str(api_name).strip().lower()
    for c in candidates:
        if c.lower() == n or n in c.lower() or c.lower() in n:
            return c
    return str(api_name).strip().lower()


def normalize_venue_to_dataset(api_venue: str, candidates: List[str]) -> str:
    if not api_venue:
        return ""
    flat = re.sub(r"\s+", " ", str(api_venue).lower().replace(".", " ").strip())
    for c in candidates:
        cs = re.sub(r"\s+", " ", str(c).lower().replace(".", " ").strip())
        if flat in cs or cs in flat:
            return c
    return str(api_venue).strip()


def _player_cell(p: Any) -> str:
    if p is None:
        return ""
    if isinstance(p, dict):
        return str(p.get("name") or p.get("player") or p.get("shortName") or "")
    return str(p)


def _extract_xis(m: dict, team1: str, team2: str) -> Tuple[List[str], List[str]]:
    t1xi, t2xi = [], []
    team_info = m.get("teamInfo") or []

    def assign(name: str, players: List[Any], into_t1: List[str], into_t2: List[str]):
        name_l = (name or "").lower()
        plist = [_player_cell(x) for x in players if _player_cell(x)]
        if not plist:
            return
        if team1 and team1.lower() in name_l:
            into_t1.extend(plist)
        elif team2 and team2.lower() in name_l:
            into_t2.extend(plist)

    for ti in team_info:
        assign(ti.get("name", ""), ti.get("players") or [], t1xi, t2xi)

    if not t1xi and not t2xi:
        for sq in m.get("squads") or []:
            assign(sq.get("name", ""), sq.get("players") or [], t1xi, t2xi)

    return t1xi, t2xi


def _current_score_from_match(m: dict) -> Dict[str, Any]:
    scores = m.get("score") or []
    if not scores:
        return {"runs": 0, "wickets": 0, "overs": 0.0}
    inn = scores[-1]
    return {
        "runs": int(inn.get("r", 0) or 0),
        "wickets": int(inn.get("w", 0) or 0),
        "overs": round(float(inn.get("o", 0) or 0), 2),
    }


def _batters_bowler_from_match(m: dict) -> Tuple[str, str, str]:
    striker = non_striker = bowler_name = ""
    for inn in reversed(m.get("score") or []):
        bats = inn.get("batsman") or inn.get("batsmen") or []
        if isinstance(bats, list) and bats:
            for b in bats:
                if not isinstance(b, dict):
                    continue
                nm = _player_cell(b)
                if (b.get("active") or b.get("strike") or "").lower() in ("true", "y", "yes", "striker"):
                    striker = nm
                elif not non_striker:
                    non_striker = nm
        bw = inn.get("bowler") or inn.get("bowlers")
        if isinstance(bw, list) and bw:
            bowler_name = _player_cell(bw[0] if isinstance(bw[0], dict) else bw)
        elif isinstance(bw, dict):
            bowler_name = _player_cell(bw)
        if striker or bowler_name:
            break
    return striker, non_striker, bowler_name


def map_players_through_name_map(names: List[str]) -> List[str]:
    try:
        from name_resolver import resolve_name
        out = []
        seen = set()
        for n in names:
            if not n:
                continue
            r = resolve_name(n) or n
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out
    except ImportError:
        return names


# ==================== MAIN FUNCTIONS ====================

def build_todays_match_payload(
    m: dict,
    match_id: str,
    *,
    dataset_teams: List[str],
    dataset_venues: List[str],
    ball_by_ball: Optional[List[dict]] = None,
) -> dict:
    """Build payload ensuring core fields are populated."""
    teams = m.get("teams") or []
    t1_raw = teams[0] if len(teams) > 0 else ""
    t2_raw = teams[1] if len(teams) > 1 else ""

    team1 = normalize_team_to_dataset(t1_raw, dataset_teams) or t1_raw.lower()
    team2 = normalize_team_to_dataset(t2_raw, dataset_teams) or t2_raw.lower()

    t1xi, t2xi = _extract_xis(m, team1, team2)
    t1xi = map_players_through_name_map(t1xi)
    t2xi = map_players_through_name_map(t2xi)

    venue_raw = m.get("venue") or ""
    venue_norm = normalize_venue_to_dataset(venue_raw, dataset_venues)

    tw = m.get("tossWinner") or ""
    td = m.get("tossChoice") or m.get("toss_decision") or ""

    toss_winner_norm = normalize_team_to_dataset(tw, dataset_teams) if tw else ""

    striker, non_striker, bowler_name = _batters_bowler_from_match(m)

    return {
        "match_id": match_id,
        "match_name": m.get("name") or f"{team1.title()} vs {team2.title()}",
        "match_date": m.get("date") or (m.get("dateTimeGMT") or "")[:10],
        "league": "IPL",
        "venue": venue_norm or venue_raw,
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner_norm,
        "toss_decision": str(td).lower() if td else "",
        "team1_xi": t1xi,
        "team2_xi": t2xi,
        "current_score": _current_score_from_match(m),
        "batters": {"striker": striker, "non_striker": non_striker},
        "bowler": bowler_name,
        "ball_by_ball": ball_by_ball or [],
        "match_status": m.get("status") or "",
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": "cricdata",
    }


def merge_manual_fields(saved: dict, incoming: dict) -> dict:
    """
    FIXED: Properly merges manual overrides with fresh fetch data.
    - Core match information (team1, team2, venue, etc.) comes from 'incoming' (fetch)
    - Manual fields (pitch, dew, toss, XIs) are preserved from 'saved'
    """
    if not saved:
        return incoming.copy()

    merged = incoming.copy()

    # Core fields: Always prefer fresh data from fetch / local cache
    core_fields = ["team1", "team2", "venue", "match_name", "match_date", "league", "match_status"]
    for key in core_fields:
        if incoming.get(key):  # if new data has value
            merged[key] = incoming[key]

    # Preserve manual overrides
    manual_fields = ["pitch_type", "dew_expected", "toss_winner", "toss_decision"]
    for key in manual_fields:
        if saved.get(key) not in (None, "", [], {}):
            merged[key] = saved[key]

    # Playing XI: Keep manual XI if user has entered better/longer list
    for xi_key in ["team1_xi", "team2_xi"]:
        saved_xi = saved.get(xi_key, [])
        incoming_xi = incoming.get(xi_key, [])
        if saved_xi and len(saved_xi) >= len(incoming_xi):
            merged[xi_key] = saved_xi

    return merged


def fetch_live_match_for_dashboard(
    match_id: str,
    *,
    api_key: Optional[str] = None,
    dataset_teams: List[str],
    dataset_venues: List[str],
    include_bbb: bool = True,
) -> Tuple[Optional[dict], Optional[str]]:
    """Main fetch function with local cache priority."""
    mid = (match_id or "").strip()
    if not mid:
        return None, "match_id is required"

    # 1. Try local cache first (most reliable for basic info)
    local_match = find_match_in_local_cache(mid)
    if local_match:
        print(f"✅ Loaded from local cache: {local_match.get('name', mid)}")
        payload = build_todays_match_payload(
            local_match, mid, dataset_teams=dataset_teams, dataset_venues=dataset_venues
        )

        # Optional: Enrich with live CricAPI data (toss, score, etc.)
        try:
            key = api_key or get_api_key()
            if key:
                api_resp = _request("match_info", key, {"id": mid})
                if api_resp.get("status") == "success" and api_resp.get("data"):
                    api_m = api_resp["data"]
                    live_payload = build_todays_match_payload(
                        api_m, mid, dataset_teams=dataset_teams, dataset_venues=dataset_venues
                    )
                    # Enrich only live fields
                    for k in ["toss_winner", "toss_decision", "current_score", "batters", "bowler", "team1_xi", "team2_xi"]:
                        if live_payload.get(k):
                            payload[k] = live_payload[k]
        except Exception:
            pass

        return payload, None

    # 2. Fallback to CricAPI
    key = (api_key or get_api_key() or "").strip()
    if not key:
        return None, "Match not found in local cache and no CRICAPI_KEY available"

    data = _request("match_info", key, {"id": mid})
    if data.get("status") != "success":
        return None, data.get("reason") or "Failed to fetch from CricAPI"

    m = data.get("data") or {}
    if not m:
        return None, "Empty data from CricAPI"

    bbb_list = []
    if include_bbb and m.get("bbbEnabled"):
        bbb_list, _ = fetch_match_bbb(key, mid) if 'fetch_match_bbb' in globals() else ([], None)

    payload = build_todays_match_payload(
        m, mid, dataset_teams=dataset_teams, dataset_venues=dataset_venues, ball_by_ball=bbb_list
    )
    return payload, None


# ==================== UTILITY FUNCTIONS ====================

def save_todays_match(path: str, payload: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_todays_match(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_blank_template(match_id: str = "") -> dict:
    return {
        "match_id": match_id,
        "match_name": "",
        "match_date": "",
        "league": "IPL",
        "venue": "",
        "team1": "",
        "team2": "",
        "toss_winner": "",
        "toss_decision": "",
        "team1_xi": [],
        "team2_xi": [],
        "current_score": {"runs": 0, "wickets": 0, "overs": 0.0},
        "batters": {"striker": "", "non_striker": ""},
        "bowler": "",
        "ball_by_ball": [],
        "match_status": "",
        "last_updated": "",
        "data_source": "cricdata",
    }