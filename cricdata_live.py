"""
PITCHMIND — Cricdata (CricAPI) live match fetch
================================================
Single module for match_id-based live data. No ESPN / URL scraping.

Env: CRICAPI_KEY (optional; falls back to key used in espn_live_scraper/app.py flow).
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


def get_api_key() -> str:
    global _CACHED_API_KEY
    if _CACHED_API_KEY is not None:
        return _CACHED_API_KEY
    k = os.environ.get("CRICAPI_KEY", "").strip()
    if k:
        _CACHED_API_KEY = k
        return _CACHED_API_KEY
    try:
        from espn_live_scraper.app import API_KEY as app_key
        if app_key and str(app_key).strip():
            _CACHED_API_KEY = str(app_key).strip()
            return _CACHED_API_KEY
    except Exception:
        pass
    try:
        import runpy
        ns = runpy.run_path(os.path.join(_REPO_ROOT, "5_live_data_fetch.py"))
        lk = ns.get("CRICAPI_KEY") or ""
        _CACHED_API_KEY = str(lk).strip()
        return _CACHED_API_KEY
    except Exception:
        _CACHED_API_KEY = ""
        return ""


def _request(endpoint: str, api_key: str, params: Optional[dict] = None) -> dict:
    p = dict(params or {})
    p["apikey"] = api_key
    try:
        r = requests.get(f"{CRICAPI_BASE}/{endpoint}", params=p, timeout=15)
        if r.status_code == 429:
            return {"status": "failure", "reason": "Rate limit (100/day). Try again tomorrow."}
        if r.status_code == 401:
            return {"status": "failure", "reason": "Invalid CricAPI key."}
        if r.status_code != 200:
            return {"status": "failure", "reason": f"HTTP {r.status_code}"}
        return r.json()
    except requests.exceptions.Timeout:
        return {"status": "failure", "reason": "Request timed out."}
    except requests.exceptions.RequestException as e:
        return {"status": "failure", "reason": str(e)}


def overs_str_to_float(o: Any) -> float:
    if o is None:
        return 0.0
    s = str(o).strip()
    if not s:
        return 0.0
    if "." in s:
        whole, frac = s.split(".", 1)
        try:
            return int(whole or 0) + int(frac[:1] or 0) / 6.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def normalize_team_to_dataset(api_name: str, candidates: List[str]) -> str:
    """Map CricAPI team label to a name from master_features team list."""
    if not api_name:
        return ""
    n = api_name.strip().lower()
    for c in candidates:
        if c.lower() == n:
            return c
    for c in candidates:
        cl = c.lower()
        if n in cl or cl in n:
            return c
    # Title-style fallback aligned with dataset (mostly lowercase in CSV)
    return api_name.strip().lower()


def normalize_venue_to_dataset(api_venue: str, candidates: List[str]) -> str:
    if not api_venue:
        return ""
    flat = re.sub(r"\s+", " ", api_venue.lower().replace(".", " "))
    for c in candidates:
        cs = c.lower().replace(".", " ")
        if c.lower() in flat or flat in c.lower():
            return c
        # prefix match on first 3 words
        words = [w for w in re.split(r"[\s,]+", flat) if len(w) > 2]
        cw = [w for w in re.split(r"[\s,]+", cs) if len(w) > 2]
        if words and cw and words[0] == cw[0] and (words[1] == cw[1] if len(words) > 1 and len(cw) > 1 else True):
            return c
    return api_venue.strip()


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
        "overs": round(overs_str_to_float(inn.get("o", 0)), 2),
    }


def _batters_bowler_from_match(m: dict) -> Tuple[str, str, str]:
    """Best-effort from CricAPI fields; structure varies by match state."""
    striker = non_striker = bowler_name = ""
    # Some responses nest batting info under score entries
    for inn in reversed(m.get("score") or []):
        bats = inn.get("batsman") or inn.get("batsmen") or []
        if isinstance(bats, list) and bats:
            for b in bats:
                if not isinstance(b, dict):
                    continue
                nm = _player_cell(b)
                st = (b.get("active") or b.get("strike") or "").lower()
                if st in ("true", "y", "yes", "striker", "s") or b.get("isStrike"):
                    striker = nm
                elif not non_striker:
                    non_striker = nm
        bw = inn.get("bowler") or inn.get("bowlers")
        if isinstance(bw, list) and bw:
            bowler_name = _player_cell(bw[0] if isinstance(bw[0], dict) else bw[0])
        elif isinstance(bw, dict):
            bowler_name = _player_cell(bw)
        if striker or bowler_name:
            break
    return striker, non_striker, bowler_name


def _normalize_bbb_items(raw: Any) -> List[dict]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                {
                    "over": item.get("o") or item.get("over"),
                    "ball": item.get("b") or item.get("ball"),
                    "runs": item.get("r") or item.get("runs"),
                    "batsman": item.get("bat") or item.get("batsman") or item.get("batsmanName"),
                    "bowler": item.get("bow") or item.get("bowler") or item.get("bowlerName"),
                    "comment": item.get("txt") or item.get("commentary") or item.get("c"),
                    "isWicket": item.get("w") or item.get("isWicket"),
                }
            )
        else:
            out.append({"raw": str(item)[:200]})
    return out


def fetch_match_bbb(api_key: str, match_id: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    data = _request("match_bbb", api_key, {"id": match_id})
    if data.get("status") != "success":
        reason = data.get("reason") or data.get("info") or "BBB unavailable"
        return None, str(reason)
    bbb = data.get("data")
    if isinstance(bbb, list):
        return _normalize_bbb_items(bbb), None
    if isinstance(bbb, dict) and "balls" in bbb:
        return _normalize_bbb_items(bbb.get("balls")), None
    if isinstance(bbb, dict):
        return _normalize_bbb_items(bbb.get("bbb") or bbb.get("ballByBall")), None
    return [], None


def map_players_through_name_map(names: List[str]) -> List[str]:
    try:
        from name_resolver import resolve_name
    except ImportError:
        return names
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


def build_todays_match_payload(
    m: dict,
    match_id: str,
    *,
    dataset_teams: List[str],
    dataset_venues: List[str],
    ball_by_ball: Optional[List[dict]] = None,
) -> dict:
    teams = m.get("teams") or []
    t1_raw = teams[0] if len(teams) > 0 else ""
    t2_raw = teams[1] if len(teams) > 1 else ""
    team1 = normalize_team_to_dataset(t1_raw, dataset_teams) or (t1_raw or "").lower()
    team2 = normalize_team_to_dataset(t2_raw, dataset_teams) or (t2_raw or "").lower()

    t1xi, t2xi = _extract_xis(m, team1, team2)
    t1xi = map_players_through_name_map(t1xi)
    t2xi = map_players_through_name_map(t2xi)

    venue_norm = normalize_venue_to_dataset(m.get("venue") or "", dataset_venues)

    tw = m.get("tossWinner") or ""
    td = m.get("tossChoice") or ""
    toss_winner_norm = normalize_team_to_dataset(tw, dataset_teams) if tw else ""
    toss_winner_norm = toss_winner_norm or (tw or "")

    striker, non_striker, bowler_name = _batters_bowler_from_match(m)
    striker = map_players_through_name_map([striker])[0] if striker else ""
    non_striker = map_players_through_name_map([non_striker])[0] if non_striker else ""
    bowler_resolved = map_players_through_name_map([bowler_name])[0] if bowler_name else ""

    return {
        "match_id": match_id,
        "match_name": m.get("name") or "",
        "match_date": m.get("date") or "",
        "league": "IPL",
        "venue": venue_norm or (m.get("venue") or ""),
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner_norm if m.get("tossWinner") else "",
        "toss_decision": (td or "").lower() if td else "",
        "team1_xi": t1xi,
        "team2_xi": t2xi,
        "current_score": _current_score_from_match(m),
        "batters": {"striker": striker, "non_striker": non_striker},
        "bowler": bowler_resolved,
        "ball_by_ball": ball_by_ball or [],
        "match_status": m.get("status") or "",
        "bbb_enabled": bool(m.get("bbbEnabled")),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": "cricdata",
    }


def fetch_live_match_for_dashboard(
    match_id: str,
    *,
    api_key: Optional[str] = None,
    dataset_teams: List[str],
    dataset_venues: List[str],
    include_bbb: bool = True,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    One match_info call; optional second match_bbb call only if bbbEnabled (saves quota).
    Returns (payload dict, error_message).
    """
    mid = (match_id or "").strip()
    if not mid:
        return None, "match_id is required"
    key = (api_key or get_api_key() or "").strip()
    if not key:
        return None, "Set CRICAPI_KEY or configure API_KEY in espn_live_scraper/app.py"

    data = _request("match_info", key, {"id": mid})
    if data.get("status") != "success":
        return None, data.get("reason") or data.get("info") or "match_info failed"
    m = data.get("data") or {}
    if not m:
        return None, "Empty match payload"

    bbb_list: List[dict] = []
    bbb_err = None
    if include_bbb and m.get("bbbEnabled"):
        bbb_list, bbb_err = fetch_match_bbb(key, mid)
        if bbb_list is None:
            bbb_list = []
    payload = build_todays_match_payload(
        m, mid, dataset_teams=dataset_teams, dataset_venues=dataset_venues, ball_by_ball=bbb_list
    )
    if bbb_err and not bbb_list:
        payload["ball_by_ball_note"] = bbb_err
    return payload, None


def merge_manual_fields(saved: dict, incoming: dict) -> dict:
    """Preserve manual pitch/dew overrides when refreshing live data."""
    manual_keys = ("pitch_type", "dew_expected")
    for k in manual_keys:
        if k in saved and saved.get(k) not in (None, "", []):
            incoming[k] = saved[k]
    return incoming


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

