"""
scraper.py — CricAPI data fetcher
===================================
Replaces the old ESPN scraper (which returned 403 errors).
Uses CricAPI v1 (free plan: 100 req/day).
Sign up: https://cricapi.com/
"""

import requests
try:
    from .cache_manager import save_matches_to_cache, load_matches_from_cache
except ImportError:
    from cache_manager import save_matches_to_cache, load_matches_from_cache

BASE_URL = "https://api.cricapi.com/v1"


def _get(endpoint: str, params: dict) -> dict:
    """
    Central request handler.
    All API calls go through here so errors are handled consistently.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/{endpoint}",
            params=params,
            timeout=10
        )

        if response.status_code == 429:
            return {
                "error": (
                    "Rate limit exceeded. Free plan allows ~100 requests/day. "
                    "Try again tomorrow or upgrade at cricapi.com"
                )
            }
        if response.status_code == 401:
            return {"error": "Invalid API key. Get a free key at https://cricapi.com/"}
        if response.status_code != 200:
            return {"error": f"API request failed with status {response.status_code}"}

        data = response.json()

        # CricAPI always returns a status field
        if data.get("status") != "success":
            return {"error": data.get("info", "Unknown API error")}

        return data

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to CricAPI. Check your internet connection."}
    except Exception as e:
        return {"error": str(e)}


def fetch_live_matches(api_key: str) -> dict:
    """
    Fetch all currently live cricket matches.
    Endpoint: GET /v1/currentMatches
    """
    data = _get("currentMatches", {"apikey": api_key, "offset": 0})
    if "error" in data:
        return data

    matches = data.get("data", [])
    live = []

    for m in matches:
        # Only include matches that have started but not ended
        if m.get("matchStarted") and not m.get("matchEnded"):
            teams     = m.get("teams", [])
            score     = m.get("score", [])

            live.append({
                "id"        : m.get("id"),
                "name"      : m.get("name"),
                "status"    : m.get("status"),
                "venue"     : m.get("venue"),
                "date"      : m.get("date"),
                "matchType" : m.get("matchType"),
                "teams"     : teams,
                "score"     : score,
                "teamInfo"  : m.get("teamInfo", []),
            })

    return {
        "live_count"     : len(live),
        "matches"        : live,
        "api_hits_today" : data.get("info", {}).get("hitsToday", "N/A"),
        "api_hits_limit" : data.get("info", {}).get("hitsLimit", 100),
    }


def fetch_match_score(api_key: str, match_id: str) -> dict:
    """
    Fetch live/completed scorecard for a specific match.
    Endpoint: GET /v1/match_info
    """
    data = _get("match_info", {"apikey": api_key, "id": match_id})
    if "error" in data:
        return data

    m = data.get("data", {})
    if not m:
        return {"error": "No match data found for this ID"}

    score_list = m.get("score", [])
    formatted_scores = []
    for innings in score_list:
        formatted_scores.append({
            "inning"    : innings.get("inning", "N/A"),
            "runs"      : innings.get("r", 0),
            "wickets"   : innings.get("w", 0),
            "overs"     : innings.get("o", "0.0"),
            "score_str" : f"{innings.get('r', 0)}/{innings.get('w', 0)} ({innings.get('o', '0.0')} ov)"
        })

    return {
        "id"           : m.get("id"),
        "name"         : m.get("name"),
        "status"       : m.get("status"),
        "venue"        : m.get("venue"),
        "date"         : m.get("date"),
        "matchType"    : m.get("matchType"),
        "teams"        : m.get("teams", []),
        "teamInfo"     : m.get("teamInfo", []),
        "scores"       : formatted_scores,
        "tossWinner"   : m.get("tossWinner"),
        "tossChoice"   : m.get("tossChoice"),
        "matchWinner"  : m.get("matchWinner"),
        "matchStarted" : m.get("matchStarted"),
        "matchEnded"   : m.get("matchEnded"),
    }


def fetch_match_info(api_key: str, series_id: str, use_cache: bool = True) -> dict:
    """
    Fetch full series info + all match list.
    Endpoint: GET /v1/series_info

    Args:
        api_key: CricAPI key
        series_id: Series ID to fetch
        use_cache: If True, load from cache first. If cache misses, fetch & save to cache.
    """
    # Try cache first
    if use_cache:
        cached_data = load_matches_from_cache()
        if cached_data:
            # Return from cache with a note
            return {
                "series": {
                    "name"          : "IPL 2026",
                    "start_date"    : "2026-03-27",
                    "end_date"      : "2026-06-01",
                    "total_matches" : cached_data.get("total_matches"),
                    "source"        : "cache",
                },
                "total_matches"  : cached_data.get("total_matches"),
                "matches"        : cached_data.get("matches", []),
                "api_hits_today" : "N/A",
                "api_hits_limit" : 100,
                "from_cache"     : True,
                "cached_at"      : cached_data.get("fetched_at"),
            }

    # Fetch fresh from API
    data = _get("series_info", {"apikey": api_key, "id": series_id})
    if "error" in data:
        return data

    series_data  = data.get("data", {})
    info         = series_data.get("info", {})
    match_list   = series_data.get("matchList", [])

    # Sort matches chronologically
    match_list_sorted = sorted(
        match_list,
        key=lambda x: x.get("dateTimeGMT", "")
    )

    # Save to cache
    save_matches_to_cache(match_list_sorted)

    return {
        "series": {
            "name"          : info.get("name"),
            "start_date"    : info.get("startdate"),
            "end_date"      : info.get("enddate"),
            "total_matches" : info.get("matches"),
            "squads"        : info.get("squads"),
            "t20"           : info.get("t20"),
        },
        "total_matches"  : len(match_list_sorted),
        "matches"        : match_list_sorted,
        "api_hits_today" : data.get("info", {}).get("hitsToday", "N/A"),
        "api_hits_limit" : data.get("info", {}).get("hitsLimit", 100),
        "from_cache"     : False,
    }