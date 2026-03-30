"""
PITCHMIND — STEP 5: Live Match Data Fetcher
============================================
Run this script ~1 hour before match (at toss time = 6:30 PM IST).
ESPN scraping replaced with CricAPI (free, 100 req/day, no 403 errors).

Usage:
    python 5_live_data_fetch.py

Output:
    data/live/todays_match.json
"""

import os
import json
import requests
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────────────
LIVE_DIR = os.path.join("data", "live")
OUTPUT   = os.path.join(LIVE_DIR, "todays_match.json")
os.makedirs(LIVE_DIR, exist_ok=True)

# ── CRICAPI KEY ───────────────────────────────────────────────────────────────
# Free plan: 100 req/day — sign up at https://cricapi.com/
CRICAPI_KEY = "af68909b-997d-4d4a-988e-857215d852df"

# ── VENUE COORDINATES (for weather) ───────────────────────────────────────────
VENUE_COORDS = {
    "Eden Gardens"                      : (22.5645,  88.3433),
    "Wankhede Stadium"                  : (18.9388,  72.8258),
    "Narendra Modi Stadium"             : (23.0902,  72.0851),
    "MA Chidambaram Stadium"            : (13.0633,  80.2790),
    "Rajiv Gandhi International Stadium": (17.4065,  78.5464),
    "M Chinnaswamy Stadium"             : (12.9784,  77.5996),
    "DY Patil Stadium"                  : (19.0579,  72.9980),
    "Brabourne Stadium"                 : (18.9333,  72.8278),
    "Punjab Cricket Association Stadium": (30.6780,  76.8566),
    "Sawai Mansingh Stadium"            : (26.8947,  75.8069),
    "BRSABV Ekana Cricket Stadium"      : (26.8717,  80.9520),
    "Arun Jaitley Stadium"              : (28.6366,  77.2411),
}

# ── TODAY'S MATCH CONFIG ──────────────────────────────────────────────────────
# Fill these before running ~1 hour before match.
#
# How to find cricapi_match_id:
#   1. Run the espn_live_scrapper Flask app:  python espn_live_scrapper/app.py
#   2. Open http://127.0.0.1:5000 → IPL Schedule tab → LOAD SCHEDULE
#   3. Click "COPY ID" next to today's match — paste it below.
#
# OR run this in terminal to see all live/upcoming matches:
#   python -c "
#   import requests, json
#   r = requests.get('https://api.cricapi.com/v1/currentMatches',
#       params={'apikey': 'af68909b-997d-4d4a-988e-857215d852df', 'offset': 0})
#   for m in r.json().get('data', []):
#       print(m['id'], '|', m['name'])
#   "

MATCH_CONFIG = {
    "team1"               : "royal challengers bengaluru",
    "team2"               : "sunrisers hyderabad",
    "match_name"          : "IPL 2026 — use Cricdata fetch in dashboard for live name",
    "venue_name"        : "M Chinnaswamy Stadium",
    "match_date"        : "2026-03-28",
    "league"            : "IPL",
    "cricapi_match_id"  : "55fe0f15-6eb0-4ad5-835b-5564be4f6a21",
    "openweather_api_key": None,
}


# ── CRICAPI: Fetch toss + playing XI ─────────────────────────────────────────
def fetch_cricapi(match_id: str) -> dict | None:
    """
    Fetch toss winner, toss decision, and playing XIs from CricAPI.
    Replaces the old fetch_espncricinfo() which returned 403.

    Endpoint: GET https://api.cricapi.com/v1/match_info?apikey=...&id=...
    Free plan: counts as 1 API hit.
    """
    if not match_id:
        print("⚠️  No CricAPI match ID set in MATCH_CONFIG['cricapi_match_id']")
        print("    → Open espn_live_scrapper app, go to IPL Schedule, copy the match ID.")
        return None

    print(f"🔄 Fetching match info from CricAPI for ID: {match_id}")

    try:
        response = requests.get(
            "https://api.cricapi.com/v1/match_info",
            params={"apikey": CRICAPI_KEY, "id": match_id},
            timeout=10,
        )

        if response.status_code == 429:
            print("❌ CricAPI rate limit hit (100 req/day). Try again tomorrow.")
            return None
        if response.status_code == 401:
            print("❌ Invalid CricAPI key. Check CRICAPI_KEY in this file.")
            return None
        if response.status_code != 200:
            print(f"❌ CricAPI returned HTTP {response.status_code}")
            return None

        data = response.json()
        if data.get("status") != "success":
            print(f"❌ CricAPI error: {data.get('info', 'Unknown error')}")
            return None

        m = data.get("data", {})
        if not m:
            print("❌ No match data returned for this ID.")
            return None

        # ── Extract toss ──────────────────────────────────────────────────────
        toss_winner   = m.get("tossWinner")   or None
        toss_decision = m.get("tossChoice")   or None   # "bat" or "field"

        # ── Extract playing XIs ───────────────────────────────────────────────
        # CricAPI returns teamInfo list with players array when available
        team_info = m.get("teamInfo", [])
        teams     = m.get("teams", [])

        team1_xi, team2_xi = [], []

        for ti in team_info:
            name    = ti.get("name", "")
            players = ti.get("players", [])   # list of player name strings

            if not players:
                continue

            # Match to team1 or team2 by partial name
            if MATCH_CONFIG["team1"].lower() in name.lower():
                team1_xi = [p.get("name", p) if isinstance(p, dict) else str(p)
                            for p in players]
            elif MATCH_CONFIG["team2"].lower() in name.lower():
                team2_xi = [p.get("name", p) if isinstance(p, dict) else str(p)
                            for p in players]

        # ── Fallback: try squads field if teamInfo has no players ─────────────
        if not team1_xi and not team2_xi:
            squads = m.get("squads", [])
            for sq in squads:
                name    = sq.get("name", "")
                players = sq.get("players", [])
                if MATCH_CONFIG["team1"].lower() in name.lower():
                    team1_xi = [p.get("name", p) if isinstance(p, dict) else str(p)
                                for p in players]
                elif MATCH_CONFIG["team2"].lower() in name.lower():
                    team2_xi = [p.get("name", p) if isinstance(p, dict) else str(p)
                                for p in players]

        # ── Print summary ─────────────────────────────────────────────────────
        print(f"✅ CricAPI fetch successful")
        print(f"   Match  : {m.get('name', '—')}")
        print(f"   Toss   : {toss_winner or '—'} chose to {toss_decision or '—'}")
        print(f"   {MATCH_CONFIG['team1']} XI : {len(team1_xi)} players found")
        print(f"   {MATCH_CONFIG['team2']} XI : {len(team2_xi)} players found")

        if not toss_winner:
            print("   ⚠️  Toss not available yet — fill manually via dashboard")
        if not team1_xi and not team2_xi:
            print("   ⚠️  Playing XIs not available yet — fill manually via dashboard")

        return {
            "toss_winner"   : toss_winner,
            "toss_decision" : toss_decision,
            "team1_xi"      : team1_xi,
            "team2_xi"      : team2_xi,
            "source"        : "cricapi_auto",
        }

    except requests.exceptions.Timeout:
        print("❌ CricAPI request timed out. Check your internet and retry.")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to CricAPI. Check your internet connection.")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


# ── WEATHER (unchanged — OpenWeatherMap still works fine) ─────────────────────
def fetch_weather(venue_name: str, api_key: str) -> dict:
    """
    Gets humidity + dew point from OpenWeatherMap for the venue.
    Free tier = 1000 calls/day — sign up at openweathermap.org.
    """
    coords = VENUE_COORDS.get(venue_name)
    if not coords:
        print(f"⚠️  No coordinates for venue '{venue_name}' — skipping weather")
        return {}
    if not api_key:
        print("⚠️  No OpenWeatherMap API key — skipping weather")
        return {}

    lat, lon = coords
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        w    = r.json()
        main = w.get("main", {})
        weather_data = {
            "temperature_c"    : main.get("temp"),
            "humidity_pct"     : main.get("humidity"),
            "feels_like_c"     : main.get("feels_like"),
            "description"      : w.get("weather", [{}])[0].get("description", ""),
            "dew_point_approx" : round(
                main.get("temp", 0) - ((100 - main.get("humidity", 0)) / 5), 1
            ),
        }
        print(f"✅ Weather: {weather_data['temperature_c']}°C, "
              f"Humidity: {weather_data['humidity_pct']}%, "
              f"Dew point: ~{weather_data['dew_point_approx']}°C")
        return weather_data
    except Exception as e:
        print(f"❌ Weather fetch failed: {e}")
        return {}


# ── BUILD todays_match.json (IPL / Cricdata schema) ───────────────────────────
def build_live_json(auto_data: dict | None, weather_data: dict) -> dict:
    """
    Output matches `cricdata_live.build_todays_match_payload` / dashboard load_live_match.
    Optional weather is stored only if the key exists and is non-empty (OpenWeather).
    """
    mid = MATCH_CONFIG.get("cricapi_match_id") or ""
    payload = {
        "match_id"       : mid,
        "match_name"     : MATCH_CONFIG["match_name"],
        "match_date"     : MATCH_CONFIG["match_date"],
        "league"         : MATCH_CONFIG.get("league", "IPL"),
        "venue"          : MATCH_CONFIG["venue_name"],
        "team1"          : MATCH_CONFIG["team1"],
        "team2"          : MATCH_CONFIG["team2"],
        "toss_winner"    : (auto_data.get("toss_winner") if auto_data else None) or "",
        "toss_decision"  : (auto_data.get("toss_decision") if auto_data else None) or "",
        "team1_xi"       : (auto_data.get("team1_xi", []) if auto_data else []),
        "team2_xi"       : (auto_data.get("team2_xi", []) if auto_data else []),
        "current_score"  : {"runs": 0, "wickets": 0, "overs": 0.0},
        "batters"        : {"striker": "", "non_striker": ""},
        "bowler"         : "",
        "ball_by_ball"   : [],
        "match_status"   : "",
        "last_updated"   : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source"    : "cricdata" if auto_data else "manual",
    }
    if weather_data:
        payload["weather"] = weather_data
    return payload


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PITCHMIND — Live Data Fetcher (CricAPI Edition)")
    print(f"  {MATCH_CONFIG.get('league', 'IPL')} | match_id={MATCH_CONFIG.get('cricapi_match_id')}")
    print(f"  {MATCH_CONFIG['match_name']}")
    print(f"  {MATCH_CONFIG['match_date']}")
    print("=" * 60 + "\n")

    # Step 1: Fetch from CricAPI (replaces broken ESPN fetch)
    auto_data    = fetch_cricapi(MATCH_CONFIG["cricapi_match_id"])

    # Step 2: Fetch weather (unchanged)
    weather_data = fetch_weather(
        MATCH_CONFIG["venue_name"],
        MATCH_CONFIG["openweather_api_key"]
    )

    # Step 3: Build JSON in exact format 4_dashboard.py expects
    live = build_live_json(auto_data, weather_data)

    # Step 4: Save to data/live/todays_match.json
    with open(OUTPUT, "w") as f:
        json.dump(live, f, indent=2)

    print(f"\n✅ Saved → {OUTPUT}")

    # Step 5: Report what still needs manual filling in dashboard
    missing = [k for k, v in live.items()
               if v is None or v == [] or v == ""]
    if missing:
        print(f"\n⚠️  Fields to fill manually via dashboard (Live Match tab):")
        for field in missing:
            print(f"   → {field}")
        print(f"\n   Open the dashboard: streamlit run 4_dashboard.py")
        print(f"   Go to: 🔴 Live Match → Manual Entry → Save Field Overrides")
    else:
        print("\n🎉 All fields populated automatically!")

    print("\n" + "=" * 60)
    print("  Next → streamlit run 4_dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()