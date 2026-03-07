"""
PITCHMIND — STEP 5: Live Match Data Fetcher
============================================
Run this script ~1 hour before match (at toss time = 6:30 PM IST).
It tries to auto-fetch toss, playing XI, and weather.
If scraping fails, it creates a template JSON for you to fill manually.

Usage:
    python 5_live_data_fetch.py

Output:
    data/live/todays_match.json
"""

import os
import json
import requests
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
LIVE_DIR   = os.path.join("data", "live")
OUTPUT     = os.path.join(LIVE_DIR, "todays_match.json")
os.makedirs(LIVE_DIR, exist_ok=True)

# ── VENUE COORDINATES ─────────────────────────────────────────────────────────
# Add/update as needed for the match venue
VENUE_COORDS = {
    "Eden Gardens"             : (22.5645, 88.3433),
    "Wankhede Stadium"         : (18.9388, 72.8258),
    "Narendra Modi Stadium"    : (23.0902, 72.0851),
    "MA Chidambaram Stadium"   : (13.0633, 80.2790),
    "Rajiv Gandhi International Stadium": (17.4065, 78.5464),
    "M Chinnaswamy Stadium"    : (12.9784, 77.5996),
    "DY Patil Stadium"         : (19.0579, 72.9980),
    "Brabourne Stadium"        : (18.9333, 72.8278),
    # T20 World Cup venues
    "Nassau County International": (40.7128, -73.9060),
    "Kensington Oval"          : (13.0965, -59.6147),
    "Daren Sammy Stadium"      : (14.0167, -60.9833),
    "Providence Stadium"       : (6.8013,  -58.1547),
    "Arnos Vale Ground"        : (13.1400, -61.2100),
    # IND vs NZ T20 WC Final 2024 — Eden Gardens was venue
    # Update this to the actual venue below
}

# TODAY'S MATCH — UPDATE THESE BEFORE RUNNING
MATCH_CONFIG = {
    "team1"      : "India",
    "team2"      : "New Zealand",
    "match_name" : "IND vs NZ — T20 World Cup Final",
    "venue_name" : "Narendra Modi Stadium",   # ← Ahmedabad, NOT Eden Gardens
    "match_date" : "2026-03-08",
    "series"     : "ICC Men's T20 World Cup 2025-26",
    "espncricinfo_match_id": 1512773,          # ← from your URL
    "openweather_api_key": None,               # add yours if you have it
}

# ── ATTEMPT 1: ESPNcricinfo unofficial JSON API ───────────────────────────────
def fetch_espncricinfo(match_id):
    """
    Fetches match data from ESPNcricinfo's unofficial match JSON endpoint.
    Returns dict with toss + playing XIs if available, else None.
    """
    if not match_id:
        print("⚠️  No ESPNcricinfo match ID set — skipping auto-fetch")
        return None

    url = f"https://www.espncricinfo.com/matches/engine/match/{match_id}.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()

        match_info = data.get("match", {})
        innings    = data.get("innings", [])

        # Extract toss
        toss_winner   = match_info.get("toss_winner_team_id")
        toss_decision = match_info.get("toss_decision", "")

        # Extract team names
        teams = {
            str(match_info.get("team1_id")): match_info.get("team1_name", ""),
            str(match_info.get("team2_id")): match_info.get("team2_name", ""),
        }

        # Extract playing XIs
        def get_xi(team_players):
            return [p.get("known_as", p.get("name_full", "")) for p in team_players]

        team1_xi = []
        team2_xi = []
        for team_id, players in data.get("team", {}).items():
            name = teams.get(str(team_id), "")
            xi = get_xi(players.get("player", []))
            if MATCH_CONFIG["team1"].lower() in name.lower():
                team1_xi = xi
            else:
                team2_xi = xi

        print(f"✅ ESPNcricinfo fetch successful")
        print(f"   Toss: {toss_winner} chose to {toss_decision}")
        print(f"   {MATCH_CONFIG['team1']} XI: {len(team1_xi)} players")
        print(f"   {MATCH_CONFIG['team2']} XI: {len(team2_xi)} players")

        return {
            "toss_winner"   : toss_winner,
            "toss_decision" : toss_decision,
            "team1_xi"      : team1_xi,
            "team2_xi"      : team2_xi,
            "source"        : "espncricinfo_auto",
        }

    except Exception as e:
        print(f"❌ ESPNcricinfo fetch failed: {e}")
        return None


# ── ATTEMPT 2: Weather ────────────────────────────────────────────────────────
def fetch_weather(venue_name, api_key):
    """
    Gets humidity + dew point from OpenWeatherMap for the venue.
    Free tier = 1000 calls/day, sign up at openweathermap.org.
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
        w = r.json()
        main = w.get("main", {})
        weather_data = {
            "temperature_c"   : main.get("temp"),
            "humidity_pct"    : main.get("humidity"),
            "feels_like_c"    : main.get("feels_like"),
            "description"     : w.get("weather", [{}])[0].get("description", ""),
            "dew_point_approx": round(main.get("temp", 0) - ((100 - main.get("humidity", 0)) / 5), 1),
        }
        print(f"✅ Weather: {weather_data['temperature_c']}°C, "
              f"Humidity: {weather_data['humidity_pct']}%, "
              f"Dew point: ~{weather_data['dew_point_approx']}°C")
        return weather_data
    except Exception as e:
        print(f"❌ Weather fetch failed: {e}")
        return {}


# ── BUILD LIVE JSON ────────────────────────────────────────────────────────────
def build_live_json(auto_data, weather_data):
    """
    Merges auto-fetched data with template.
    Any field left as None = fill manually in the JSON or via dashboard.
    """
    live = {
        # Match identity
        "match_name"     : MATCH_CONFIG["match_name"],
        "match_date"     : MATCH_CONFIG["match_date"],
        "series"         : MATCH_CONFIG["series"],
        "venue"          : MATCH_CONFIG["venue_name"],
        "team1"          : MATCH_CONFIG["team1"],
        "team2"          : MATCH_CONFIG["team2"],

        # Toss — from auto fetch OR fill manually
        "toss_winner"    : auto_data.get("toss_winner")   if auto_data else None,
        "toss_decision"  : auto_data.get("toss_decision") if auto_data else None,

        # Playing XIs — from auto fetch OR fill manually
        "team1_xi"       : auto_data.get("team1_xi", [])  if auto_data else [],
        "team2_xi"       : auto_data.get("team2_xi", [])  if auto_data else [],

        # Weather
        "weather"        : weather_data,

        # Manual context (filled via dashboard UI)
        "pitch_type"     : None,   # "Good" / "Dry" / "Seamer" / "Flat"
        "dew_expected"   : None,   # "Yes-heavy" / "Yes-light" / "No" / "Unknown"
        "expert_notes"   : "",     # paste from broadcast/commentary

        # Metadata
        "data_source"    : auto_data.get("source", "manual") if auto_data else "manual",
        "fetched_at"     : datetime.now().isoformat(),
    }
    return live


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PITCHMIND — Live Data Fetcher")
    print(f"  {MATCH_CONFIG['match_name']}")
    print(f"  {MATCH_CONFIG['match_date']}")
    print("=" * 60 + "\n")

    # Try auto-fetch
    auto_data    = fetch_espncricinfo(MATCH_CONFIG["espncricinfo_match_id"])
    weather_data = fetch_weather(MATCH_CONFIG["venue_name"], MATCH_CONFIG["openweather_api_key"])

    # Build JSON
    live = build_live_json(auto_data, weather_data)

    # Save
    with open(OUTPUT, "w") as f:
        json.dump(live, f, indent=2)

    print(f"\n✅ Saved → {OUTPUT}")

    # Show what needs manual filling
    missing = [k for k, v in live.items() if v is None or v == [] or v == ""]
    if missing:
        print(f"\n⚠️  Fields needing manual fill in dashboard or JSON file:")
        for field in missing:
            print(f"   → {field}")
        print(f"\n   Open {OUTPUT} and fill these in,")
        print(f"   OR use the manual entry panel in the dashboard.")
    else:
        print("\n🎉 All fields populated automatically!")

    print("\n" + "=" * 60)
    print("  Next → streamlit run 4_dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()