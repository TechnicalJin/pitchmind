import os
import sys

from flask import Flask, render_template, request, jsonify
from scraper import fetch_live_matches, fetch_match_info, fetch_match_score

# Repo root (parent of espn_live_scraper/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cricdata_live import fetch_live_match_for_dashboard  # noqa: E402

app = Flask(__name__)

# ── CricAPI key (free plan: 100 req/day) ─────────────────────────────────────
# Sign up at https://cricapi.com/ to get your free key
API_KEY = "af68909b-997d-4d4a-988e-857215d852df"


def _dataset_teams_venues():
    try:
        import pandas as pd
        path = os.path.join(_REPO_ROOT, "data", "master_features.csv")
        if not os.path.exists(path):
            return [], []
        df = pd.read_csv(path)
        teams = sorted(set(df["team1"].unique()) | set(df["team2"].unique()))
        venues = sorted(df["venue"].unique())
        return [str(t) for t in teams], [str(v) for v in venues]
    except Exception:
        return [], []


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/live", methods=["GET"])
def live():
    """Fetch all currently live matches via CricAPI"""
    data = fetch_live_matches(API_KEY)
    return jsonify(data)


@app.route("/score", methods=["POST"])
def score():
    """Fetch score for a specific match by match ID"""
    match_id = request.form.get("match_id")
    if not match_id:
        return jsonify({"error": "No match ID provided"})
    data = fetch_match_score(API_KEY, match_id)
    return jsonify(data)


@app.route("/series", methods=["GET"])
def series():
    """Fetch IPL 2026 series info with all matches"""
    # IPL 2026 series ID on CricAPI
    series_id = "87c62aac-bc3c-4738-ab93-19da0690488f"
    fresh = request.args.get("fresh", "false").lower() == "true"
    data = fetch_match_info(API_KEY, series_id, use_cache=(not fresh))
    return jsonify(data)


@app.route("/api/match/live", methods=["GET"])
def api_match_live():
    """
    GET /api/match/live?matchId=<uuid>
    Manual fetch only (no polling). Maps CricAPI match_info (+ optional BBB) to todays_match schema.
    """
    match_id = (request.args.get("matchId") or request.args.get("match_id") or "").strip()
    if not match_id:
        return jsonify({"ok": False, "error": "matchId query parameter is required"}), 400

    want_bbb = request.args.get("bbb", "true").lower() in ("1", "true", "yes")
    teams, venues = _dataset_teams_venues()
    payload, err = fetch_live_match_for_dashboard(
        match_id,
        api_key=API_KEY,
        dataset_teams=teams,
        dataset_venues=venues,
        include_bbb=want_bbb,
    )
    if err:
        return jsonify({"ok": False, "error": err}), 502
    return jsonify({"ok": True, "data": payload})


if __name__ == "__main__":
    print("=" * 55)
    print("  CricTrack — Live IPL 2026 Dashboard")
    print("  Running on: http://127.0.0.1:5000")
    print("  ESPN scraping replaced with CricAPI ✅")
    print("=" * 55)
    app.run(debug=True)