from flask import Flask, render_template, request, jsonify
from scraper import fetch_live_matches, fetch_match_info, fetch_match_score

app = Flask(__name__)

# ── CricAPI key (free plan: 100 req/day) ─────────────────────────────────────
# Sign up at https://cricapi.com/ to get your free key
API_KEY = "af68909b-997d-4d4a-988e-857215d852df"


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


if __name__ == "__main__":
    print("=" * 55)
    print("  CricTrack — Live IPL 2026 Dashboard")
    print("  Running on: http://127.0.0.1:5000")
    print("  ESPN scraping replaced with CricAPI ✅")
    print("=" * 55)
    app.run(debug=True)