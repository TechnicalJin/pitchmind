"""
Cache Manager — Save & Load Match Data
=======================================
Saves all IPL 2026 matches to JSON to avoid API rate limits (100 req/day).
After first load, all matches are cached and can be shown without API calls.
"""

import json
import os
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MATCHES_CACHE = CACHE_DIR / "ipl_2026_matches.json"
CACHE_METADATA = CACHE_DIR / "cache_metadata.json"


def save_matches_to_cache(matches: list) -> bool:
    """Save fetched matches to JSON cache file."""
    try:
        cache_data = {
            "fetched_at": datetime.now().isoformat(),
            "total_matches": len(matches),
            "matches": matches
        }

        with open(MATCHES_CACHE, "w") as f:
            json.dump(cache_data, f, indent=2)

        # Update metadata
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "cached_matches": len(matches),
            "source": "CricAPI"
        }
        with open(CACHE_METADATA, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Cached {len(matches)} matches to {MATCHES_CACHE}")
        return True
    except Exception as e:
        print(f"❌ Failed to save cache: {e}")
        return False


def load_matches_from_cache() -> dict | None:
    """Load matches from cache. Returns None if cache doesn't exist."""
    try:
        if not MATCHES_CACHE.exists():
            print(f"⚠️  Cache file not found: {MATCHES_CACHE}")
            return None

        with open(MATCHES_CACHE, "r") as f:
            data = json.load(f)

        print(f"✅ Loaded {data.get('total_matches', 0)} matches from cache")
        return data
    except Exception as e:
        print(f"❌ Failed to load cache: {e}")
        return None


def get_cached_match_ids() -> dict:
    """Get all match IDs from cache as {match_id: match_name}."""
    data = load_matches_from_cache()
    if not data:
        return {}

    ids = {}
    for match in data.get("matches", []):
        ids[match.get("id")] = match.get("name", "Unknown")

    return ids


def is_cache_fresh(max_age_hours: int = 12) -> bool:
    """Check if cache is less than max_age_hours old."""
    if not CACHE_METADATA.exists():
        return False

    try:
        with open(CACHE_METADATA, "r") as f:
            metadata = json.load(f)

        last_updated = datetime.fromisoformat(metadata.get("last_updated", ""))
        age = (datetime.now() - last_updated).total_seconds() / 3600

        return age < max_age_hours
    except:
        return False


def clear_cache():
    """Clear the cache files."""
    try:
        if MATCHES_CACHE.exists():
            MATCHES_CACHE.unlink()
        if CACHE_METADATA.exists():
            CACHE_METADATA.unlink()
        print("✅ Cache cleared")
        return True
    except Exception as e:
        print(f"❌ Failed to clear cache: {e}")
        return False


def get_cache_status() -> dict:
    """Get current cache status."""
    if not MATCHES_CACHE.exists():
        return {
            "has_cache": False,
            "message": "No cache file found"
        }

    try:
        with open(CACHE_METADATA, "r") as f:
            metadata = json.load(f)

        return {
            "has_cache": True,
            "last_updated": metadata.get("last_updated"),
            "cached_matches": metadata.get("cached_matches", 0),
            "source": metadata.get("source")
        }
    except:
        return {
            "has_cache": True,
            "message": "Cache file exists but unreadable"
        }
