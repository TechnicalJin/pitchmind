# 🏏 PitchMind Cache System — Workflow Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   PITCHMIND DASHBOARD                       │
│                  (streamlit run 4_dashboard.py)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FLASK SCHEDULE SERVER                           │
│         (cd espn_live_scraper && python app.py)             │
│  http://127.0.0.1:5000                                      │
└────────────────┬────────────────────────┬────────────────────┘
                 │                        │
        ┌────────▼──────────┐    ┌────────▼──────────┐
        │  IPL SCHEDULE TAB │    │   LIVE MATCHES   │
        │   (Cached Data)   │    │   (Always Fresh) │
        └────────┬──────────┘    └────────┬──────────┘
                 │                        │
                 ▼                        ▼
     ┌────────────────────┐    ┌──────────────────┐
     │  cache_manager.py  │    │  scraper.py      │
     │  - Load from cache │    │  - Fetch live    │
     │  - Save to cache   │    │  - Process data  │
     └────────┬───────────┘    └──────────┬───────┘
              │                           │
              ▼                           ▼
    ┌─────────────────────┐      ┌──────────────────────┐
    │ JSON Cache Files    │      │ CricAPI Endpoint     │
    │  ipl_2026_matches   │      │ (100 req/day limit)  │
    │     .json           │      └──────────────────────┘
    │  cache_metadata     │
    │     .json           │
    └─────────────────────┘
```

## Workflow: First Time User

```
Step 1: INITIALIZE CACHE
┌──────────────────────────────────────────┐
│ python initialize_match_cache.py          │
└──────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │  Check: Has cache?   │
    │  (First time = NO)   │
    └──────────────────────┘
           │ NO
           ▼
    ┌──────────────────────────────────────┐
    │ Fetch from CricAPI                    │
    │ /v1/series_info (1 credit used)     │
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │ Save to JSON cache                    │
    │ espn_live_scraper/cache/...json     │
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │ Display summary:                      │
    │ ✅ 74 matches cached                  │
    │ ✅ All match IDs available            │
    │ ✅ Ready to use!                      │
    └──────────────────────────────────────┘


Step 2: START SERVER
┌──────────────────────────────────────────┐
│ cd espn_live_scraper && python app.py    │
└──────────────────────────────────────────┘
           │
           ▼
    Server ready at http://127.0.0.1:5000


Step 3: LOAD SCHEDULE
┌──────────────────────────────────────────┐
│ Open http://127.0.0.1:5000               │
│ Click "IPL SCHEDULE" tab                  │
│ Click "LOAD SCHEDULE" button              │
└──────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │  Check: Has cache?   │
    │  (Now = YES!)        │
    └──────────────────────┘
           │ YES
           ▼
    ┌──────────────────────────────────────┐
    │ Load from JSON cache                  │
    │ (0 API credits! <0.1 sec!)           │
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │ Display all 74 matches instantly      │
    │ Show: "📦 Data loaded from cache"     │
    │ No API used!                          │
    └──────────────────────────────────────┘
```

## Workflow: Subsequent Times

```
DAY 1 (After initialization):
┌─ LOAD SCHEDULE ────────────┐
│ Cache: YES (just created)   │
│ Load from cache             │
│ API Credits Used: 0         │
└─────────────────────────────┘

DAY 2:
┌─ LOAD SCHEDULE ────────────┐
│ Cache: YES (still valid)    │
│ Load from cache             │
│ API Credits Used: 0         │
└─────────────────────────────┘

DAY 3:
┌─ LOAD SCHEDULE ────────────┐     ┌─ REFRESH ──────────────┐
│ Cache: YES                  │ OR  │ Force API fetch (new)   │
│ Load from cache             │     │ Update cache            │
│ API Credits Used: 0         │     │ API Credits Used: 1     │
└─────────────────────────────┘     └─────────────────────────┘

REPEAT FOR ENTIRE MONTH:
Total API Credits Used: 1 (not 100!)
Saved: 99 credits! 🎉
```

## Feature: Export Match IDs

```
Browser UI:
┌──────────────────────────────────────────┐
│  LOAD SCHEDULE    REFRESH    EXPORT IDs  │
└───────────┬────────────────────────┬─────┘
            │                        │
            ▼                        ▼
    Load from cache            Download JSON file
    or API                      with all 74 IDs
                                    │
                                    ▼
                            ┌──────────────────┐
                            │ ipl_2026_***.js  │
                            │                  │
                            │ {                │
                            │  "id1": { ... }, │
                            │  "id2": { ... }, │
                            │  ...             │
                            │ }                │
                            └──────────────────┘
                                    │
                                    ▼
                        Use in 5_live_data_fetch.py:
                        MATCH_CONFIG = {
                            "cricapi_match_id": "id1"
                        }
```

## Data Flow: Manual Fetch vs Cache

### Manual Fetch (Before Cache System)
```
Browser
   │
   ▼
Flask App
   │
   ▼
CricAPI (1 credit)
   │
   ▼
Display in Browser
[Cache? NONE - wasted!]
```

### With Cache System
```
First Load:
Browser
   │
   ├─→ Check Cache? (NO)
   │
   ▼
Flask App
   │
   ▼
CricAPI (1 credit)
   │
   ├─→ Save to JSON
   │
   ▼
Display in Browser

Subsequent Loads:
Browser
   │
   ├─→ Check Cache? (YES)
   │
   ├─→ Load from JSON (0 credits)
   │
   ▼
Display in Browser (instant!)
```

## API Credit Impact

### Without Cache System
```
100 API Credits/Day

Load Schedule 10 times = 10 credits
Refresh Schedule 5 times = 5 credits
Check Live Matches 30 times = 0 credits (live has no cache)
Other operations = 50 credits
Total: 65 credits used (35 wasted!)
```

### With Cache System
```
100 API Credits/Day

Load Schedule 1 time (cached) = 1 credit
Load Schedule 9 times (from cache) = 0 credits
Refresh Schedule 5 times = 5 credits
Check Live Matches 30 times = 0 credits
Other operations = 50 credits
Total: 56 credits used (saved 9!)
```

## Cache File Structure

```
espn_live_scraper/
├── cache/
│   ├── ipl_2026_matches.json
│   │   {
│   │     "fetched_at": "2026-03-27T10:30:00",
│   │     "total_matches": 74,
│   │     "matches": [
│   │       {
│   │         "id": "55fe0f15...",
│   │         "name": "MI vs CSK",
│   │         "dateTimeGMT": "2026-03-28T19:30:00Z",
│   │         "venue": "Wankhede Stadium",
│   │         "teams": ["Mumbai Indians", "Chennai Super Kings"],
│   │         ...
│   │       }
│   │     ]
│   │   }
│   │
│   └── cache_metadata.json
│       {
│         "last_updated": "2026-03-27T10:30:00",
│         "cached_matches": 74,
│         "source": "CricAPI"
│       }
│
└── ...
```

## Performance Comparison

```
Operation                  Without Cache    With Cache
──────────────────────────────────────────────────────
Load Schedule (1st time)   2-3 seconds      2-3 seconds (API)
Load Schedule (2nd+ time)  2-3 seconds      <0.1 seconds ⚡
Export IDs (1st time)      Not possible     1 click
API Credits Used           100% wasted      1% efficient
Scalability                Limited          Unlimited loads
```

---

**Summary:** After first load, you get unlimited schedule access with 0 API credits! 🚀
