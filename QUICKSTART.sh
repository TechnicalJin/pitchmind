#!/bin/bash
# Quick Start — Run PitchMind with Match Cache
# =============================================

echo "🏏 PITCHMIND QUICK START"
echo "======================="
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python --version)"
echo ""

# Step 1: Initialize Cache
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Initialize Match Cache (First Time Only)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Run this command to load all IPL 2026 matches:"
echo ""
echo "  python initialize_match_cache.py"
echo ""
echo "This will:"
echo "  ✅ Fetch all matches from CricAPI (1 credit)"
echo "  ✅ Save to JSON cache"
echo "  ✅ Show summary of matches"
echo ""
read -p "Press Enter to continue or run above command first..."
echo ""

# Step 2: Start Flask App
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Start Flask IPL Schedule Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Run command:"
echo ""
echo "  cd espn_live_scraper"
echo "  python app.py"
echo ""
echo "Server will start at: http://127.0.0.1:5000"
echo ""
read -p "Run the command above in a new terminal window, then press Enter..."
echo ""

# Step 3: Open Browser
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Open Browser & Load Schedule"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Actions:"
echo "  1. Open browser → http://127.0.0.1:5000"
echo "  2. Click '📅 IPL SCHEDULE' tab"
echo "  3. Click 'LOAD SCHEDULE' button"
echo ""
echo "What happens:"
echo "  ✅ Shows all IPL 2026 matches from cache"
echo "  ✅ Shows '📦 Data loaded from cache'"
echo "  ✅ NO API CREDITS USED!"
echo ""

# Step 4: Main Dashboard
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Run Main Dashboard (Optional)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Run command in main pitchmind directory:"
echo ""
echo "  streamlit run 4_dashboard.py"
echo ""
echo "Features:"
echo "  🎯 IPL Match Prediction"
echo "  🏏 Live Match Tracking"
echo "  📊 Player Statistics"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete! You're ready to go 🚀"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
