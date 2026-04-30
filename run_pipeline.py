import os

scripts = [
     "download_cricsheet_ipl.py",   # 🔥 NEW STEP
    "0_json_to_csv.py",
    "1_data_cleaning.py",
    "2_feature_engineering.py",
    "5_live_data_fetch.py",
    "6_player_features.py",
    "7_phase_predictor.py",
    "build_master_players.py",
    "name_resolver.py",
    "pitchmind_player_features.py"
]

for script in scripts:
    print(f"Running {script}...")
    os.system(f"python {script}")

print("✅ Pipeline completed successfully!")