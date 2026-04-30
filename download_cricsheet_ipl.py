import requests
import zipfile
import os
from io import BytesIO

URL = "https://cricsheet.org/downloads/ipl_json.zip"
EXTRACT_PATH = "data/Data_Cricsheet"

os.makedirs(EXTRACT_PATH, exist_ok=True)

print("⬇️ Downloading IPL dataset...")

response = requests.get(URL)
zip_file = zipfile.ZipFile(BytesIO(response.content))

new_files = 0

for file in zip_file.namelist():
    file_path = os.path.join(EXTRACT_PATH, file)

    # If file does NOT exist → extract
    if not os.path.exists(file_path):
        zip_file.extract(file, EXTRACT_PATH)
        new_files += 1

print(f"✅ Added {new_files} new match files")