import zipfile
import os

# Path to the zip file
zip_file_path = "/home/nishant/Smart wiper final/Raindrop_audio_classification/UrbanSound8K.zip"

# Directory to extract files
extract_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification"

# Create the extract directory if it doesn't exist
os.makedirs(extract_directory, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_directory)

print(f"Files extracted to '{extract_directory}'.")
