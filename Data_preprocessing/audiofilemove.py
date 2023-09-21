import os
import shutil

# Source directory
source_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Raindrop_class/Moderate Rain/"

# Destination directory
destination_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Mixedaudio_raindrop_class/Moderate Rain/"

# List of audio file extensions
audio_extensions = (".wav", ".mp3", ".flac")  # Add more extensions as needed

# Walk through the source directory
for root, _, files in os.walk(source_directory):
    for filename in files:
        if filename.lower().endswith(audio_extensions):
            source_path = os.path.join(root, filename)
            destination_path = os.path.join(destination_directory, filename)
            
            # Move the audio file
            shutil.copyfile(source_path, destination_path)
            
            print(f"Moved '{filename}' to '{destination_directory}'.")
            
print("All audio files moved.")
