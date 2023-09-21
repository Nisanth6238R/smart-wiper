import os
import librosa
import numpy as np
import shutil

# Directory containing audio files
input_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Slicedaudio/Rain/"

# Output directory for each rain class
output_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Raindrop_class/"

# Define the thresholds for each rain level
light_rain_threshold = 0.01
medium_rain_threshold = 0.20
high_rain_threshold = 0.35

# Define the minimum number of samples required for each rain level
light_rain_min_samples = 4000
medium_rain_min_samples = 4000
high_rain_min_samples = 1000

# Loop through audio files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        audio_path = os.path.join(input_directory, filename)
        
        # Load the audio using librosa
        y, sr = librosa.load(audio_path, sr=None)
        
        num_samples_above_0_01 = np.sum(y > light_rain_threshold)
        num_samples_above_0_1 = np.sum(y > medium_rain_threshold)
        num_samples_above_0_2 = np.sum(y > high_rain_threshold)
        
        
        #num_samples = len(y)
        
        # Classify the audio based on the rain levels
        rain_class = "No Rain"
        if num_samples_above_0_2 >= high_rain_min_samples:
            rain_class = "High Rain"
        elif num_samples_above_0_1 >= medium_rain_min_samples:
            rain_class = "Moderate Rain"
        elif num_samples_above_0_01 >= light_rain_min_samples:
            rain_class = "Light Rain"
        
        # Create the output directory if it doesn't exist
        output_class_directory = os.path.join(output_directory, rain_class)
        os.makedirs(output_class_directory, exist_ok=True)
        
        # Rename and move the file to the appropriate folder
        new_filename = f"{os.path.splitext(filename)[0]}_{rain_class}.wav"
        new_audio_path = os.path.join(output_class_directory, new_filename)
        shutil.copyfile(audio_path, new_audio_path)
        
       # print(f"File: {filename} | Rain Class: {rain_class} | Moved to: {new_audio_path}")
