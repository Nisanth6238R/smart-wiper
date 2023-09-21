import os
import librosa
import soundfile as sf
import numpy as np
import shutil

# Input directory containing audio clips
input_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification/pt/"

# Output directory to save extended audio clips
output_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification/pt1/"

# Desired duration in seconds
desired_duration = 4.0

# Loop through audio files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        input_audio_path = os.path.join(input_directory, filename)
        output_audio_path = os.path.join(output_directory, filename)
        
        # Load the audio using librosa
        y, sr = librosa.load(input_audio_path, sr=None)
        
        # Calculate the current duration in seconds
        current_duration = len(y) / sr
        
        if current_duration < desired_duration:
            # Calculate the amount of padding needed
            padding_samples = int((desired_duration - current_duration) * sr)
            
            # Generate padding audio with zeros
            padding_audio = np.zeros(padding_samples)
            
            # Concatenate the padding audio to the original audio
            extended_audio = np.concatenate((y, padding_audio))
            
            # Save the extended audio to the output path
            sf.write(output_audio_path, extended_audio, sr)
            
            print(f"Extended and saved: {output_audio_path}")
        else:
            # If the audio is longer than or equal to desired duration, just copy it
            shutil.copy(input_audio_path, output_audio_path)
            
            print(f"Copied: {output_audio_path}")
