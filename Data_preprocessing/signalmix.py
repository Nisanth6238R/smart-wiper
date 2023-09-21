import os
import librosa
import soundfile as sf
import numpy as np
from librosa.util import pad_center

# Set the desired SNR levels in dB
snr_dbs = [10, 20]

# Set the number of files to create for each SNR level
num_files = 5000

# Directory paths for rain and no rain classes
rain_dirs = [
    "/home/nishant/Smart wiper final/Raindrop_audio_classification/Raindrop_class/High Rain/",
    "/home/nishant/Smart wiper final/Raindrop_audio_classification/Raindrop_class/Light Rain/",
    "/home/nishant/Smart wiper final/Raindrop_audio_classification/Raindrop_class/Moderate Rain/"
]
no_rain_dir = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Raindrop_class/No Rain/"

# Set the output directory for the mixed audio files
out_dir = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Mixedaudio_raindrop_class/"

# Loop over the SNR levels
for snr_db in snr_dbs:
    print('Mixing audios at SNR =', snr_db)
    
    # Calculate the power ratio for the desired SNR level
    power_ratio = 10**(-snr_db/10)
    
    # Loop over the rain directories
    for rain_dir in rain_dirs:
        rain_class = os.path.basename(os.path.normpath(rain_dir))
        class_out_dir = os.path.join(out_dir, rain_class)
        os.makedirs(class_out_dir, exist_ok=True)
        
        # Loop over the number of files to create for this SNR level
        for i in range(num_files):
            print(f'\tMixing {rain_class} - SNR = {snr_db} dB - File {i+1}/{num_files}')
            
            # Load a random rain audio file
            rain_files = os.listdir(rain_dir)
            rain_file = np.random.choice(rain_files)
            rain_audio, sr = librosa.load(os.path.join(rain_dir, rain_file), sr=None)
            
            # Load a random "No Rain" audio file
            no_rain_files = os.listdir(no_rain_dir)
            no_rain_file = np.random.choice(no_rain_files)
            no_rain_audio, sr = librosa.load(os.path.join(no_rain_dir, no_rain_file), sr=None)
            
            # Pad the shorter audio signal with zeros
            if len(rain_audio) > len(no_rain_audio):
                no_rain_audio = pad_center(data=no_rain_audio, size=len(rain_audio))
            else:
                rain_audio = pad_center(data=rain_audio, size=len(no_rain_audio))
            
            # Calculate the powers of the audio signals
            rain_power = np.mean(np.abs(rain_audio)**2)
            no_rain_power = np.mean(np.abs(no_rain_audio)**2)
            
            # Calculate the scaling factor for the "No Rain" audio signal
            scale_factor = np.sqrt(rain_power*power_ratio/no_rain_power)
            
            # Scale the "No Rain" audio signal
            no_rain_audio_scaled = no_rain_audio * scale_factor
            
            # Mix the two audio signals
            mixed_audio = rain_audio + no_rain_audio_scaled
            
            # Save the mixed audio file
            mixed_filename = f'{rain_class}_SNR{snr_db}_{i+1}.wav'
            mixed_filepath = os.path.join(class_out_dir, mixed_filename)
            sf.write(mixed_filepath, mixed_audio, sr)
