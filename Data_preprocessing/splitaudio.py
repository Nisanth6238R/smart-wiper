import os
import librosa
import soundfile as sf

"""
os: Imports the os module to interact with the operating system
librosa: Imports the librosa library, which provides tools for audio analysis and processing
soundfile: Imports the soundfile library, which provides an easy way to read and write audio files.
"""

# Set the duration of each clip in seconds
clip_duration = 4

# Set the input and output folder paths
#input_folder_path = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Audio/No Rain/"
input_folder_path = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Slicedaudio/No Rain/"
output_folder_path = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Raindrop_class/No Rain/"

# Loop through each file in the input folder
for file_name in os.listdir(input_folder_path):
    if file_name.endswith(".wav"):
        # Load the audio file using librosa
        input_file_path = os.path.join(input_folder_path, file_name)
        audio, sr = librosa.load(input_file_path, sr=None)

        # Calculate the number of clips
        clip_length = sr * clip_duration
        num_clips = len(audio) // clip_length

        # Loop through each clip
        for i in range(num_clips):
            clip_start = i * clip_length
            clip_end = (i + 1) * clip_length

            # Extract the clip
            clip = audio[clip_start:clip_end]

            # Set the output file path
            output_file_name = f"{os.path.splitext(file_name)[0]}_{i}.wav"
            output_file_path = os.path.join(output_folder_path, output_file_name)

            # Save the clip as a .wav file using soundfile
            sf.write(output_file_path, clip, sr, subtype="PCM_16")
