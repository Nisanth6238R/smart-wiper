import librosa
import soundfile as sf

# Input audio file
input_audio_path = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Audio/Rain/highwaydriving in the rain.wav"

# Output audio file (sliced portion)
output_audio_path = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Audio/Rain/sliced_audio.wav"

# Define the start and end times in seconds
start_time = 30 * 60  # 30 minutes in seconds
end_time = 120 * 60   # 120 minutes in seconds

# Load the audio using librosa
y, sr = librosa.load(input_audio_path, sr=None)

# Calculate frame indices for the desired time range
start_frame = int(start_time * sr)
end_frame = int(end_time * sr)

# Extract the desired portion of the audio
sliced_audio = y[start_frame:end_frame]

# Save the sliced audio to the output path
sf.write(output_audio_path, sliced_audio, sr)

print("Sliced audio saved as:", output_audio_path)
