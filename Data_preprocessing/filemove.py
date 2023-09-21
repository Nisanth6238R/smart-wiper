import shutil

# Source directory
source_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Slicedaudio/NoRain/"

# Destination directory
destination_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Raindrop_class/No Rain/"

# Move the folder
shutil.move(source_directory, destination_directory)

print(f"Folder moved from '{source_directory}' to '{destination_directory}'.")
