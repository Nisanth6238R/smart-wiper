import os

# Directory containing files to be deleted
directory_path = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Raindrop_class/No Rain/"

# Get a list of all files in the directory
file_list = os.listdir(directory_path)

# Loop through the files and delete them
for filename in file_list:
    file_path = os.path.join(directory_path, filename)
    try:
        os.remove(file_path)
        print(f"Deleted: {filename}")
    except Exception as e:
        print(f"Error deleting {filename}: {e}")
        
print("All files deleted.")
