import os
import csv

# Directory containing subdirectories (classes)
base_directory = "/home/nishant/Smart wiper final/Raindrop_audio_classification/Mixedaudio_raindrop_class/"

# Output CSV file path
csv_file_path = "Mixed_rain_classes.csv"

# Create a CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    fieldnames = ["s.no", "file location", "file name", "class"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Loop through class directories
    for class_name in os.listdir(base_directory):
        class_directory = os.path.join(base_directory, class_name)
        
        if os.path.isdir(class_directory):
            # Loop through the files in the class directory
            for s_no, filename in enumerate(os.listdir(class_directory), start=1):
                file_location = os.path.join(class_directory, filename)
                
                # Write a row to the CSV file
                csv_writer.writerow({
                    "s.no": s_no,
                    "file location": file_location,
                    "file name": filename,
                    "class": class_name
                })

print(f"CSV file '{csv_file_path}' created.")
