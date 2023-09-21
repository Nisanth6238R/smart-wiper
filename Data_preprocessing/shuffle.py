import pandas as pd

# Specify the CSV file paths
input_csv_file_path = "/home/nishant/Smart wiper final/Data_preprocessing/Mixed_rain_classes.csv"
output_csv_file_path = "/home/nishant/Smart wiper final/Data_preprocessing/Shuffled_Mixed_rain_classes.csv"

# Read the input CSV file into a DataFrame
data = pd.read_csv(input_csv_file_path)

# Shuffle the rows randomly
shuffled_data = data.sample(frac=1).reset_index(drop=True)

# Write the shuffled data to the output CSV file
shuffled_data.to_csv(output_csv_file_path, index=False)

print("Rows shuffled and saved to", output_csv_file_path)
