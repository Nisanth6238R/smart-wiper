import pandas as pd
import librosa
import cv2
import time
import numpy as np
from librosa import util

def process_audio_files(audio_dataset_path):
    start_time = time.time()  # get start time
    df = pd.read_csv(audio_dataset_path)
    X = []
    Y = []
    up_width = 173
    up_height = 40
    error_files = []  # to store the names of error files

    # loop over each row in the DataFrame
    for data in df.iterrows():
        #try:
            # load the audio file with librosa
            raw, sr = librosa.load(data[1][1], res_type='kaiser_fast')
            #print(data[1])
            #print(data[1][1])
            # normalize the raw audio
            raw = util.normalize(raw)
            # extract the MFCC features
            #X_mfcc = librosa.feature.mfcc(y=raw, sr=sr, n_mfcc=40)  

             # extract the Mel Spectrogram features
            #mel_spectrogram = librosa.feature.melspectrogram(y=raw, sr=sr, n_mels=40)
            #log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            #mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            # resize the Mel Spectrogram features using OpenCV
            up_points = (up_width, up_height)
            #resized_log_mel_spectrogram = cv2.resize(log_mel_spectrogram, up_points, interpolation=cv2.INTER_LINEAR)
            #resized_mel_spectrogram = cv2.resize(mel_spectrogram, up_points, interpolation=cv2.INTER_LINEAR)
            X_chroma_stft1 = librosa.feature.chroma_stft(y=raw, sr=sr)
            X_chroma_stft = cv2.resize(X_chroma_stft1, up_points, interpolation= cv2.INTER_LINEAR)
            # append the Mel Spectrogram features and label to X and Y respectively
            X.append(X_chroma_stft)
            Y.append(data[1][3])
            #print(X_mfcc.shape)   
            # resize the MFCC features using OpenCV       
            #up_points = (up_width, up_height)
            #print(up_points)
            #X_mfcc = cv2.resize(X_mfcc, up_points, interpolation= cv2.INTER_LINEAR)
            #print(X_mfcc.shape)
            

            # append the MFCC features and label to X and Y respectively
            #X.append(X_mfcc)
            #Y.append(data[1][3])
            #print("Hey")
        #except:
            # if an error occurs while processing the audio file, append the name of the file to error_files
            #error_files.append(data[1][1])
            #print(data[1][1])

    # convert X and Y to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    # save X and Y as numpy binary files
    np.save("mfcc_X.npy", X)
    np.save("mfcc_Y.npy", Y)
    # save X and Y as numpy binary files
    #np.save("mfcc_X.npy", X)
    #np.save("mfcc_Y.npy", Y)
    end_time = time.time()  # get end time
    print(f"Runtime: {end_time - start_time} seconds")

# Set the path to the CSV file containing information about the audio files
audio_dataset_path = "/home/nishant/Smart wiper final/Data_preprocessing/Shuffled_Mixed_rain_classes.csv"
# Call the process_audio_files function and pass the audio_dataset_path as argument
process_audio_files(audio_dataset_path)

