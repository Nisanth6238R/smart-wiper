import tensorflow as tf   
from tensorflow import keras
import keras.models as models
import keras.layers as layers
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter

# function to preprocess the data
def preprocess_data(X, Y):
    # Convert class labels to categorical
    # Use LabelEncoder to encode string labels to integer
    # Then use to_categorical to convert integers to binary class matrix
    a=[]
    for i in range(len(Y)):
        if Y[i] == "Light Rain":
            a.append(0)
        elif Y[i] == "Moderate Rain":
            a.append(1)
        elif Y[i] == "High Rain":
            a.append(2)        
        elif Y[i] == "No Rain":
            a.append(3)
        
    le = LabelEncoder()
    Y = np.array(a)
    print(Y)
    Y = to_categorical(le.fit_transform(Y))
    print(Y)

    # Reshape X for CNN input
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    print("X Shape is: ", X.shape)
    print("y Shape is: ", Y.shape)
    return X, Y


# function to split the data into training and testing sets
def split_data(X, Y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=44)

    # Define the input shape for the CNN model
    up_width = 173
    up_height = 40 
    INPUTSHAPE = (up_height, up_width, 1)
    return X_train, X_test, y_train, y_test, INPUTSHAPE


# function to create the CNN model
def create_model(INPUTSHAPE):
    model = models.Sequential([
        layers.Conv2D(32 , (3,3),activation = 'relu',padding='valid', input_shape = INPUTSHAPE),  
        layers.AveragePooling2D(2, padding='same'),
        layers.Conv2D(128, (3,3), activation='relu',padding='valid'),
        layers.Dropout(0.3),
        layers.MaxPooling2D(2, padding='same'),
        layers.Conv2D(256, (3,3), activation='relu',padding='valid'),
        layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128 , activation = 'relu'),
        layers.Dense(4 , activation = 'softmax')
    ])
    # Compile the model with Adam optimizer and categorical_crossentropy loss function
    opt = keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = 'acc')
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    # Set batch size and early stopping callback
    batch_size = 20
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False)

    # Train the model with the training data and validation data, and use the early stopping callback
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40,
              callbacks=[callback], batch_size=batch_size)
              
    # Use the trained model to predict on the test data
    Y_pred = model.predict(X_test)
    # Get the predicted labels and true labels
    Y_pred1 = [np.argmax(i) for i in Y_pred]
    y_test1 = [np.argmax(i) for i in y_test]
    # Count the number of samples with each predicted label and true label
    Y_predcount = Counter(Y_pred1)
    y_testcount = Counter(y_test1)
    print(Y_predcount)
    print(y_testcount)
    
    # Save the trained model as a .h5 file and a .tflite file
    model.save("audioclassification_mfcc.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open('audioclassification_mfcc.tflite','wb').write(tflite_model)
    
    # Print the confusion matrix for the predicted and true labels
    print(confusion_matrix(y_test1 , Y_pred1))
   

# Load MFCC features from numpy files
X = np.load("mfcc_X.npy")
Y = np.load("mfcc_Y.npy")
print(len(Y))

# Preprocess the data
X, Y = preprocess_data(X, Y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test, INPUTSHAPE = split_data(X, Y)
# Define the input shape for the model
up_width = 173
up_height = 40
INPUTSHAPE = (up_height, up_width, 1)
# Create the deep learning model
model = create_model(INPUTSHAPE)
# Train the model
train_model(model, X_train, y_train, X_test, y_test)
