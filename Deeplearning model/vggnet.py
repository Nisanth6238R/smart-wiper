import tensorflow as tf   
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten

# function to preprocess the data
def preprocess_data(X, Y):
    # Convert class labels to categorical
    # Use LabelEncoder to encode string labels to integer
    # Then use to_categorical to convert integers to binary class matrix
    a = []
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
    Y = to_categorical(le.fit_transform(Y))

    # Reshape X for CNN input and repeat single channel 3 times
    X = np.repeat(X[:, :, :, np.newaxis], 3, axis=3)

    print("X Shape is: ", X.shape)
    print("y Shape is: ", Y.shape)
    return X, Y

# function to split the data into training and testing sets
def split_data(X, Y):
    X_model, X_validate, Y_model, Y_validate = train_test_split(X,Y,test_size=0.05,random_state = 40)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_model, Y_model, test_size=0.10, random_state=44)

    # Define the input shape for the CNN model
    up_width = 173
    up_height = 40 
    INPUTSHAPE = (up_height, up_width, 1)
    return X_train, X_test, y_train, y_test, X_validate, Y_validate, INPUTSHAPE


# function to create the VGGNet model
def create_model(INPUTSHAPE):
    # Load VGG16 without top layers and custom input shape
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(INPUTSHAPE[0], INPUTSHAPE[1], 3))

    # Freeze the layers of the base VGG16 model
    for layer in base_model.layers:
        layer.trainable = False

    # Add your own top layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)  # 4 classes

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test,X_validate, Y_validate):
    # Set batch size and early stopping callback
    batch_size = 20
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False)

    # Train the model with the training data and validation data, and use the early stopping callback
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=32,
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
    model.save("audioclassification_mfcc_vgg.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open('audioclassification_mfcc_vgg.tflite','wb').write(tflite_model)
    
    # Print the confusion matrix for the predicted and true labels
    print(confusion_matrix(y_test1 , Y_pred1))
    confusion =  confusion_matrix(y_test1, Y_pred1)
    # Generate a heatmap for the confusion matrix
    class_names = ["Light Rain", "Moderate Rain", "High Rain", "No Rain"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Heatmap of Test Sets')
    # Save the heatmap as an image
    plt.savefig('confusion_matrix_heatmap_test_mfcc.png')

    #validation results
    Y_preidcted_validation =model.predict(X_validate)
    # Get the predicted labels and true labels
    Y_preidcted_validation1 = [np.argmax(i) for i in Y_preidcted_validation]
    Y_validate1 = [np.argmax(i) for i in Y_validate]
    # Count the number of samples with each predicted label and true label
    Y_predvalidation_count = Counter(Y_preidcted_validation1)
    y_testvalidation_count = Counter(Y_validate1)
    print("Validation predicted count: ", Y_predvalidation_count)
    print("Validation test count: ", y_testvalidation_count)
    confusion1 =  confusion_matrix(Y_validate1, Y_preidcted_validation1)
    print(confusion1)
    # Generate a heatmap for the confusion matrix
    class_names1 = ["Light Rain", "Moderate Rain", "High Rain", "No Rain"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion1, annot=True, fmt='d', cmap='Blues', xticklabels=class_names1, yticklabels=class_names1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Heatmap of Validation Sets')
    plt.savefig('confusion_matrix_heatmap_validation_mfcc.png')

# Load MFCC features from numpy files
X = np.load("mfcc_X.npy")
Y = np.load("mfcc_Y.npy")
print(len(Y))

# Preprocess the data
X, Y = preprocess_data(X, Y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test, X_validate, Y_validate, INPUTSHAPE = split_data(X, Y)
# Define the input shape for the model
up_width = 173
up_height = 40
INPUTSHAPE = (up_height, up_width, 1)
# Create the deep learning model
model = create_model(INPUTSHAPE)
# Train the model
train_model(model, X_train, y_train, X_test, y_test, X_validate, Y_validate)

