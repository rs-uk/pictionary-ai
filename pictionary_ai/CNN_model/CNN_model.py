'''
Full CNN model, including:
* Downloading the numpy files for the 50 class subset from QuickDraw buckets
* Initialising and training a CNN model

'''

####################################
### Setup
####################################

# Import required packages for dataset preparation
from google.cloud import storage
import json
# import ndjson
import random
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# Import relevant packages for building and training CNN model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple
import time

# Define required parameters and variables
dict_50_class_subset = {'aircraft carrier': 0, 'arm': 1, 'asparagus': 2, 'backpack': 3, 'banana': 4, 'basketball': 5,
                        'bottlecap': 6, 'bread': 7, 'broom': 8, 'bulldozer': 9, 'butterfly': 10, 'camel': 11, 'canoe': 12,
                        'chair': 13, 'compass': 14, 'cookie': 15, 'drums': 16, 'eyeglasses': 17, 'face': 18, 'fan': 19,
                        'fence': 20, 'fish': 21, 'flying saucer': 22, 'grapes': 23, 'hand': 24, 'hat': 25, 'horse': 26,
                        'light bulb': 27, 'lighthouse': 28, 'line': 29, 'marker': 30, 'mountain': 31, 'mouse': 32,
                        'parachute': 33, 'passport': 34, 'pliers': 35, 'potato': 36, 'sea turtle': 37, 'snowflake': 38,
                        'spider': 39, 'square': 40, 'steak': 41, 'swing set': 42, 'sword': 43, 'telephone': 44,
                        'television': 45, 'tooth': 46, 'traffic light': 47, 'trumpet': 48, 'violin': 49}

file_list = [name+'.npy' for name in list(dict_50_class_subset.keys())]

bucket_name = 'quickdraw_dataset'
blob_prefix = 'full/numpy_bitmap/'
# local_folder_path = '/home/jupyter/data/numpy_bitmap/'
local_folder_path = '/home/raj/code/rs-uk/pictionary-ai/raw_data/numpy_bitmap/'



####################################
### Download our 50 class subset
####################################

# Download a blob from a bucket and store it in memory
def download_numpy_bitmap_to_file(bucket_name, source_blob_name, destination_path) -> None:
    '''
    Downloads a (ndjson) blob from the bucket and return json file as dict
    '''
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(source_blob_name)

    # Download the file to local storage
    blob.download_to_filename(destination_path)

    return None

def download_50_classes(
    file_list: list,
    bucket_name: str,
    blob_prefix: str,
    local_folder_path: str
    ) -> None :
    '''
    This function:
    * Downloads numpy bitmap files for the 50 class subset from Google Bucket to local folder
    '''

    # Create a list of the numpy bitmap files related to the 50 classes
    print('Downloading 50 class subset numpy bitmap files from Google Bucket...')

    for file in file_list :
        source_blob_name = blob_prefix+file
        print(source_blob_name)
        destination_file_path = local_folder_path+file
        print(destination_file_path)
        download_numpy_bitmap_to_file(bucket_name, source_blob_name, destination_file_path)
        print(f'Downloaded {file}')

    print('✅ File downloads complete')
    return None


def create_bmp_50class_10pc(
    file_list: list,
    local_folder_path: str
    ) -> list :
    '''
    This function:
    * Iterates through the downloaded files and loads the nparray
    * Take a 10% sample from each file and concatenates the classes together into one list
    * Creates bmp_50class_10pc - a list of dictionaries
    '''

    print('Creating 10pc sample of 50 classes ...')

    step_size = 10 # i.e. 10% sample
    bmp_50class_10pc = []

    # Loop through each numpy file
    for i in range(len(file_list)):
        file = file_list[i]
        file_path = local_folder_path+file

        # Load .npy file
        array = np.load(file_path)
        print(f'Loaded {file} with {len(array)} rows')

        # Create a list of dictionaries - stepping through sampled array (i.e step size of 10 = 10% sample)
        dict_list = [{'class':file.replace('.npy',''),'bmp':bmp} for bmp in array[::step_size]]
        print(f'Adding {len(dict_list)} rows')

        bmp_50class_10pc = bmp_50class_10pc + dict_list

    print('✅ File 10pc sample extract complete')

    return bmp_50class_10pc


####################################
## Prepare combined dataset for model
####################################

def OHE_class_name(class_name: str, mapping_dict: dict) -> list:
    '''
    Manually One Hot Encode the y-values using mapping dictionary
    '''
    nb_classes = len(mapping_dict)
    OHE_output = [0] * nb_classes
    OHE_output[mapping_dict[class_name]] = 1
    # Need to convert the np.ndarray into a list so it can be parsed into JSON
    return OHE_output

def tranform_shuffle_split_data(bmp_50class_10pc: list) -> list :
    '''
    * Normalised X - i.e. convert values from [0,255] to [0,1]
    * OHC Y manually using mapping dictionary
    * Shuffle list
    * Split data into train and test
    * Returns X_train, X_test, y_train and y_test
    '''

    print('Starting to transform, shuffle and split the dataset...')

    for i in range(len(bmp_50class_10pc)) :
        drawing = bmp_50class_10pc[i]

        # Normalise X value - i.e. divide by 255
        drawing['bmp'] = drawing['bmp']/255
        # Reshape X in 28x28 bmp shape
        drawing['bmp'] = drawing['bmp'].reshape(28,28)
        # OHE class
        drawing['OHE_class'] = OHE_class_name(drawing['class'], dict_50_class_subset)

    print('✅ Datset transformations complete')

    # Shuffle list
    random.shuffle(bmp_50class_10pc)
    print('✅ Dataset shuffle complete')


    X = np.array([drawing['bmp'] for drawing in bmp_50class_10pc])
    y = np.array([drawing['OHE_class'] for drawing in bmp_50class_10pc])

    # Split X_train & y_train in train and val sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    print('✅ Dataset split into train and test complete')

    return X_train, X_test, y_train, y_test


####################################
### Design our CNN model
####################################

# Initialise model
def initialize_model() -> models.Model:
    '''
    Initialise model with the same CNN structure we used for number recognition - accepting bmp files of dimension 28x28 bits
    * `Conv2D` layer with 8 filters, each of size (4, 4), an input shape of (28x28), the `relu` activation function, and `padding='same'
    * `MaxPool2D` layer with a `pool_size` equal to (2, 2)
    * second `Conv2D` layer with 16 filters, each of size (3, 3), and the `relu` activation function
    * second `MaxPool2D` layer with a `pool_size` equal to (2, 2)

    * `Flatten` layer
    * first `Dense` layer with 10 neurons and the `relu` activation function
    * last softmax predictive layer of 50 classes - i.e. number of classes in data subset
    '''

    model = models.Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, (4,4), input_shape=(28, 28, 1), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer - Classification Layer with 10 outputs corresponding to 10 digits
    model.add(layers.Dense(50, activation='softmax'))

    print("✅ Model initialized")

    return model

def compile_model(model: models.Model, learning_rate=0.0005) -> models.Model:
    '''
    Compile the model, which:
    * optimizes the `categorical_crossentropy` loss function,
    * with the `adam` optimizer with learning_rate=0.0005,
    * and the `accuracy` as the metrics
    '''

    # Create optimizer with custom learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    print("✅ Model compiled")

    return model

# Save json to local folder - to store training history
def save_json_to_local (data: list, folder_path: str, file_name: str) -> None:
    file_path = folder_path+file_name
    with open(file_path, 'w') as file :
        json.dump(data, file)
    print(f'Saved data to {file_path}')
    return None

# Train the model
def train_model(
            model: models.Model,
            X: np.ndarray,
            y: np.ndarray,
            batch_size=256,
            patience=5,
            validation_data=None, # overrides validation_split, if available
            validation_split=0.2
            ) -> Tuple[models.Model, dict]:
    '''
    Train on the model and return a tuple (fitted_model, history)
    Checkpoints have also been included to store model weights after each epoch
    '''

    es = callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1
            )

    # Create checkpoints - add timestamp to prevent writing over old training runs
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #checkpoint_filepath = '/home/jupyter/data/numpy_bitmap/CNN_model'+timestr
    checkpoint_filepath = '/home/raj/code/rs-uk/pictionary-ai/raw_data/numpy_bitmap/CNN_model'+timestr


    #Save the checkpoints in the checkpoint_filepath
    model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
            )

    history = model.fit(
            X,
            y,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=1000,
            batch_size=batch_size,
            callbacks=[es, model_checkpoint_callback],
            verbose=1
            )

    print(f"✅ Model trained on {len(X)} rows with maximum val accuracy: {round(np.min(history.history['accuracy']), 2)}")

    save_json_to_local (history, checkpoint_filepath, 'model_history_'+timestr)

    return model, history

def evaluate_model(
            model: models.Model,
            X: np.ndarray,
            y: np.ndarray,
            batch_size=64
            ) -> Tuple[models.Model, dict]:
    '''
    Evaluate performance of the trained model on the test dataset
    Returns evaluation metrics
    '''

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
            x=X,
            y=y,
            batch_size=batch_size,
            verbose=1,
            # callbacks=None,
            return_dict=True
            )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)}")

    return metrics


###### Function calls to prepare dataset ######
# download_50_classes(file_list, bucket_name, blob_prefix, local_folder_path)
# bmp_50class_10pc = create_bmp_50class_10pc(file_list, local_folder_path)
# X_train, X_test, y_train, y_test = tranform_shuffle_split_data(bmp_50class_10pc)

# ###### Run the CNN model ######
# # Model summary
# model = initialize_model()
# model = compile_model(model)
# model.summary()
# model, history = train_model(model, X_train, y_train, validation_split=0.2)
# evaluate_model(model, X_test, y_test)



####################################
### Convert API JSON into format ready for prediction
####################################

def normalize_strokes(strokes, epsilon=1.0, resample_spacing=1.0):
    '''
    Function centres and resize the strokes to a format similar to that on which the model was trained
    '''
    if len(strokes) == 0:
        raise ValueError('empty image')

    # find min and max
    amin = None
    amax = None
    for x, y in strokes:
        cur_min = [np.min(x), np.min(y)]
        cur_max = [np.max(x), np.max(y)]
        amin = cur_min if amin is None else np.min([amin, cur_min], axis=0)
        amax = cur_max if amax is None else np.max([amax, cur_max], axis=0)

    # drop any drawings that are linear along one axis
    arange = np.array(amax) - np.array(amin)
    if np.min(arange) == 0:
        raise ValueError('bad range of values')

    arange = np.max(arange)
    output = []
    for x, y in strokes:
        xy = np.array([x, y], dtype=float).T
        xy -= amin
        xy *= 255.
        xy /= arange
        #resampled = resample(xy[:, 0], xy[:, 1], resample_spacing)
        #simplified = simplify_coords(xy, epsilon)
        xy = np.around(xy).astype(np.uint8)
        output.append(xy.T.tolist())

    return output

# Create a bmp np.array fromt the strokes in a drawing file
def draw_image_from_strokes(raw_strokes, size=256, lw=6, augmentation = False):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
    if augmentation:
        if random.random() > 0.5:
            img = np.fliplr(img)
    return img
