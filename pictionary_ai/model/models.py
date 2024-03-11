from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import utils
from keras import Model, Sequential, layers, regularizers, optimizers
from sklearn.preprocessing import TargetEncoder
from colorama import Fore, Style
import numpy as np
import pandas as pd
from typing import Tuple
from pictionary_ai.params import *


#no of classes we are using
num_classes = 10
def model_bidirectional() -> Model:
    """
    Initialize the Neural Network with random weights, using bidirectional LTSM
    masking layer
    it has 2 Bidirectional LSTM layers
    3 dense layers
    and dropout layers
    """

    model = Sequential()

    # Add Masking layer to handle variable-length sequences
    #put in 99 as 0 may effect the data
    model.add(layers.Masking(mask_value=99, input_shape=( MAX_LENGTH, 3)))

    #do we want to customize backwards layer?

    model.add(layers.Bidirectional(layers.LSTM(196, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)))

    # Add Dense layers
    model.add(layers.Dense(128, activation='linear'))
    #dropoutlayer
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(64, activation='linear'))
    #dropoutlayer
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(32, activation='linear'))
    #dropoutlayer
    model.add(layers.Dropout(rate=0.2))

    # Add final Softmax layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    # Replace 'num_classes' with the actual number of classes in your problem

    print("✅ Model initialized")

    return model


def model_LTSM() -> Model:
    """
    Initialize the Neural Network with random weights
    model that just has LSTM same structure otherwise
    """

    model = Sequential()

    # Add Masking layer to handle variable-length sequences
    #put in 99 as 0 may effect the data
    model.add(layers.Masking(mask_value=99, input_shape=(MAX_LENGTH, 3)))

    # Add LSTM layers
    model.add(layers.LSTM(64, activation='tanh', return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

    model.add(layers.LSTM(32, activation='tanh', dropout=0.2, recurrent_dropout=0.2))


    # Add Dense layers
    model.add(layers.Dense(128, activation='linear'))
    #dropoutlayer
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(64, activation='linear'))
    #dropoutlayer
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(32, activation='linear'))
    #dropoutlayer
    model.add(layers.Dropout(rate=0.2))

    # Add final Softmax layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    # Replace 'num_classes' with the actual number of classes in your problem

    print("✅ Model initialized")

    return model

def model_LTSM_conv() -> Model:
    '''model has conv1d layer and max pooling adn than LTSM, got this model from
    https://medium.com/@www.seymour/training-a-recurrent-neural-network-to-recognise-sketches-in-a
    realtime-game-of-pictionary-16c91e185ce6'''

    model = Sequential()

      # Input layer
    model.add(layers.Masking(mask_value=99, input_shape=(MAX_LENGTH, 3)))

      # Masking layer
    model.add(layers.Masking(mask_value=99))

      # 1D Convolutional Layers- should i have more or less dropout?
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(rate=0.2))

    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(rate=0.2))

      # Recurrent layers (e.g., LSTM)
    model.add(layers.LSTM(128, return_sequences=True,dropout=0.2, recurrent_dropout=0.2))

    model.add(layers.LSTM(128,dropout=0.2, recurrent_dropout=0.2))


      # Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.2))

      # Output layer
    model.add(layers.Dense(self.num_categories, activation='softmax'))


    return model

def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    with loss categorical_crossentropy, optimiser adam, metrics, accuracy
    returns model
    """
    #what loss do we want?
    #i think should be using categorical
    #which metrics?
    #do i want to create my own and what are the advantages of this

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # look at custum loss function
    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=3,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    added in checkpoint as well
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_filepath = '/home/jupyter/lewagon_projects/pictionary-ai/raw_data/models'

    #this will save the checkpoints in the checkpoint_filepath
    model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    #in fit is where we put in the padding, cant remember how
    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=50,
        batch_size=batch_size,
        callbacks=[es, model_checkpoint_callback],
        verbose=1
    )

    print(f"✅ Model trained on {len(X)} rows with min val accuracy: {round(np.min(history.history['accuracy']), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset, returns metrics
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)}")

    return metrics
