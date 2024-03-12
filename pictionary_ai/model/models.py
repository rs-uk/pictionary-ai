# from keras.models import Sequential
# from keras import Sequential
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
# from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.keras import callbacks
# from tensorflow.python.keras import utils
# from keras import Sequential
# from keras import Model, layers
# from keras import optimizers, regularizers
from tensorflow.keras import layers, Model
# from sklearn.preprocessing import TargetEncoder
from colorama import Fore, Style
import numpy as np
from typing import Tuple
from pictionary_ai.params import *


def initialize_model() -> Model:
    '''
    Initialize the Neural Network with random weights, using bidirectional LTSM
    masking layer.
    We use:
        - 2 Bidirectional LSTM layers
        - 3 dense layers
        - dropout layers between all dense layers
    '''
    model = Sequential()

    # Add Masking layer to handle variable-length sequences
    model.add(layers.Masking(mask_value=PADDING_VALUE, input_shape=(MAX_LENGTH, 3)))

    # Bidirectional LSTM layers with dropout option
    model.add(layers.Bidirectional(layers.LSTM(196, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)))

    # Dense layers with Dropout layers
    model.add(layers.Dense(128, activation='linear'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(64, activation='linear'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(32, activation='linear'))
    model.add(layers.Dropout(rate=0.2))

    # Categarization layer with the correct input size as the number of classes used in training
    model.add(layers.Dense(NUMBER_CLASSES, activation='softmax'))

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    '''
    Compile the Neural Network with loss categorical_crossentropy,
    optimiser Adam, and accuracy for the metric.
    Return the compiled model.
    '''
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("✅ Model compiled")

    return model


def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=256,
                patience=3,
                validation_data=None, # overrides validation_split
                validation_split=0.3,
                checkpoint_path:str = MODELS_PATH) -> Tuple[Model, dict]:
    '''
    Fit the model and return a tuple (fitted_model, history).
    We save checkpoints as well.
    '''
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = callbacks.EarlyStopping(monitor="val_accuracy",
                                 patience=patience,
                                 restore_best_weights=True,
                                 verbose=1
                                 )

    model_checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                          save_weights_only=True,
                                                          monitor='val_accuracy',
                                                          mode='max',
                                                          save_best_only=True
                                                          )

    history = model.fit(X,
                        y,
                        validation_data=validation_data,
                        validation_split=validation_split,
                        epochs=50,
                        batch_size=batch_size,
                        callbacks=[es, model_checkpoint_callback],
                        verbose=1
                        )

    print(f"✅ Model trained on {len(X)} drawings in {NUMBER_CLASSES} classes, with min val accuracy: {round(np.min(history.history['accuracy']), 2)}")

    return model, history


def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64
                   ) -> list:
    '''
    Evaluate trained model performance on the test subset, return metrics.
    '''

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(X=X,
                             y=y,
                             batch_size=batch_size,
                             verbose=0,
                             return_dict=True
                             )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)}")

    return metrics






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
    model.add(layers.Dense(NUMBER_CLASSES, activation='softmax'))
    # Replace 'NUMBER_CLASSES' with the actual number of classes in your problem

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
