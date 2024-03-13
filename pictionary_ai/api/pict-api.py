from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pictionary_ai.main import preprocessor
from pictionary_ai.model import models
from pictionary_ai.CNN_model import CNN_model
import ujson
import requests
import numpy as np

from pictionary_ai.model import models_rs

# initiate model
model = models_rs.model_bidirectional()
model = models_rs.compile_model(model)
#load wieghts
model.load_weights('../shared_data/models_1003_50classes')

# initiate CNN model
model_CNN = CNN_model.initialize_model()
model_CNN = CNN_model.compile_model(model_CNN)
#load wieghts
model_CNN.load_weights('../shared_data/CNN_checkpoints/CNN_model20240313-022137')



app = FastAPI()
# app.state.model = main.load_model()  # make sure that the main contains the load_model() function

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],  # Allows all origins
                   allow_credentials=True,
                   allow_methods=["*"],  # Allows all methods
                   allow_headers=["*"],  # Allows all headers
                   )

@app.get("/home")
def homepage():

    return "Welcome to the pictionary ai api"

# get canvas capture and send back to streamlit the JSON
@app.post("/predict")
async def get_prediction(request: Request) -> dict:
    # Get the drawing at the end of each new stroke
    json_drawing = await request.json()
    # Process the drawing to the expect list format
    list_processed_drawing = (preprocessor.process_drawing_data(json_drawing)).tolist()
    # Should we pad the drawing??
    list_padded_drawing = preprocessor.add_padding(list_processed_drawing)

    X_processed = np.expand_dims(list_padded_drawing,0)

    #predict - this is gonna give me an array with the percentage that is in each class
    res = model.predict(X_processed)[0]
    prediction = np.argmax(res)
    # this will rturn the index of the highest percentage prediction, we need to map this to its key

    prediction = np.argmax(res)
    return_dict = {'result': str(res), 'prediction': str(prediction)}

    return return_dict

##### RS ADDITION #####
# CNN Version : get canvas capture and send back to streamlit the JSON
@app.post("/predict_CNN")
async def get_prediction(request: Request) -> dict:
    # Get the drawing at the end of each new stroke
    json_drawing = await request.json()
    # Extract strokes from json file
    strokes = json_drawing['drawing']
    # Process the drawing to the expect bmp format
    strokes = CNN_model.normalize_strokes(strokes)
    bmp = CNN_model.draw_image_from_strokes(strokes, size=28)
    X_CNN = np.expand_dims(bmp,0)

    #predict - this is gonna give me an array with the percentage that is in each class
    res_CNN = model_CNN.predict(X_CNN)[0]
    # this will rturn the index of the highest percentage prediction, we need to map this to its key
    rev_dict_50_class_subset = {0: 'aircraft carrier', 1: 'arm', 2: 'asparagus', 3: 'backpack', 4: 'banana', 5: 'basketball',
                                6: 'bottlecap', 7: 'bread', 8: 'broom', 9: 'bulldozer', 10: 'butterfly', 11: 'camel', 12: 'canoe',
                                13: 'chair', 14: 'compass', 15: 'cookie', 16: 'drums', 17: 'eyeglasses', 18: 'face', 19: 'fan',
                                20: 'fence', 21: 'fish', 22: 'flying saucer', 23: 'grapes', 24: 'hand', 25: 'hat', 26: 'horse',
                                27: 'light bulb', 28: 'lighthouse', 29: 'line', 30: 'marker', 31: 'mountain', 32: 'mouse',
                                33: 'parachute', 34: 'passport', 35: 'pliers', 36: 'potato', 37: 'sea turtle', 38: 'snowflake',
                                39: 'spider', 40: 'square', 41: 'steak', 42: 'swing set', 43: 'sword', 44: 'telephone', 45: 'television',
                                46: 'tooth', 47: 'traffic light', 48: 'trumpet', 49: 'violin'}

    prediction = rev_dict_50_class_subset(np.argmax(res_CNN))

    return prediction
