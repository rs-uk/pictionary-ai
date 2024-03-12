from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pictionary_ai.main import preprocessor
import ujson
import requests
import numpy as np


from pictionary_ai.model import models_rs
dictionary = {"aircraft carrier": 0, "arm": 1, "asparagus": 2, "backpack": 3,
              "banana": 4, "basketball": 5, "bottlecap": 6, "bread": 7, "broom": 8,
              "bulldozer": 9, "butterfly": 10, "camel": 11, "canoe": 12, "chair": 13,
              "compass": 14, "cookie": 15, "drums": 16, "eyeglasses": 17, "face": 18,
              "fan": 19, "fence": 20, "fish": 21, "flying saucer": 22, "grapes": 23,
              "hand": 24, "hat": 25, "horse": 26, "light bulb": 27, "lighthouse": 28,
              "line": 29, "marker": 30, "mountain": 31, "mouse": 32, "parachute": 33,
              "passport": 34, "pliers": 35, "potato": 36, "sea turtle": 37, "snowflake": 38,
              "spider": 39, "square": 40, "steak": 41, "swing set": 42, "sword": 43,
              "telephone": 44, "television": 45, "tooth": 46, "traffic light": 47, "trumpet": 48, "violin": 49}
reversed_dict = {v: k for k, v in dictionary.items()}


# initiate model
model = models_rs.model_bidirectional()
model = models_rs.compile_model(model)
#load wieghts
model.load_weights('../raw_data/models/models_1003_50classes')


#we need the models at the end- as name of model and need it


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
async def get_prediction(request: Request):
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

    prob_dict = {}

    for key_, prob in zip(dictionary.keys(),res):

        prob_dict[key_]= str(np.round(prob,3))

    print(prediction)
    # this will rturn the index of the highest percentage prediction, we need to map this to its key

    # y_pred = app.state.model.predict(X_processed)

    return_dict = {'result':str(res), 'prob': str(np.max(res))
                   , 'prediction': str(prediction), 'probabilities':str(prob_dict)}

    return return_dict
