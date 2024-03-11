from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pictionary_ai.main import preprocessor
import ujson
import requests
import numpy as np
from pictionary_ai.model import models


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
@app.post("/api")
async def get_json(request: Request):
    # Get the drawing at the end of each new stroke
    json_drawing = await request.json()
    # Process the drawing to the expect list format
    list_processed_drawing = (preprocessor.process_drawing_data(json_drawing)).tolist()
    # Should we pad the drawing??
    list_padded_drawing = preprocessor.add_padding(list_processed_drawing)
    X_processed = np.expand_dims(list_padded_drawing,0)

    # initiate model
    model = models.model_bidirectional()
    model = models.compile_model(model)
    #load wieghts
    model.load_weights('../raw_data/models')
    #predict - this is gonna give me an array with the percentage that is in each class
    res = model.predict(X_processed)[0]
    prediction = np.argmax(res)
    # this will rturn the index of the highest percentage prediction, we need to map this to its key

    # y_pred = app.state.model.predict(X_processed)

    return json_drawing, prediction
