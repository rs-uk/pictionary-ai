from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pictionary_ai.preprocessing import preprocessor
import json


app = FastAPI()
# app.state.model = registry.load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/predict")
def predict(json_drawing: json):
    '''
    Make a single drawing guess.
    Assumes json_drawing is a json of a drawing as provided by the front-end capture tool:
        - the key is 'drawing'
        - the value is a list of strokes
            - the strokes are lists of coordinates
                - the coordinates are lists of xs and ys
    '''
    X_pred =
