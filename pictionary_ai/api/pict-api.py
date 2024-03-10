from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pictionary_ai.preprocessing import preprocessor
import json


app = FastAPI()
# app.state.model = main.load_model()  # make sure that the main contains the load_model() function


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



# get canvas capture and send back to streamlit the JSON
@app.get("/home")
def homepage():

    return "Welcome to the pictionary ai api"


@app.post("/api")
async def get_json(request: Request):
    json_drawing = await request.json()

    return json_drawing
