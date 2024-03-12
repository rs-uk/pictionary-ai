import streamlit as st
import time
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain
import requests
import random



import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
import requests


# from params import *

import matplotlib.pyplot as plt
import random



# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3, # this can be adjusted during testing
    stroke_color='#000000', # we only want to draw in black
    background_color='#ffffff', # we set the background color to white
    background_image=None, # we do not need a backgorund image on the canvas
    update_streamlit=True, # we want the output to be live
    height=400,
    width=600,
    drawing_mode='freedraw', # we only want that option from st_canvas
    point_display_radius=0, # we only care about freedraw mode here
    key="canvas"
)


# Show the outputs on streamlit
if canvas_result.json_data is not None:
    # need to convert obj to str because of PyArrow
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    # st.dataframe(objects)


# Show the resulting JSON on streamlit
st.json(canvas_result.json_data['objects'])


# Extract the drawing and process to match the expected format
outputs = canvas_result.json_data['objects']
lst_strokes = []
# Going stroke by stroke:
for stroke in outputs:
    stroke = stroke['path'] # we only want the 'path' of the stroke in the JSON
    xs = []
    ys = []
    for step in stroke: # the steps are either one or two points
        # Build list of xs and ys.
        # Only 1 point for the first and last steps of each stroke ('M' and 'L')
        if step[0] == 'M' or step[0] == 'L':
            xs.append(int(step[1]))
            ys.append(int(step[2]))
        # 2 points for the intermediary steps of each stroke ('Q')
        elif step[0] == 'Q':
            # Adding both sets of coords to x and y
            xs.append(int(step[1]))
            xs.append(int(step[3]))
            ys.append(int(step[2]))
            ys.append(int(step[4]))
    lst_strokes.append([xs, ys])
dict_strokes = {'drawing': lst_strokes}
# Convert the dict to JSON
json_drawing = json.dumps(dict_strokes)


# Resampling the drawing (all strokes) based on time
def resampling_time_post(json_drawing: json, step: int = 5) -> dict:
    lst_strokes = json.loads(json_drawing)['drawing']
    lst_strokes_resampled = []
    for stroke in lst_strokes:
        stroke_resampled = []
        stroke_resampled.append(stroke[0][::step]) # resampled xs
        stroke_resampled.append(stroke[1][::step]) # resampled ys
        lst_strokes_resampled.append(stroke_resampled)
    dict_strokes_resampled = {'drawing': lst_strokes_resampled}
    return dict_strokes_resampled




st.set_page_config(page_icon=":pencil:")

st.set_page_config(
            page_title="Game", # => Quick reference - Streamlit
            page_icon=":pencil:",
            layout="centered", # wide
            initial_sidebar_state="auto")


st.title('Pictionary :blue[AI] :pencil:')

st.markdown("We will randomly choose one out of 50 images for you to draw in 20 seconds...")
add_vertical_space(3)



def main():
    countdown_seconds = 20
    # making sure the play button is centralised, and that it disappears once pressed.
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
    with col2:
        start_countdown_button = st.empty()
    with col3:
        st.write("")

    if start_countdown_button.button(":blue[Play game!]"):
        start_countdown_button.empty()
        countdown_with_progress(countdown_seconds)

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
#reversing dictionary

#this will give a random class
random_class = random.choice(list(reversed_dict.values()))



def countdown_with_progress(countdown_seconds):

    canvas_result = st_canvas(fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3, # this can be adjusted during testing
    stroke_color='#000000', # we only want to draw in black
    background_color='#FFFFFF', # we set the background color to white
    background_image=None, # we do not need a background image on the canvas
    update_streamlit=True, # we want the output to be live
    height=400,
    width=600,
    drawing_mode='freedraw', # we only want that option from st_canvas
    point_display_radius=0, # we only care about freedraw mode here
    key="canvas")

    # Show the outputs on streamlit
    if canvas_result.json_data is not None:
        # need to convert obj to str because of PyArrow
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        # st.dataframe(objects)
    # Show the resulting JSON on streamlit
    #st.json(canvas_result.json_data['objects'])
    # Extract the drawing and process to match the expected format
    outputs = canvas_result.json_data['objects']
    lst_strokes = []
    # Going stroke by stroke:
    for stroke in outputs:
        stroke = stroke['path'] # we only want the 'path' of the stroke in the JSON
        xs = []
        ys = []
        for step in stroke: # the steps are either one or two points
            # Build list of xs and ys.
            # Only 1 point for the first and last steps of each stroke ('M' and 'L')
            if step[0] == 'M' or step[0] == 'L':
                xs.append(int(step[1]))
                ys.append(int(step[2]))
            # 2 points for the intermediary steps of each stroke ('Q')
            elif step[0] == 'Q':
                # Adding both sets of coords to x and y
                xs.append(int(step[1]))
                xs.append(int(step[3]))
                ys.append(int(step[2]))
                ys.append(int(step[4]))
        lst_strokes.append([xs, ys])
    dict_strokes = {'drawing': lst_strokes}
    # Convert the dict to JSON
    json_drawing = json.dumps(dict_strokes)
    # Resampling the drawing (all strokes) based on time

    progress_bar = st.progress(0)
    countdown_placeholder = st.empty()
      # begin_game()
    for i in range(countdown_seconds, 0, -1): #counts down backwards
        countdown_placeholder.text(f"{i} seconds left!") #verbal output
        progress_bar.progress((countdown_seconds - i + 1) / countdown_seconds)
        time.sleep(1)
    # predicting_class()
    countdown_placeholder.empty()
    st.success("Your time is up!") #What happens when the time runs out


countdown_with_progress(20)


    # def resampling_time_post(json_drawing: json, step: int = 5) -> dict:
    #     lst_strokes = json.loads(json_drawing)['drawing']
    #     lst_strokes_resampled = []
    #     for stroke in lst_strokes:
    #         stroke_resampled = []
    #         stroke_resampled.append(stroke[0][::step]) # resampled xs
    #         stroke_resampled.append(stroke[1][::step]) # resampled ys
    #         lst_strokes_resampled.append(stroke_resampled)
    #     dict_strokes_resampled = {'drawing': lst_strokes_resampled}
    #     return dict_strokes_resampled


    # another_game()




if __name__ == "__main__":
    main()
# Create a canvas component

# canvas_result = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#     stroke_width=3, # this can be adjusted during testing
#     stroke_color='#000000', # we only want to draw in black
#     background_color='#FFFFFF', # we set the background color to white
#     background_image=None, # we do not need a backgorund image on the canvas
#     update_streamlit=True, # we want the output to be live
#     height=400,
#     width=600,
#     drawing_mode='freedraw', # we only want that option from st_canvas
#     point_display_radius=0, # we only care about freedraw mode here
#     key="canvas"
# )
# # Show the outputs on streamlit
# if canvas_result.json_data is not None:
#     # need to convert obj to str because of PyArrow
#     objects = pd.json_normalize(canvas_result.json_data["objects"])
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     # st.dataframe(objects)
# # Show the resulting JSON on streamlit
# #st.json(canvas_result.json_data['objects'])
# # Extract the drawing and process to match the expected format
# outputs = canvas_result.json_data['objects']
# lst_strokes = []
# # Going stroke by stroke:
# for stroke in outputs:
#     stroke = stroke['path'] # we only want the 'path' of the stroke in the JSON
#     xs = []
#     ys = []
#     for step in stroke: # the steps are either one or two points
#         # Build list of xs and ys.
#         # Only 1 point for the first and last steps of each stroke ('M' and 'L')
#         if step[0] == 'M' or step[0] == 'L':
#             xs.append(int(step[1]))
#             ys.append(int(step[2]))
#         # 2 points for the intermediary steps of each stroke ('Q')
#         elif step[0] == 'Q':
#             # Adding both sets of coords to x and y
#             xs.append(int(step[1]))
#             xs.append(int(step[3]))
#             ys.append(int(step[2]))
#             ys.append(int(step[4]))
#     lst_strokes.append([xs, ys])
# dict_strokes = {'drawing': lst_strokes}
# # Convert the dict to JSON
# json_drawing = json.dumps(dict_strokes)
# # Resampling the drawing (all strokes) based on time
# def resampling_time_post(json_drawing: json, step: int = 5) -> dict:
#     lst_strokes = json.loads(json_drawing)['drawing']
#     lst_strokes_resampled = []
#     for stroke in lst_strokes:
#         stroke_resampled = []
#         stroke_resampled.append(stroke[0][::step]) # resampled xs
#         stroke_resampled.append(stroke[1][::step]) # resampled ys
#         lst_strokes_resampled.append(stroke_resampled)
#     dict_strokes_resampled = {'drawing': lst_strokes_resampled}
#     return dict_strokes_resampled





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
#reversing dictionary

#this will give a random class
random_class = random.choice(list(reversed_dict.values()))




# def begin_game():

    #randomly select one of the 50 classes to be drawn

    #canvas to come up on the page?

    #guesses every stroke (top 3 accuracy classes?), pings back and forth to the api
    #guesses to be plug through a voice transformer, akin to the google draw ui.

    #if there is over X% accuracy, game ends and countdown stops + congrats message
    # (this will likely have to be included in countdown_progress function)
    # +++ click to play again

    #if time ends, tell the user whether they have won or not ++ click to play again

def another_game():
    col1, col2, col3 = st.columns(3)

    col1.write("")  # Empty column
    col2.button("   Click here to play again")
    col3.write("")  # Empty column








add_vertical_space(10)


#image1 = '/Users/gregorytaylor/code/pictionary-ai/raw_data/preview.jpg'
#st.image(image1, width=710)

image1 = 'https://storage.googleapis.com/pictionary-ai-website-bucket/preview.jpg'
st.image(image1, width=710)


#trying to add in prediction
predict_url = "http://localhost:8080/predict"


post_dict = resampling_time_post(json_drawing)

res = requests.post(url=predict_url, json=post_dict, headers={'Content-Type':'application/json'})
#this request returns a dictionary with the array of percentages and the hgihest class


#decode turns to string, eval turns into dictionary
eval(res.decode())['prediction']


#this is the predicted class
class_pred = reversed_dict[int(eval(res.decode())['prediction'])]


#while the predicted class doesn't equal the actual class, it should write it

# in future we want it to speak it
# and than when guesses right we want it to stop the clock and maybe say you won
# def predicting_class():
#     countdown_placeholder = st.empty()
#     while class_pred != random_class:
#         st.write(reversed_dict[int(eval(res.decode())['prediction'])])
#         class_pred == random_class
#     else: #do we want
#         st.write(reversed_dict[int(eval(res.decode())['prediction'])])
#         st.write('Congratulations, you drew accurately')
#         countdown_placeholder.empty()

#- in future we want it to speak it
# and than when guesses right we want it to stop the clock and maybe say you won
while class_pred != random_class:
    st.write(reversed_dict[int(eval(res.decode())['prediction'])])
    class_pred == random_class
else: # do we wnat
    st.write(reversed_dict[int(eval(res.decode())['prediction'])])
