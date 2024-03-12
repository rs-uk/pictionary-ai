import streamlit as st
import time
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain
import requests
import random

st.set_page_config(page_icon=":pencil:")

st.title('Pictionary :blue[AI] :pencil:')

st.markdown("We will randomly choose one out of 50 images for you to draw in 20 seconds...")
add_vertical_space(3)

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
    progress_bar = st.progress(0)
    countdown_placeholder = st.empty()

    for i in range(countdown_seconds, 0, -1):
        countdown_placeholder.text(f"{i} seconds left!")
        progress_bar.progress((countdown_seconds - i + 1) / countdown_seconds)
        time.sleep(1)


    countdown_placeholder.empty()
    st.success("Your time is up!")

def main():
    countdown_seconds = 20
    start_countdown_button = st.button("Play game!")


    if start_countdown_button:
        countdown_with_progress(countdown_seconds)





if __name__ == "__main__":
    main()



add_vertical_space(10)

image1 = '/Users/gregorytaylor/code/pictionary-ai/raw_data/preview.jpg'
st.image(image1, width=710)

#trying to add in prediction
predict_url = "http://localhost:8080/api"

post_dict = #this is the dictionary received from loic
res = requests.post(url=predict_url, json=post_dict, headers={'Content-Type':'application/json'})
#this request returns a dictionary with the array of percentages and the hgihest class

#decode turns to string, eval turns into dictionary
eval(res.decode())['prediction']


#this is the predicted class
class_pred = reversed_dict[int(eval(res.decode())['prediction'])]


#while the predicted class doesn't equal the actual class, it should write it
#- in future we want it to speak it
# and than when guesses right we want it to stop the clock and maybe say you won
while class_pred != random_class:
    st.write(reversed_dict[int(eval(res.decode())['prediction'])])
    class_pred == random_class
else: # do we wnat
    st.write(reversed_dict[int(eval(res.decode())['prediction'])])
