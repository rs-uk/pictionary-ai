import streamlit as st
import time
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain
import requests
import random

st.set_page_config(
            page_title="Game", # => Quick reference - Streamlit
            page_icon=":pencil:",
            layout="centered", # wide
            initial_sidebar_state="auto")

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

    # begin_game()
    for i in range(countdown_seconds, 0, -1): #counts down backwards
        countdown_placeholder.text(f"{i} seconds left!") #verbal output
        progress_bar.progress((countdown_seconds - i + 1) / countdown_seconds)
        time.sleep(1)



    countdown_placeholder.empty()
    st.success("Your time is up!") #What happens when the time runs out
    another_game()


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





def main():
    countdown_seconds = 5
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





if __name__ == "__main__":
    main()



add_vertical_space(10)

image1 = '/Users/gregorytaylor/code/pictionary-ai/raw_data/preview.jpg'
st.image(image1, width=710)

#trying to add in prediction
predict_url = "http://localhost:8080/api"

post_dict = [] #this is the dictionary received from
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
else: # do we want
    st.write(reversed_dict[int(eval(res.decode())['prediction'])])
