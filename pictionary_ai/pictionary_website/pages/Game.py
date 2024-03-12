import streamlit as st
import time
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain

st.set_page_config(
            page_title="Game", # => Quick reference - Streamlit
            page_icon=":pencil:",
            layout="centered", # wide
            initial_sidebar_state="auto")

st.title('    Pictionary :blue[AI] :pencil:')
add_vertical_space(1)
st.markdown("We will randomly choose one out of 50 images for you to draw in 20 seconds...")
add_vertical_space(3)

def countdown_with_progress(countdown_seconds): #function for time limit on drawing
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
