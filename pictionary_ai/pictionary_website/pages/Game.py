import streamlit as st
import time
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain

st.set_page_config(page_icon=":pencil:")

st.title('Pictionary :blue[AI] :pencil:')

st.markdown("We will randomly choose one out of 345 images for you to draw in 20 seconds...")
add_vertical_space(3)

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
