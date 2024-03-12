import streamlit as st
import pandas as pd
import numpy as np
import requests
import streamlit as st
import time

from streamlit_extras.add_vertical_space import add_vertical_space

image_path = "/Users/gregorytaylor/code/pictionary-ai/raw_data/pictionary_ai.jpg" #this needs to be amended

# st.set_page_config(page_icon=":pencil:") #this sets the icon for the tab

st.set_page_config(
            page_title="Homepage", # => Quick reference - Streamlit
            page_icon=":pencil:",
            layout="centered", # wide
            initial_sidebar_state="auto")







st.markdown("<h1 style='text-align: center;'>Pictionary AI</h1>", unsafe_allow_html=True) #centralised title




st.image(image_path, use_column_width=True) #pictionary ai logo image

st.markdown("<h1 style='text-align: center; font-weight: normal;font-size: 22px;'>Welcome to Pictionary AI, the transformative new website that can predict your drawings!</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 14px;'>You have 20 seconds to draw whatever image we give to you. After that we will test it to see if your drawing is good     enough for the AI model to match it.</h1>"
, unsafe_allow_html=True)





image1 = '/Users/gregorytaylor/code/pictionary-ai/raw_data/preview.jpg' #this also needs to be changed
add_vertical_space(3) #formatting

col1, col2, col3 = st.columns(3)


col1.write("")  # Empty column
col2.link_button(":blue[Click here to play the game]", "http://localhost:8501/Game")  # Link button in the middle
col3.write("")  # Empty column






add_vertical_space(3) #formatting
st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 18px;'>Below are some examples of the sorts of images you may be asked to draw.</h1>", unsafe_allow_html=True)

image1 = '/Users/gregorytaylor/code/pictionary-ai/raw_data/preview.jpg'

st.image(image1, width=710) #bottom of page image
