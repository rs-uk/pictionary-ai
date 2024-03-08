import streamlit as st
import pandas as pd
import numpy as np
import requests

import streamlit as st
import time

from streamlit_extras.add_vertical_space import add_vertical_space

image_path = "/Users/gregorytaylor/code/pictionary-ai/pictionary_ai.jpg"

st.set_page_config(page_icon=":shark:")



backgroundColor = "#F0F0F0"




st.markdown("<h1 style='text-align: center;'>Pictionary AI</h1>", unsafe_allow_html=True)




st.image(image_path, use_column_width=True)

st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 22px;'>Welcome to Pictionary AI, the transformative new website that can predict your drawings!</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 14px;'>You have 20 seconds to draw whatever image we give to you. After that we will test it to see if your drawing is good     enough for the AI model to match it.</h1>"
, unsafe_allow_html=True)





image1 = '/Users/gregorytaylor/code/pictionary-ai/raw_data/preview.jpg'
add_vertical_space(3)

col1, col2, col3 = st.columns(3)

# Add content to the columns
col1.write("       ")  # Empty column
col2.link_button("   Click here to play", "http://localhost:8502/Game")  # Link button in the middle
col3.write("")  # Empty column






add_vertical_space(3)
st.markdown("<h1 style='text-align: center; font-weight: normal; font-size: 18px;'>Below are some examples of the sorts of images you may be asked to draw.</h1>", unsafe_allow_html=True)


st.image(image1, width=710)



image1 = '/Users/gregorytaylor/code/pictionary-ai/raw_data/preview.jpg'
