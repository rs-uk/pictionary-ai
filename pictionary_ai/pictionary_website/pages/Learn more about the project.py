import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_icon=":shark:")


st.title(':blue[Learn more about this project]')

add_vertical_space(3)
st.markdown("<h2 style='font-size: 1.5em; text-decoration: underline;'>Where to find the data</h2>", unsafe_allow_html=True)
st.markdown("* Click below to visualize all of the previous drawing data that Google collected from their model")
st.link_button("Previous examples", "https://quickdraw.withgoogle.com/data")



add_vertical_space(1)
st.markdown("* Click below to access all of the simplfied data that we used")
st.link_button("Simplified data", "https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified")


add_vertical_space(2)
st.markdown("<h2 style='font-size: 1.5em; text-decoration: underline;'>How This Project Was Created</h2>", unsafe_allow_html=True)

add_vertical_space(1)
st.markdown("Preprocessing")
st.markdown(" - There were four different formats of data available for us to use. We chose to use the simplified data, which contained 345 different drawings and 50 million different drawings in totality")



add_vertical_space(10)

image1 = '/Users/gregorytaylor/code/pictionary-ai/raw_data/preview.jpg'
st.image(image1, width=710)
