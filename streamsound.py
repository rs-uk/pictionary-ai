import streamlit as st
from gtts import gTTS
import io
from IPython.display import Audio
import base64

# Run this command if you want to try independantly:
# streamlit run streamsound.py 

import streamlit as st
from gtts import gTTS
import io

def autoplay_audio(audio):
    '''Function Purpose:
            The autoplay_audio function is designed to create an HTML audio player that plays audio automatically
            without requiring manual interaction (i.e., clicking the “play” button).
            It takes an audio bytes object as input.
        Function Implementation:
            data = audio.read(): Reads the audio bytes from the input audio object.
            b64 = base64.b64encode(data).decode(): Encodes the audio data using base64 encoding.
            md: Constructs an HTML snippet for the audio player.
            The <audio> tag includes the controls attribute (which displays audio controls like play, pause, and volume) 
            and the autoplay="true" attribute (which ensures automatic playback).
            The src attribute specifies the audio source as a base64-encoded data URI.
            The type attribute indicates that the audio is in MP3 format.
            st.markdown(md, unsafe_allow_html=True): Displays the HTML snippet within the Streamlit app.
        Usage:
            You can call this function with an audio bytes object (e.g., generated using gTTS).
            It will render an audio player in your Streamlit app that plays the audio automatically.
    '''
    data = audio.read() 
    b64 = base64.b64encode(data).decode()
    md = f"""
    <audio controls autoplay="true">
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(
        md,
        unsafe_allow_html=True,
        )        

def tts():
    '''
    Function Purpose:
        The tts function is designed to create a simple Streamlit app for text-to-speech (TTS) conversion.
        It allows the user to input a word or text, converts it to audio using gTTS, and plays it automatically.
    Function Implementation:
        st.title("Automatic Text-to-Speech Demo"): Sets the title of the Streamlit app.
        user_input = st.text_input("Enter a word:"): Creates a text input field where the user can enter a word or text.
        if user_input:: Checks if the user input is not empty and execure inner function play_word().
        tts = gTTS(text=user_input, lang="en", slow=False): Creates a gTTS object with the user input text and English language.
        audio_bytes = io.BytesIO(): Initializes an in-memory byte stream to store the audio data.
        tts.write_to_fp(audio_bytes): Writes the audio data to the byte stream.
        audio_bytes.seek(0): Resets the stream position to the beginning.
        autoplay_audio(audio_bytes): Calls the autoplay_audio function with the audio bytes.
    Usage:
        The user enters a word or text in the input field.
        The gTTS library converts the input to audio and stores it in the audio_bytes object.
        The autoplay_audio function plays the audio automatically.
    '''
    st.title("Automatic Text-to-Speech Demo")    
    
    def play_word(user_input):
        tts = gTTS(text=user_input, lang="es", tld="es", slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        #st.audio(audio_bytes, format="audio/mp3", start_time=0)
        audio_bytes.seek(0) 
        autoplay_audio(audio_bytes)
    
    user_input = st.text_input("Enter a word:", key="user_input")  # Add a unique key
    
    if user_input:
        play_word(user_input)
      
if __name__ == "__main__":
    tts()
