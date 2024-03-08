import streamlit as st
from gtts import gTTS
import io

def main():
    st.title("Automatic Text-to-Speech Demo")
    user_input = st.text_input("Enter a word:")

    # Check if the user input is not empty
    if user_input:
        tts = gTTS(text=user_input, lang="en", slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    main()
