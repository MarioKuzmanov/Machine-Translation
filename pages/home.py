import streamlit as st
import time
from implementation.inference import Inference

st.markdown("""<h1 align='center'>Grapheme to Phoneme</h1>
                <h3 align='center'>2024 Shared Task</h3>
            """, unsafe_allow_html=True)

inference_container = st.container(border=True)
transcription_container = st.container(border=True)

inference_container.markdown("### Inference")

w = inference_container.text_input('Enter word in bulgarian/Get phonetic transcription')

t = ""
if w.strip():
    with st.spinner('Loading...'):
        time.sleep(1)
        infer = Inference(path2model="bestmodel/translator.pt")
        try:
            t = infer.predict((w, "a"))
        except KeyError:
            st.warning("The input has to consist only of cyrillic characters and no whitespaces.")

transcription_container.markdown("### Model Transcription")

transcription_container.markdown(f'<b>transcribed: {t}</b>', unsafe_allow_html=True)
