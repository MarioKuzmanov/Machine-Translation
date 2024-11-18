import streamlit as st

st.html("<h1 align='center'>Grapheme to Phoneme Prediction</h1>")

st.markdown("""
1. **Task**

Create computational model that map native orthography lemmas ("graphemes") to valid IPA transcriptions of phonemic pronunciation.
This task is crucial for speech processing, namely text-to-speech synthesis.""")


st.markdown("""
2. **Evaluation**

The metric used to rank systems is word error rate (WER), the percentage of words for which the hypothesized transcription sequence does not match the gold transcription. 
This value, in accordance with common practice, is a decimal value multiplied by 100 (e.g.: 13.53).""")



st.markdown("""
3. **More Information**

[Check-out real task](https://github.com/sigmorphon/2024G2PST)
""")

st.markdown("""
4. **Important**

The dataset consists of a number of target languages, for my purposes I simplified a lot of the things and used only the bulgarian dataset for 
training and evaluation. 
""")