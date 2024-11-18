import streamlit as st

import pandas as pd

df_train = pd.read_csv("data/bul_train.tsv", sep="\t", names=['grapheme sequence', 'phoneme sequence'])

df_val = pd.read_csv("data/bul_val.tsv", sep="\t", names=['grapheme sequence', 'phoneme sequence'])
df_test = pd.read_csv("data/bul_test.tsv", sep="\t", names=['grapheme sequence', 'phoneme sequence'])

st.markdown("<h1 align='center'>Available Data</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 align='center'>Examples: {len(df_train) + len(df_val) + len(df_test)}</h3>", unsafe_allow_html=True)
st.markdown('---')

st.subheader("Training Pairs")
st.dataframe(df_train, width=1000, height=500)
st.write(f"Total examples: {len(df_train)}")

st.markdown('---')

st.subheader("Validation Pairs")
st.dataframe(df_val, width=1000, height=500)
st.write(f"Total examples: {len(df_val)}")

st.markdown('---')

st.subheader("Test Pairs")
st.dataframe(df_test, width=1000, height=500)
st.write(f"Total examples: {len(df_val)}")

st.markdown('---')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Graphemes")
    graphemes = ['ч', 'з', 'я', 'т', 'в', 'ь', 'г', 'с', 'и', 'п', 'е', 'ю', 'й', 'д', 'ж', 'л', 'к', 'а', 'ш', 'м',
                 'ц',
                 'у', 'р', 'ѝ', 'о', 'ъ', 'х', 'ф', 'н', 'б', 'щ']

    st.write(graphemes)

with col2:
    st.subheader("Phonemes")
    phonemes = ['kʲ', 'd', 'v', 'ʊ', 'ĩ', 't', 's̪', 'lˠ', 'ŋ', 'n̪', 't̪', 'fʲ', 'vʲ', 's', 't͡s', 'ɔ', 'o', 'ɛ̃',
                'r', 'ɑ', 'ə', 'ũ', 't͡sʲ', 'ɫ', 'd̠ʒ', 'p', 'u', 'o̟', 'l', 'm', 'rʲ', 'ɛ', 'f', 'ʒ', 'ʉ', 'o̝', 'tʲ',
                'j', 't͡ʃ', 'æ', 'mʲ', 'nʲ', 'ɡ', 'u̟', 'e', 'zʲ', 'ʃ', 'bʲ', 'pʲ', 'sʲ', 'd͡ʒ', 'a̟', 't̠ʃ', 'ɐ', 'n',
                'z', 'kʲʰ', 'iː', 'k', 'xʲ', 'ɡʲ', 'x', 'a', 'ɤ̃', 'g', 'i', 'lʲ', 'ʊ̯', 'pʰ', 'dʲ', 'tʲʰ', 'ɱ', 'ɑ̃',
                'ɤ̟', 'kʰ', 'ɔ̃', 'pʲʰ', 'b', 'tʰ', 'ɤ']
    st.write(phonemes)

st.subheader("Data Generation Methods")
st.write("""
The data is scraped from Wiktionary using WikiPron (Lee et al. 2020). Data is augmented with IPA transcriptions produced by GPT. 
All data is validated by organizers.
""")
