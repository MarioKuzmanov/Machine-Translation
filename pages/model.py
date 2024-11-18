import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st

text_container = st.container(border=True)

with text_container:
    text_container.markdown("<h1 align='center'>Encoder-Decoder</h1>", unsafe_allow_html=True)

    text_container.write("""My implementation relies on the Encoder-Decoder architecture which is quite popular for MT tasks. The 
    `encoder` hidden vector, represents the sequence in the input language ; in this case of the bulgarian 
    graphemes and is learnt by three-layered bidirectional LSTM units. Each input grapheme sequence is prepended and appended 
    artificial tags, and the output target sequence of phonemes has an artificial end tag. The sequences are processed 
    letter by letter, where each letter has a 200 dimensional embedding representation. Now, the task is cast into 
    language generation by deterministic sampling, we start with the first tag and given it, generate new phoneme. Given 
    our generations so far with some certainty we continue until the artificial end tag is sampled. To maximize the 
    probability of selecting the correct next token, a separate `decoder` network (uni-directional LSTMs) is used. During 
    training, teacher forcing is enforced to help reduce possibly erroneous behaviour from the model. During testing, 
    a sequence is being produced until end tag or maximum length (hyperparameter) is reached.
    I measure the distance between the model distribution and the correct one with `CrossEntropy` as a loss function and optimize both 
    networks with `AdamW`, further hyperparameters can be found in the repository. """)

    text_container.write(""" The model was trained for about 50 epochs on the training dataset with evaluation on a validation dataset every **5 epochs**. 
    In case there is no improvement in the WER score ( it doesn't get smaller ) on the validation data, I stop the training. The final WER in the training was **22,4** 
    during the 30th step. The WER on the test data is **10,5**.
    """)
