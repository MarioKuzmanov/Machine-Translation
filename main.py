import streamlit as st
from st_pages import get_nav_from_toml

st.set_page_config(page_title="G2P Translation", page_icon='images/sigmorphon.jpg')

st.logo("images/sigmorphon.jpg")
pg = st.navigation(get_nav_from_toml(".streamlit/pages.toml"))

pg.run()
