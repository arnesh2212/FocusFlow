import streamlit as st
import os

# Load the text content from the TXT file

#center h1 title called FocusFlow

st.markdown("<h1 style='text-align: center; color: white;'>FocusFlow</h1>", unsafe_allow_html=True)


#image BG.jpeg
from PIL import Image

image = Image.open('BG.jpeg')
st.image(image, caption='FocusFlow',use_column_width=True)
# Load the text content from the TXT filehttps://tricky-core-981052.framer.app/
#