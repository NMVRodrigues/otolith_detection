import streamlit as st
import pandas as pd
from PIL import Image
from app_utils import store_data, clear_data_storage, folder_selector, run_detection, make_results_dir
import numpy as np
from multiprocessing import Process, Queue
import os
from glob import glob
import sys

@st.cache_data
def load_image(path):
    uploaded_file_path, temp_path = store_data(path)
    img = Image.open(uploaded_file_path)
    return img, uploaded_file_path, temp_path

@st.cache_data
def get_output_path(path):
    output_path = make_results_dir(path)
    return output_path

left, center, right = st.columns([1, 4, 1])

option = None

if st.button("Quit app 🚫"):
    st.stop()
    
with center:
    loader = st.file_uploader("Load the image 📷")
    if loader is not None:
        img, uploaded_path, temp_path = load_image(loader)
        st.image(img,
                 clamp=True,
                 use_column_width=True
                 )

        option = st.selectbox(
            'Chose the model 🤖',
            ('None', 'Otolith Detection'))

        if option != 'None':
            model = option

        file_format = st.selectbox(
            'Chose the file format 🤖',
            ('.jpg', '.png'))

        if st.button("Run model ⚙️"):
            output_path = get_output_path(uploaded_path.split(os.sep)[-1])
            with st.spinner(f"Running {model}..."):
                output_images = run_detection(uploaded_path, output_path, option, file_format)
                st.markdown(''':green[Success!]''')
                # need to chage this so that it shows the images in a row
                st.image(output_images[0],
                 clamp=True,
                 use_column_width=True
                 )
                st.image(output_images[1],
                 clamp=True,
                 use_column_width=True
                 )
                #clear_data_storage(temp_path)



