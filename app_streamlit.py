import streamlit as st
import pandas as pd
from PIL import Image
from app_utils import store_data, clear_data_storage, select_folder, run_detection, make_results_dir
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

#left, center, right = st.columns([1, 4, 1])

option = None

if st.button("Quit app 🚫"):
    st.stop()

if 'selected_folder' not in st.session_state:
    st.session_state['selected_folder'] = None
if 'uploaded_path' not in st.session_state:
    st.session_state['uploaded_path'] = None
    
#with center:
#if loader is not None:
option = st.selectbox(
    'Chose the model 🤖',
    ('None', 'Otolith Detection'))

if option != 'None':
    model = option

file_format = st.selectbox(
    'Chose the file format 🤖',
    ('.jpg', '.png'))

file_upload = st.selectbox(
    'Chose the upload mode 📤',
    ('single', 'batch'))

if file_upload == "batch":
    if st.button("Select Folder"):
        st.session_state['uploaded_path'] = select_folder()

elif file_upload == "single":
    loader = st.file_uploader("Load the image or folder 🖼️")
    if loader is not None:
        img, uploaded_path, temp_path = load_image(loader)
        st.session_state['uploaded_path'] = uploaded_path
        st.image(img,
                    clamp=True,
                    use_column_width=True
                    )

if st.button("Run model ⚙️"):
    output_path = get_output_path(st.session_state['uploaded_path'].split(os.sep)[-1])
    with st.spinner(f"Running {model}..."):
        output_images = run_detection(st.session_state['uploaded_path'], output_path, file_upload, file_format)
        st.markdown(''':green[Success!]''')
        clear_data_storage(temp_path)

        if file_upload == "single":
            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(output_images[0],
                            clamp=True,
                            use_column_width=True
                            )
            with col2:
                st.image(output_images[1],
                            clamp=True,
                            use_column_width=True
                            )
        elif file_upload == "batch":
            st.write("Batch upload concluded, images saved to output folder")



