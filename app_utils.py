
import os
import shutil
import random
import string
import streamlit as st
from tkinter import Tk, filedialog
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# name random name for temp folder
def get_random_string(length):
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    return result_str


# clear temp data
def clear_data_storage(path):

    if os.path.isfile(path):
        os.remove(path)

    if os.path.isdir(path):
        shutil.rmtree(path)

def store_data(file):
    temp_data_directory = os.path.join(os.getcwd(), 'output', get_random_string(15))
    os.makedirs(temp_data_directory, exist_ok=True)
    
    uploaded_file_path = os.path.join(temp_data_directory, file.name)
    with open(uploaded_file_path, 'wb') as f:
        f.write(file.getbuffer())

    return uploaded_file_path, temp_data_directory

def make_results_dir(filename):
    results_dir_path = os.path.join(os.getcwd(), 'output', filename)
    os.makedirs(results_dir_path, exist_ok=True)

    return results_dir_path

#@st.cache_data
def image_selector(_queue):
    root = Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    file = filedialog.askopenfilename(multiple=False)
    _queue.put(file)

def select_folder():
    root = Tk()
    root.withdraw()  
    root.attributes('-topmost', True) 
    
    folder_path = filedialog.askdirectory()
    
    if folder_path:
        # Ensure we have the full path
        full_path = os.path.abspath(folder_path)
        st.session_state['selected_folder'] = full_path
        return full_path
    else:
        return None

def run_detection(input_path, output_path, file_upload, file_format, checkpoint_name="best.pt",  tmp_dir=".tmp"):

    model = YOLO(checkpoint_name)

    if file_upload == "single":

        img_id = input_path.split(os.sep)[-1].split(file_format)[0]

        img = Image.open(input_path)

        output_images = []
        
        predictions = model.predict(img, stream=False, imgsz=[1920,1080])

        topb = predictions[0].boxes[0]
        botb = predictions[0].boxes[1]

        for side, box in zip(["left", "right"], [topb, botb]):
            x, y, w, h = box.xywhn.cpu().numpy()[0].astype(float)
            # padding of 15 pixels
            x, y, w, h = int(x*1280), int(y*960), (int(w*1280))//2, (int(h*960))//2
            cropped = img.crop((x-w, y-h, x+w, y+h))
            cropped.save(os.path.join(output_path, img_id+"_"+side+file_format))
            output_images.append(cropped)

        return output_images
    
    elif file_upload == "batch":

        images = os.listdir(input_path)

        with tqdm(total=len(images)) as pbar:
            for i, image in enumerate(images):
                img_id = image.split(os.sep)[-1].split(file_format)[0]
                pbar.set_description("Predicting {}".format(img_id))
                print(f'Processing image {i+1}/{len(images)}')
                img = Image.open(image)

                predictions = model.predict(img, stream=False, imgsz=[1920,1080])

                topb = predictions[0].boxes[0]
                botb = predictions[0].boxes[1]

                for side, box in zip(["left", "right"], [topb, botb]):
                    x, y, w, h = box.xywhn.cpu().numpy()[0].astype(float)
                    # padding of 15 pixels
                    x, y, w, h = int(x*1280), int(y*960), (int(w*1280))//2, (int(h*960))//2
                    cropped = img.crop((x-w, y-h, x+w, y+h))
                    cropped.save(os.path.join(output_path, img_id+"_"+side+file_format))
        
        return None

