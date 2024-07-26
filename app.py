import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os
import shutil
from tqdm import tqdm

# Load the YOLOv8 model
model = YOLO('best.pt')  # or use your custom trained model

def process_images(input_path, output_folder, progress=gr.Progress()):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if os.path.isfile(input_path):
        # Process single file
        file_name, file_extension = os.path.splitext(input_path)
        output_images = []

        img = Image.open(input_path)
        predictions = model.predict(img, stream=False, imgsz=[1920,1080])

        topb = predictions[0].boxes[0]
        botb = predictions[0].boxes[1]

        for side, box in zip(["left", "right"], [topb, botb]):
            x, y, w, h = box.xywhn.cpu().numpy()[0].astype(float)
            # padding of 15 pixels
            x, y, w, h = int(x*1280), int(y*960), (int(w*1280))//2, (int(h*960))//2
            cropped = img.crop((x-w, y-h, x+w, y+h))
            print(os.path.join(output_folder, file_name.split(os.sep)[-1] + "_" + side + file_extension))
            cropped.save(os.path.join(output_folder, file_name.split(os.sep)[-1] + "_" + side + file_extension))
            output_images.append(cropped)

        return f"Processed 1 image. Saved to {output_folder}", img, output_images[0], output_images[1]
    else:
        return "Invalid input path"

# Create the Gradio interface
iface = gr.Interface(
    fn=process_images,
    inputs=[
        gr.File(label="Input Image", type="filepath"),
        gr.Textbox(label="Output Folder", elem_id="output_folder")
    ],
    outputs=[
        gr.Textbox(label="Processing Result"),
        gr.Image(label="Input Image"),
        gr.Image(label="Output Image with Detections Left"),
        gr.Image(label="Output Image with Detections Right")
    ],
    title="Otolith Detection and cropping",
    description="Upload an image or folder.",
)

# Launch the interface
iface.launch()