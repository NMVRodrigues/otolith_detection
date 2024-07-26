from ultralytics import YOLO
from PIL import Image
import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--ptype",
        dest="pred_type",
        type=str,
        help="Prediction type [img, folder]",
        required=True,
    )
    parser.add_argument(
        "--name",
        dest="input_name",
        type=str,
        help="Name of the input file or folder",
        required=True,
    )

    parser.add_argument(
        "--ftype",
        dest="ftype",
        type=str,
        help="File type, defaults to jpg",
        default=".jpg",
    )

    parser.add_argument(
        "--o",
        dest="output_path",
        type=str,
        help="Output path to where the predictions are saved, defauts to local folder",
        default=os.getcwd()
    )
    

    args = parser.parse_args()

    model = YOLO("best.pt")

    if args.pred_type == "img":

        img_id = args.input_name.split(os.sep)[-1].split(args.ftype)[0]

        img = Image.open(args.input_name)
        
        predictions = model.predict(img, stream=False, imgsz=[1920,1080])

        topb = predictions[0].boxes[0]
        botb = predictions[0].boxes[1]

        for side, box in zip(["left", "right"], [topb, botb]):
            x, y, w, h = box.xywhn.cpu().numpy()[0].astype(float)
            # padding of 15 pixels
            x, y, w, h = int(x*1280), int(y*960), (int(w*1280))//2, (int(h*960))//2
            cropped = img.crop((x-w, y-h, x+w, y+h))
            cropped.save(os.path.join(args.output_path, img_id+"_"+side+args.ftype))

    elif args.pred_type == "folder":
        images = os.listdir(args.input_name)

        with tqdm(total=len(images)) as pbar:
            for i, image in enumerate(images):
                img_id = image.split(os.sep)[-1].split(args.ftype)[0]
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
                    cropped.save(os.path.join(args.output_path, image.split(args.ftype)[0]+"_"+side+args.ftype))




