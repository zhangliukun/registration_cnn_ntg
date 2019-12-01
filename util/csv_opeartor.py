import pandas as pd
import os

def read_csv_file(file_path):
    df = pd.read_csv(file_path)

    return df


def extract_image_files(path):
    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
    vid_formats = ['.mov', '.avi', '.mp4']

    with open(path,'r') as f:
        img_files = [x.replace('/',os.sep) for x in f.read().splitlines()
                 if os.path.splitext(x)[-1].lower() in img_formats]

    return img_files