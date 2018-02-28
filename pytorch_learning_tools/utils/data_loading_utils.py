import torch
import numpy as np
from PIL import Image


def load_rgb_img(img_path, image_channels):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            img_arr = np.array(img.convert('RGB'))
            if image_channels is not (0,1,2):
                zero_channels = [i for i in (0,1,2) if i not in image_channels]
                img_arr[:,:,zero_channels] = 0
            return img_arr

def load_greyscale_tiff(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img

