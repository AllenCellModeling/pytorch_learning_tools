import torch
import numpy as np
from PIL import Image
import h5py
from .data_utils import eight_bit_to_float

def load_rgb_img(img_path, *args, **kwargs):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_greyscale_tiff(img_path, *args, **kwargs):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def load_h5(h5_path, image_channels, *args, **kwargs):
    f = h5py.File(h5_path, 'r')
    image = f['image'].value[image_channels, ::]
    image = eight_bit_to_float(image)
    return torch.from_numpy(image)

def pick_image_loader(image_type):
    if image_type in ['jpg','JPG','jpeg','JPEG','png','PNG','ppm','PPM','bmp','BMP']:
        return load_rgb_img
    elif image_type in ['tif', 'TIF', 'tiff', 'TIFF']:
        return load_greyscale_tiff
    else:
        raise ValueError('image_type {} is not supported, only basic images'.format(image_type))
