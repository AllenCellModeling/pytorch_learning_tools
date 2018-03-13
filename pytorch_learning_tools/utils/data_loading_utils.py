import torch
import numpy as np
from PIL import Image
import h5py
from .data_utils import eight_bit_to_float

def load_rgb_img(img_path, image_channels, *args, **kwargs):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            img_arr = np.array(img.convert('RGB'))
            if image_channels is not (0,1,2):
                zero_channels = [i for i in (0,1,2) if i not in image_channels]
                img_arr[:,:,zero_channels] = 0
            return Image.fromarray(img_arr, mode='RGB')

def load_greyscale_tiff(img_path, *args, **kwargs):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img

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

