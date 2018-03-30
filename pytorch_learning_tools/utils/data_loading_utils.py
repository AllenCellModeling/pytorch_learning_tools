import torch
import numpy as np
from PIL import Image
import h5py
from .data_utils import eight_bit_to_float


def loadPILImages(paths, mode='L', *args, **kwargs):
    tensors = []
    for path in paths:
        with open(path, 'rb') as f:
            img = Image.open(f)
            tensors += [transforms.ToTensor()(img.convert(mode))]
    return tensors

def loadH5images(paths, channels=(3,2,4), *args, **kwargs):
    tensors = []
    for path in paths:
        f = h5py.File(path, 'r')
        image = f['image'].value[channels, ::]
        image = eight_bit_to_float(image)
        tensors += [torch.from_numpy(image)]
    return tensors

