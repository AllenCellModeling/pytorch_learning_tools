import torch
import numpy as np
from torch.utils.data.dataloader import default_collate as collate

def indices_to_onehot(indices, nclasses = None):
    if nclasses == None:
        nclasses = len(np.unique(indices))

    ndat = len(indices)

    is_cuda = False
    if indices.is_cuda:
        is_cuda = True
        cuda_device = indices.get_device()

    onehot = torch.FloatTensor(ndat, nclasses)
    onehot.zero_()

    if is_cuda:
        onehot = onehot.cuda(cuda_device)

    onehot[np.arange(0, ndat), indices] = 1

    return onehot

def eight_bit_to_float(im, dtype=np.uint8):
    imax = np.iinfo(dtype).max  # imax = 255 for uint8
    im = im / imax
    return im

def classes_and_weights(dp):
    """search through a dataprovider for unique target classes and return two lists:
       classes, weights. weights are inversely proportional to class abundances"""
    classes, abundances = np.unique(dp._df[dp._target_col], return_counts=True)
    weights = 1.0/abundances
    weights = weights/np.sum(weights)
    return torch.from_numpy(classes), torch.from_numpy(weights).type(torch.FloatTensor)

def make_channel_mask(A, masked_channels=(0,1), channel_dim=0):
    mask = torch.ones_like(A)
    indices = {channel_dim: masked_channels}
    ix = [indices.get(dim, slice(None)) for dim in range(len(A.shape))]
    mask[ix] = 0
    return mask
