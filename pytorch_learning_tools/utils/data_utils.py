import torch
import numpy as np


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