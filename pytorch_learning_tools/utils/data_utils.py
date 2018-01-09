import torch

def indices_to_onehot(indices, nclasses = None):
    if nclasses == None:
        nclasses = len(np.unique(indices))
        
    onehot = torch.FloatTensor(len(indices), nclasses)
    onehot.zero_()
    onehot.scatter_(1, indices, 1)
    
    return onehot
   
