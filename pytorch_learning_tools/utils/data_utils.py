import torch

def indices_to_onehot(indices, nclasses = None):
    if nclasses == None:
        nclasses = len(np.unique(index))
        
    onehot = torch.FloatTensor(len(index), nclasses)
    onehot.zero_()
    onehot.scatter_(1, index,1)
    
    return onehot
   
