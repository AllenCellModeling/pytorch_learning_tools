import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb


class trainer(object):
    def __init__(self, dp, opt):
        
        gpu_id = opt.gpu_ids[0]
        
        self.x = Variable(dp.get_images(range(0, opt.batch_size),'train').cuda(gpu_id))
        
        if opt.n_classes > 0:
            self.classes = Variable(torch.LongTensor(opt.batch_size)).cuda(gpu_id)
        else:
            self.classes = None
        
    def iteration(self, model, param, crit, dataProvider, opt):
        gpu_id = opt.gpu_ids[0]

        rand_inds_encD = np.random.permutation(opt.ndat)
        inds = rand_inds_encD[0:opt.batch_size]
        
        self.x.data.copy_(dataProvider.get_images(inds,'train'))
        x = self.x
        
        self.classes.data.copy_(dataProvider.get_classes(inds,'train'))
        classes = self.classes

        param.zero_grad()
            
        ## train the classifier
        classes_pred = model(x)

        pred_loss = crit(classes_pred, classes)
        pred_loss.backward(retain_graph=True)        
        pred_loss = pred_loss.data[0]
       
        errors = (pred_loss,)
        
        param.step()

        return errors
    