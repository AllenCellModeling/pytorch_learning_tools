from torch import nn
import torch
import pdb
from model_utils import init_opts

ksize = 4
dstep = 2

class model(nn.Module):
    def __init__(self, n_classes, nch, gpu_ids, opt=None):
        super(model, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = 2
        
        self.n_classes = n_classes
        
        self.main = nn.Sequential(
            nn.Conv3d(nch, 64, ksize, dstep, 1),
            nn.BatchNorm3d(64),
        
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, ksize, dstep, 1),
            nn.BatchNorm3d(128),
        
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, ksize, dstep, 1),
            nn.BatchNorm3d(256),
            
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, ksize, dstep, 1),
            nn.BatchNorm3d(512),
            
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 1024, ksize, dstep, 1),
            nn.BatchNorm3d(1024),
            
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 1024, ksize, dstep, 1),
            nn.BatchNorm3d(1024),
        
            nn.ReLU(inplace=True),
        )
        
        self.classOut = nn.Sequential(
            nn.Linear(1024*int(self.fcsize*1*1), self.n_classes),
            nn.LogSoftmax()
        )
        

    def forward(self, x):
        gpu_ids = self.gpu_ids
            
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 1024*int(self.fcsize*1*1))
                
        pred = nn.parallel.data_parallel(self.classOut, x, gpu_ids)
        
        return pred

