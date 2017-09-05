import argparse

import SimpleLogger as SimpleLogger

import importlib
import numpy as np

import os
import pickle

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils

#have to do this import to be able to use pyplot in the docker image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from IPython import display
import time
from model_utils import set_gpu_recursive, load_model, save_state, save_progress, maybe_save

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', nargs='+', type=int, default=0, help='gpu id')
parser.add_argument('--myseed', type=int, default=0, help='random seed')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--nepochs', type=int, default=250, help='total number of epochs')

parser.add_argument('--model_name', default='deep_conv', help='name of the model module')
parser.add_argument('--save_dir', default='./test_classifier/deep_conv/', help='save dir')
parser.add_argument('--save_progress_iter', type=int, default=1, help='number of iterations between saving progress')
parser.add_argument('--save_state_iter', type=int, default=10, help='number of iterations between saving progress')

parser.add_argument('--ndat', type=int, default=-1, help='Number of data points to use')
parser.add_argument('--optimizer', default='adam', help='type of optimizer, can be {adam, RMSprop}')
parser.add_argument('--train_module', default='train_single_target', help='training module')

parser.add_argument('--channels', nargs='+', type=int, default=[0, 1, 2], help='channels to use for part 1')

parser.add_argument('--data_save_path', default='./data/data.pyt', help='Save path for the dataprovider object')
parser.add_argument('--data_provider', default='DataProvider3Dh5', help='Dataprovider object')
parser.add_argument('--data_dir', default='/root/data/release_4_1_17/results_v2/aligned/2D', help='location of images')

opt = parser.parse_args()
print(opt)

DP = importlib.import_module("data_providers." + opt.data_provider)
model_provider = importlib.import_module("models." + opt.model_name)
train_module = importlib.import_module("train_modules." + opt.train_module)

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

#######
### GET DATA PROVIDER
#######

if os.path.exists(opt.data_save_path):
    dp = torch.load(opt.data_save_path)
else:
    data_save_dir = os.path.dirname(opt.data_save_path)
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    dp = DP.DataProvider(opt.data_dir)
    torch.save(dp, opt.data_save_path)

if opt.ndat == -1:
    opt.ndat = dp.get_n_dat('train')

iters_per_epoch = np.ceil(opt.ndat / opt.batch_size)

#######
### TRAIN CLASSIFIER
#######

opt.channelInds = opt.channels
dp.opts['channelInds'] = opt.channels
opt.nch = len(opt.channels)

opt.n_classes = dp.get_n_classes()

train_module = train_module.trainer(dp, opt)

pickle.dump(opt, open('./{0}/opt.pkl'.format(opt.save_dir), 'wb'))

models, optimizers, criterions, logger, opt = load_model(model_provider, opt)

#######
### MAIN LOOP
#######

start_iter = len(logger.log['iter'])
zAll = list()
for this_iter in range(start_iter, math.ceil(iters_per_epoch) * opt.nepochs):
    opt.iter = this_iter

    epoch = np.floor(this_iter / iters_per_epoch)
    epoch_next = np.floor((this_iter + 1) / iters_per_epoch)

    start = time.time()

    errors = train_module.iteration(**models, **optimizers, **criterions, dp=dp, opt=opt)

    errors_eval = train_module.evaluate(**models, **criterions, dp=dp, opt=opt)

    stop = time.time()
    deltaT = stop - start

    logger.add((epoch, this_iter) + errors + errors_eval + (deltaT,))

    maybe_save(epoch, epoch_next, models, optimizers, logger, dp, opt)

#######
### DONE TRAINING CLASSIFIER MODEL
#######
