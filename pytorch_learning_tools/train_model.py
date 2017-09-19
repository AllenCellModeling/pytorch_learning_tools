import os
import pickle
import time
import math
import importlib

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from .SimpleLogger import SimpleLogger
from .model_utils import set_gpu_recursive, load_model, save_state, save_progress, maybe_save

import fire


def train_model(gpu_ids=0,
                myseed=0,
                lr=0.0005,
                batch_size=64,
                nepochs=250,
                model_name='deep_conv',
                save_dir='./test_classifier/deep_conv/',
                save_progress_iter=1,
                save_state_iter=10,
                ndat=-1,
                optimizer='adam',
                train_module='train_single_target',
                channels=[0, 1, 2],
                data_save_path='./data/data.pyt',
                data_provider='DataProvider3Dh5',
                data_dir='/root/data/release_4_1_17/results_v2/aligned/2D'):
    """
    Trains a pytorch model.
    Args:
        gpu_ids (int) default=0: gpu id
        myseed (int) default=0: random seed
        lr (float) default=0.0005: learning rate
        batch_size (int) default=64: batch size
        nepochs (int) default=250: total number of epochs
        model_name' (string) default='deep_conv': name of the model module
        save_dir' (string) default='./test_classifier/deep_conv/': save dir
        save_progress_iter (int) default=1: number of iterations between saving progress
        save_state_iter (int) default=10: number of iterations between saving progress
        ndat (int) default=-1: Number of data points to use
        optimizer (string) default='adam': type of optimizer, can be {adam, RMSprop}
        train_module (string) default='train_single_target': training module
        channels' (list) default=[0, 1, 2]: channels to use for part 1
        data_save_path (string) default='./data/data.pyt': Save path for the dataprovider object
        data_provider (string) default='DataProvider3Dh5': Dataprovider object
        data_dir (string) default='/root/data/release_4_1_17/results_v2/aligned/2D': location of images
    """

    # use this for pickling the options passed in if desired at some point
    arg_dict = locals()

    DP = importlib.import_module("data_providers." + data_provider)
    model_provider = importlib.import_module("models." + model_name)
    train_module = importlib.import_module("train_modules." + train_module)

    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    np.random.seed(myseed)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # GET DATA PROVIDER
    if os.path.exists(data_save_path):
        dp = torch.load(data_save_path)
    else:
        data_save_dir = os.path.dirname(data_save_path)
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)

        dp = DP.DataProvider(data_dir)
        torch.save(dp, data_save_path)

    if ndat == -1:
        ndat = dp.get_n_dat('train')

    iters_per_epoch = np.ceil(ndat / batch_size)

    # TRAIN CLASSIFIER
    channelInds = channels
    dp.opts['channelInds'] = channels
    nch = len(channels)

    n_classes = dp.get_n_classes()

    train_module = train_module.trainer(dp, opt)

    models, optimizers, criterions, logger, opt = load_model(model_provider, opt)

    # MAIN LOOP
    start_iter = len(logger.log['iter'])
    zAll = list()
    for this_iter in range(start_iter, math.ceil(iters_per_epoch) * nepochs):
        iter = this_iter

        epoch = np.floor(this_iter / iters_per_epoch)
        epoch_next = np.floor((this_iter + 1) / iters_per_epoch)

        start = time.time()

        errors = train_module.iteration(**models, **optimizers, **criterions, dp=dp, opt=opt)
        errors_eval = train_module.evaluate(**models, **criterions, dp=dp, opt=opt)

        stop = time.time()
        deltaT = stop - start

        logger.add((epoch, this_iter) + errors + errors_eval + (deltaT,))
        maybe_save(epoch, epoch_next, models, optimizers, logger, dp, opt)


# if called from the command line, args are `--argname argvalue` style.
# call with no args for help/info.
if __name__ == "__main__":
    fire.Fire(check_csv_column_contents)

