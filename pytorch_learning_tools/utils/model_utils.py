import torch
import importlib
import torch.optim as optim
from pytorch_learning_tools.SimpleLogger import SimpleLogger
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.misc
import pickle
import importlib

import matplotlib.pyplot as plt
from pytorch_learning_tools.imgToProjection import imgtoprojection

import pdb


def init_opts(opt, opt_default):
    vars_default = vars(opt_default)
    for var in vars_default:
        if not hasattr(opt, var):
            setattr(opt, var, getattr(opt_default, var))
    return opt


def set_gpu_recursive(var, gpu_id):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                if gpu_id != -1:
                    var[key] = var[key].cuda(gpu_id)
                else:
                    var[key] = var[key].cpu()
            except:
                pass
    return var


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_model(model_provider, opt):
    model = model_provider.model(opt['n_classes'], opt['nch'], opt['gpu_ids'], opt)
    model.apply(weights_init)

    gpu_id = opt['gpu_ids'][0]

    model.cuda(gpu_id)

    if opt['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=opt['lr'])
    elif opt['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt['lr'], betas=(0.5, 0.999))

    columns = ('epoch', 'iter', 'train_loss', 'train_acc', 'loss_eval', 'acc_eval', 'time')
    print_str = '[%d][%d] train_loss: %.6f train_acc: %.4f test_loss: %.6f test_acc: %.4f time: %.2f'

    logger = SimpleLogger(columns, print_str)

    if os.path.exists('{0}/model.pth'.format(opt['save_dir'])):
        print('Loading from ' + opt['save_dir'])

        checkpoint = torch.load('{0}/model.pth'.format(opt['save_dir']))

        model.load_state_dict(checkpoint['model'])
        model.cuda(gpu_id)

        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)

        logger = pickle.load(open('{0}/logger.pkl'.format(opt['save_dir']), "rb"))

    models = {'model': model}
    optimizers = {'optimizer': optimizer}

    criterions = dict()
    if opt['n_classes'] > 0:
        criterions['crit'] = nn.CrossEntropyLoss()
    else:
        criterions['crit'] = nn.BCEWithLogitsLoss()

    return models, optimizers, criterions, logger, opt


def maybe_save(epoch, epoch_next, models, optimizers, logger, dp, opt):
    saved = False
    if epoch != epoch_next and ((epoch_next % opt['save_progress_iter']) == 0 or
                                (epoch_next % opt['save_state_iter']) == 0):

        if (epoch_next % opt['save_progress_iter']) == 0:
            print('saving progress')
            save_progress(logger, opt['save_dir'])

        if (epoch_next % opt['save_state_iter']) == 0:
            print('saving state')
            save_state(**models, **optimizers, logger=logger, opt=opt)

        saved = True

    return saved


def save_progress(logger, save_dir):
    pickle.dump(logger, open('./{0}/logger_tmp.pkl'.format(save_dir), 'wb'))


def save_state(model, optimizer, logger, opt):
    #         for saving and loading see:
    #         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

    gpu_id = opt['gpu_ids'][0]

    model = model.cpu()

    optimizer.state = set_gpu_recursive(optimizer.state, -1)

    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, './{0}/model.pth'.format(opt['save_dir']))
    # torch.save(param.state_dict(), './{0}/param.pth'.format(opt['save_dir']))

    model.cuda(gpu_id)
    optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)

    pickle.dump(logger, open('./{0}/logger.pkl'.format(opt['save_dir']), 'wb'))
    # pickle.dump(logger_eval, open('./{0}/logger_eval.pkl'.format(opt['save_dir']), 'wb'))
