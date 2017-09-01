import torch
import importlib
import torch.optim as optim
import SimpleLogger
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.misc
import pickle
import importlib

import matplotlib.pyplot as plt
from imgToProjection import imgtoprojection

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
    model = model_provider.model(opt.n_classes, opt.nch, opt.gpu_ids, opt)
    model.apply(weights_init)

    gpu_id = opt.gpu_ids[0]
    
    model.cuda(gpu_id)
    
    if opt.optimizer == 'RMSprop':
        param = optim.RMSprop(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'adam':
        param = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    columns = ('epoch', 'iter', 'loss', 'time')
    print_str = '[%d][%d] pred_loss: %.6f time: %.2f'
    
    logger = SimpleLogger.SimpleLogger(columns,  print_str)

    this_epoch = 1
    iteration = 0
    if os.path.exists('./{0}/model.pth'.format(opt.save_dir)):
        print('Loading from ' + opt.save_dir)
        
        model.load_state_dict(torch.load('./{0}/model.pth'.format(opt.save_dir)))
        model.cuda(gpu_id)
        
        param.load_state_dict(torch.load('./{0}/param.pth'.format(opt.save_dir)))
        param.state = set_gpu_recursive(param.state, gpu_id)
        
        logger = pickle.load(open( '{0}/logger.pkl'.format(opt.save_dir), "rb" ))

        this_epoch = max(logger.log['epoch']) + 1
        iteration = max(logger.log['iter'])

    models = {'model': model}
    
    optimizers = dict()
    optimizers['param'] = param
    
    criterions = dict()
    if opt.n_classes > 0:
        criterions['crit'] = nn.NLLLoss()
    else:
        criterions['crit'] = nn.BCELoss()
 
    return models, optimizers, criterions, logger, opt

def maybe_save(epoch, epoch_next, models, optimizers, logger, dp, opt):
    saved = False
    if epoch != epoch_next and ((epoch_next % opt.save_progress_iter) == 0 or (epoch_next % opt.save_state_iter) == 0):

        if (epoch_next % opt.save_progress_iter) == 0:
            print('saving progress')
            save_progress(logger, opt)

        if (epoch_next % opt.save_state_iter) == 0:
            print('saving state')
            save_state(**models, **optimizers, logger=logger, opt=opt)

        saved = True
        
    return saved
            

def save_progress(logger, opt):
    
    ### History
    plt.figure()

    for i in range(2, len(logger.fields)-1):
        field = logger.fields[i]
        plt.plot(logger.log['iter'], logger.log[field], label=field)

    plt.legend()
    plt.title('History')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('{0}/history.png'.format(opt.save_dir), bbox_inches='tight')
    plt.close()

    ### Short History
    history = int(len(logger.log['epoch'])/2) - 1
    
    if history > 10000:
        history = 10000
    
    ydat = [logger.log['loss']]
    ymin = np.min(ydat)
    ymax = np.max(ydat)
    plt.ylim([ymin, ymax])

    x = logger.log['iter'][-history:]
    y = logger.log['loss'][-history:]

    epochs = np.floor(np.array(logger.log['epoch'][-history:-1]))
    losses = np.array(logger.log['loss'][-history:-1])
    iters = np.array(logger.log['iter'][-history:-1])
    uepochs = np.unique(epochs)

    epoch_losses = np.zeros(len(uepochs))
    epoch_iters = np.zeros(len(uepochs))
    i = 0
    for uepoch in uepochs:
        inds = np.equal(epochs, uepoch)
        loss = np.mean(losses[inds])
        epoch_losses[i] = loss
        epoch_iters[i] = np.mean(iters[inds])
        i+=1

    mval = np.mean(losses)

    plt.figure()
    plt.plot(x, y, label='loss')
    plt.plot(epoch_iters, epoch_losses, color='darkorange', label='epoch avg')
    
    plt.plot([np.min(iters), np.max(iters)], [mval, mval], color='darkorange', linestyle=':', label='window avg')

    plt.legend()
    plt.title('Short history')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('{0}/history_short.png'.format(opt.save_dir), bbox_inches='tight')
    plt.close()
    
def save_state(model, param, logger, opt):
#         for saving and loading see:
#         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718
  
    gpu_id = opt.gpu_ids[0]
    
    model = model.cpu()

    param.state = set_gpu_recursive(param.state, -1)

    torch.save(model.state_dict(), './{0}/model.pth'.format(opt.save_dir))
    torch.save(param.state_dict(), './{0}/param.pth'.format(opt.save_dir))

    model.cuda(gpu_id)
    param.state = set_gpu_recursive(param.state, gpu_id)

    pickle.dump(logger, open('./{0}/logger.pkl'.format(opt.save_dir), 'wb'))
   