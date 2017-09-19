import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb


class trainer(object):

    def __init__(self, dp, opt):

        gpu_id = opt['gpu_ids'][0]

        self.x = Variable(dp.get_images(range(0, opt['batch_size']), 'train').cuda(gpu_id))

        if opt['n_classes'] > 0:
            self.classes = Variable(torch.LongTensor(opt['batch_size'])).cuda(gpu_id)
        else:
            self.classes = None

    def get_sample(self, dp, ndat, batch_size, train_or_test='train'):
        rand_inds_encD = np.random.permutation(ndat)
        inds = rand_inds_encD[0:batch_size]

        print('batch_size = ', batch_size)
        print('data size = ', self.x.data.size())
        print('get_images size = ', dp.get_images(inds, train_or_test).size())
        self.x.data.copy_(dp.get_images(inds, train_or_test))
        x = self.x

        self.classes.data.copy_(dp.get_classes(inds, train_or_test))
        target = self.classes

        return x, target

    def iteration(self, model, optimizer, crit, dp, opt):
        gpu_id = opt['gpu_ids'][0]

        x, target = self.get_sample(dp, opt['ndat'], opt['batch_size'], 'train')

        optimizer.zero_grad()

        ## train the classifier
        target_pred = model(x)

        pred_loss = crit(target_pred, target)
        pred_loss.backward()
        pred_loss = pred_loss.data[0]

        optimizer.step()

        _, indices = torch.max(target_pred, 1)
        acc = (indices == target).double().mean().data[0]

        errors = (
            pred_loss,
            acc,)

        return errors, target, target_pred

    def evaluate(self, model, crit, dp, opt, train_or_test='test'):
        model.train(False)

        x, target = self.get_sample(dp, dp.get_n_dat(train_or_test), opt['batch_size'], train_or_test)
        x.volatile = True

        target_pred = model(x)

        pred_loss = crit(target_pred, target)
        pred_loss = pred_loss.data[0]

        _, indices = torch.max(target_pred, 1)

        acc = (indices == target).double().mean().data[0]

        x.volatile = False
        model.train(True)

        errors = (
            pred_loss,
            acc,)
        return errors
