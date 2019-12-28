import matplotlib.pyplot as plt
import numpy as np


def plot_error(logger, opt, close_fig=False, history=10000):
    plt.figure(figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(121)

    for i in range(2, len(logger.fields) - 1):
        field = logger.fields[i]
        plt.plot(logger.log['iter'], logger.log[field], label=field)

    plt.legend()
    plt.title('History')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    # plt.savefig('{0}/history.png'.format(opt['save_dir']), bbox_inches='tight')
    # plt.close()

    ### Short History
    history = int(len(logger.log['epoch']) / 2) - 1

    if history > history:
        history = history

    ydat = [logger.log['train_loss']]
    ymin = np.min(ydat)
    ymax = np.max(ydat)
    plt.ylim([ymin, ymax])

    x = logger.log['iter'][-history:]
    y = logger.log['train_loss'][-history:]

    epochs = np.floor(np.array(logger.log['epoch'][-history:]))
    losses = np.array(logger.log['train_loss'][-history:])
    iters = np.array(logger.log['iter'][-history:])
    uepochs = np.unique(epochs)

    epoch_losses = np.zeros(len(uepochs))
    epoch_iters = np.zeros(len(uepochs))
    i = 0
    for uepoch in uepochs:
        inds = np.equal(epochs, uepoch)
        loss = np.mean(losses[inds])
        epoch_losses[i] = loss
        epoch_iters[i] = np.mean(iters[inds])
        i += 1

    mval = np.mean(losses)

    # plt.figure()

    plt.subplot(122)
    plt.plot(x, y, label='loss')
    plt.plot(epoch_iters, epoch_losses, color='darkorange', label='epoch avg')

    plt.plot([np.min(iters), np.max(iters)], [mval, mval], color='darkorange', linestyle=':', label='window avg')

    plt.legend()
    plt.title('Short history')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('{0}/history.png'.format(opt['save_dir']), bbox_inches='tight')
    if close_fig:
        plt.close()
