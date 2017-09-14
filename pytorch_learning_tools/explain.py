"""
Examples using my explanation code on Greg's 3-d structure prediction model.
"""

import model_utils as mu
import importlib
import torch
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import sys
from imgToProjection import imgtoprojection
# TODO These paths will have to change depending on how the user has their
# docker env set up.

# Pull in the code to do integrated gradients
sys.path.append("/root/proj/integrated_gradients")
from integrated_gradients import integrated_gradients


def get_dp():
    """
    Retrieves the data provider Greg created for the 3d structure data.
    Requires a little finessing to get the paths right.
    """
    path_dp = ("/root/modeling/gregj/projects/pytorch_learning_tools/"
               "data/dp_h5.pyt")
    dp = torch.load(path_dp)
    dp.image_parent = ("/root/modeling/gregj/results/"
                       "ipp_dataset_cellnuc_seg_curated_8_24_17/")
    return dp


def get_model():
    """
    Retrieves the model that Greg trained on the 3d structure data.
    """
    path_opt = ("/root/modeling/gregj/projects/pytorch_learning_tools/"
                "output/classifier3D/opt.pkl")
    opt = pkl.load(open(path_opt, "rb"))
    opt.save_dir = "." + opt.save_dir
    opt.data_save_path = "." + opt.data_save_path
    opt.data_dir = ("/root/modeling/gregj/results/"
                    "ipp_dataset_cellnuc_seg_curated_8_24_17/")
    model_provider = importlib.import_module("models.conv3D")
    models, optimizers, criterions, logger, opt = mu.load_model(model_provider,
                                                                opt)
    model = models["model"]
    model.train(False)              # Put in evaluation mode.
    return model


def get_label_examples(dp, label, n):
    """
    Given a data provider `dp`, return `n` examples of images with the
    requested `label`.
    """
    labels = dp.labels
    test_ix = dp.data["test"]["inds"]
    test_labels = labels[test_ix]
    candidates = np.arange(len(test_ix))[test_labels == label]
    ixs = np.random.choice(candidates, n, replace=False)
    imgs = dp.get_images(ixs, "test")
    labels = dp.get_classes(ixs, "test")
    assert np.all(labels.numpy() == label)
    return imgs


def tensor2img(img):
    # Pulled in from Greg's pytorch_integrated_cell.model_utils.
    im_out = list()
    for i in range(0, img.shape[0]):
        im_out.append(img[i])

    img = np.concatenate(im_out, 2)

    if len(img.shape) == 3:
        img = np.expand_dims(img, 3)

    colormap = 'hsv'

    colors = plt.get_cmap(colormap)(np.linspace(0, 1, img.shape[0]+1))

    # img = np.swapaxes(img, 2,3)
    img = imgtoprojection(np.swapaxes(img, 1, 3), colors = colors, global_adjust=True)
    img = np.swapaxes(img, 0, 2)

    return img


def explain(model, img, label, n_steps,
            gpu_id=None, baseline=None, title=None):
    """
    Explain `model`'s prediction on `img`, which is of class `label`. Use
    `n_steps` iterations of integrated gradients.
    """
    # Get the integrated gradients.
    igs, diff = integrated_gradients(model, img, label,
                                     n_steps, gpu_id, baseline)
    img_fmt = tensor2img(img.unsqueeze(0).numpy())
    respons = igs.cpu().unsqueeze(0).numpy()
    respons_pos = np.copy(respons)
    respons_neg = -np.copy(respons)
    respons_pos[respons_pos < 0] = 0
    respons_neg[respons_neg < 0] = 0
    respons_pos_neg = np.concatenate([respons_pos, respons_neg], axis=0)
    fig, axs = plt.subplots(1, 2, figsize=[8, 4])
    axs[0].imshow(img_fmt)
    axs[0].set_title("Original image")
    axs[1].set_title("Positive attributions | negative attributions")
    axs[1].imshow(tensor2img(respons_pos_neg))
    if title is not None:
        fig.suptitle(title)
    return igs, diff


def format_image(img):
    """Imshow for Tensor."""
    img = img.unsqueeze(0).numpy().transpose((1, 2, 0))
    maxs = np.empty(3)
    mins = np.empty(3)
    for i in range(3):
        this_slice = img[:, :, i]
        maxs[i] = np.percentile(this_slice, 99.99)
        this_slice[this_slice > maxs[i]] = maxs[i]
        mins[i] = this_slice.min()
    ranges = maxs - mins
    img = (img - mins) / ranges
    return img
