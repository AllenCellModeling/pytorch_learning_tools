import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from ..utils.hashsplit import hashsplit
from .DataProviderABC import DataProviderABC

# these are a few of utility functions that might be better stored elsewhere
def eight_bit_to_float(im, dtype=np.uint8):
    imax = np.iinfo(dtype).max  # imax = 255 for uint8
    im = im / imax
    return im

def load_h5(h5_path, channel_inds):
    f = h5py.File(h5_path, 'r')
    image = f['image'].value[channel_inds, ::]
    image = eight_bit_to_float(image)
    return image

def load_rgb_img(img_path, channel_inds, data_as_image=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            img_arr = np.array(img.convert('RGB'))
            if data_as_image:
                zero_channels = [i for i in range(3) if i not in channel_inds]
                img_arr[:,:,zero_channels] = 0
                return Image.fromarray(img_arr)
            else:
                return eight_bit_to_float(img_arr[:,:,channel_inds])

def load_greyscale_tiff(img_path,data_as_image=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            if data_as_image:
                return img
            else:
                return eight_bit_to_float(np.array(img.convert('L')))


# this is an augmentation to PyTorch's Dataset class that our Dataprovider below will use
class dataframeDataset(Dataset):
    """dataframe Dataset."""

    def __init__(self,
                 df,
                 data_root_dir='/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/',
                 data_path_col='save_h5_reg_path',
                 data_type='hdf5',
                 data_as_image=False,
                 data_type_coerce=torch.FloatTensor,
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path',
                 channel_inds=(3, 4, 2),
                 transform=None):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the image relative locations and target data
            data_root_dir (string): full path to the directory containing all the images.
            data_path_col (string): column name in dataframe containing the paths to the files to be used as input data.
            data_type (string): hdf5, png, etc (only hdf5 and basic image type support is implemented)
            data_type_coerce (torch data type): torch.FloatTensor, torch.DoubleTensor, etc
            data_as_image (bool): if True, return a pil image, if False, return a np array of floats
	    target_col (string): column name in the dataframe containing the data to be used as prediction targets (no paths).
            unique_id_col (string): which column in the dataframe file to use as a unique identifier for each data point
            channel_inds (tuple of integers): which channels in the input image do you want to keep as data
            transform (callable, optional): Optional torchvision transform to be applied on a sample, only works if data_as_image
        """
        opts = locals()
        opts.pop('self')
        opts.pop('df')
        self._opts = opts

        # check that all files we need are present in the df
        good_rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc='scanning files'):
            data_path = os.path.join(data_root_dir, df.ix[idx, data_path_col])
            if os.path.isfile(data_path):
                good_rows += [idx]
        df = df.iloc[good_rows]
        df = df.reset_index(drop=True)

        # save df
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_path = os.path.join(self._opts['data_root_dir'], self.df.ix[idx, self._opts['data_path_col']])
        if self._opts['data_type'] == 'hdf5':
            data = load_h5(data_path, self._opts['channel_inds'])
        elif self._opts['data_type'] in ['jpg','JPG','jpeg','JPEG','png','PNG','ppm','PPM','bmp','BMP']:
            data = load_rgb_img(data_path, self._opts['channel_inds'], data_as_image=self._opts['data_as_image'])
        elif self._opts['data_type'] in ['tif', 'TIF', 'tiff', 'TIFF']:
            data = load_greyscale_tiff(data_path, data_as_image=self._opts['data_as_image'])
        else:
            raise ValueError('data_type {} is not supported, only hdf5 and basic images'.format(self.data_type))

        if self._opts['data_as_image']:
            if self._opts['transform']:
                trans = self._opts['transform']
                data = trans(data)
        else:
            data = torch.from_numpy(data)
        data = data.type(self._opts['data_type_coerce'])
 
        target = self.df.ix[idx, self._opts['target_col']]
        target = torch.from_numpy(np.array([target]))
        unique_id = self.df.ix[idx, self._opts['unique_id_col']]

        sample = (data,target,unique_id)

        return sample


# This is the dataframe-image dataprovider
class dataframeDataProvider(DataProviderABC):
    """dataframe dataprovider"""
    def __init__(self,
                 df,
                 data_root_dir='/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/',
                 batch_size=32,
                 shuffle=True,
                 data_path_col='save_h5_reg_path',
                 data_type='hdf5',
                 data_as_image=False,
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path',
                 channel_inds=(3, 4, 2),
                 split_fracs={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 transform=None,
                 num_workers=4,
                 pin_memory=True):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the relative image locations and target data
            data_root_dir = (string): full path to the directory containing all the images.
            batch_size (int): minibatch size for iterating through dataset
            shuffle (bool): shuffle the data every epoch, default True
            data_path_col (string): column name in dataframe containing the paths to the files to be used as input data.
            data_type (string): hdf5, png, jpg, tiff, etc (only hdf5 and common image format + tiff support is implemented)
            data_as_image (bool): if True, return a PIL image, else return a numpy array of floats
            target_col (string): column name in the dataframe containing the data to be used as prediction targets.
            unique_id_col (string): which column in the dataframe to use as a unique identifier for each data point
            channel_inds (tuple of integers): which channels in the input image do you want to keep as data
            split_fracs (dict): names of splits desired, and fracion of data in each split
            split_seed (int): random seed/salt for splitting has function
            transform (callable, optional): Optional transform to be applied on a samplei, only works if data_as_image.
            num_workers (int): number of cpu cores to use loading data
            pin_memory (Bool): should be True unless you're getting gpu memory errors
        """

        # save the input options a la greg's style
        opts = locals()
        opts.pop('self')
        opts.pop('df')
        self.opts = opts
        
        # split the data into the different sets: test, train, valid, whatever
        if unique_id_col is None:
            unique_id_col = 'index_as_uid'
            df[unique_id_col] = df.index
            self.opts['unique_id_col'] = unique_id_col

        self._split_inds = hashsplit(df[unique_id_col],
                                     splits=split_fracs,
                                     salt=split_seed)

        # split dataframe by split inds
        dfs = {split:df.iloc[inds].reset_index(drop=True) for split,inds in self._split_inds.items()}

        # if transform isn't a dict, make it one:
        if not isinstance(transform, dict):
            transform = {split:transform for split in self._split_inds.keys()}

        # load up all the data, before splitting it -- the data gets checked in this call
        self._datasets = {split:dataframeDataset(df_s,
                                                 data_root_dir=data_root_dir,
                                                 data_path_col=data_path_col,
                                                 data_type=data_type,
                                                 data_as_image=data_as_image,
                                                 target_col=target_col,
                                                 unique_id_col=unique_id_col,
                                                 channel_inds=channel_inds,
                                                 transform=transform[split]) for split,df_s in dfs.items()}

        # report how many data points were dropped due to missing data
        drops = {split: len(self._split_inds[split]) - len(dset) for split,dset in self._datasets.items()}
        for split in drops.keys():
            print("dropped {} data points in {} split".format(drops[split],split))

        # save filtered dfs as an accessable dict
        self.dfs = {split:dset.df for split,dset in self._datasets.items()}
        self._df = pd.concat([dset.df for dset in self._datasets.values()], ignore_index=True)

        # create data loaders to efficiently iterate though the random samples
        self.dataloaders = {split:DataLoader(dset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory) for split, dset in self._datasets.items()}

        # save a map from unique ids to splits
        splits2ids = {split:df_s[self.opts['unique_id_col']] for split,df_s in self.dfs.items()}
        self._ids2splits = {v:k for k in splits2ids for v in splits2ids[k]}

    @property
    def splits(self):
        return self._split_inds

    @property
    def classes(self):
        #return np.unique( self.df[self.opts['target_col']] ).tolist()
        unique_classes = np.union1d(*[df_s[self.opts['target_col']].unique() for split,df_s in self.dfs.items()])
        if np.any(np.isnan(unique_classes)):
            unique_classes = np.append(unique_classes[~np.isnan(unique_classes)], np.nan)
        return unique_classes.tolist()


    # WARNING: this is an inefficient way to interact with the data,
    # mosty useful for inspecting good/bad predictions post-hoc.
    # To efficiently iterate thoguh the data, use somthing like:
    #
    # for epoch in range(N_epochs):
    #     for phase, dataloader in dp.dataloaders.items():
    #         for i_mb,sample_mb in enumerate(dataloader):
    #             data_mb, targets_mb, unique_ids_mb = sample_mb['data'], sample_mb['target'], sample_mb['unique_id']
    def __getitem__(self, unique_ids):
        if not isinstance(unique_ids,list):
            unique_ids = [unique_ids]
        splits = [self._ids2splits[unique_id] for unique_id in unique_ids]
        dfs = [self.dfs[split] for split in splits]
        df_inds = [df.index[df[self.opts['unique_id_col']] == unique_id].tolist()[0] for unique_id,df in zip(unique_ids,dfs)]
        data_points = [self._datasets[split][df_ind] for df_ind,split in zip(df_inds,splits)]
        print(data_points)
        
        # group output by type rather than my data point
        data_points_grouped = [[item[i] for item in data_points] for i in range(3)]
        print(data_points_grouped)

        #for d in data_points_grouped:
        #    for x,y,u in d:
        #        print(type(d_i))
        
        # x gets stacked as a tensor from [x1,x2,x3]
        data_points_grouped[0] = torch.stack(data_points_grouped[0])

        # y gets stacked as a tensor from [y1,y2,y3]
        data_points_grouped[1] = torch.stack(data_points_grouped[1])

        # u gets converted from a list to a numpy array
        data_points_grouped[2] = np.array(data_points_grouped[2])

        return data_points_grouped
