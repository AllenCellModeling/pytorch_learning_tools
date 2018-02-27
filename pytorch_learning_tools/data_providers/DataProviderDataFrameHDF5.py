import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from random import sample
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

from ..utils.hashsplit import hashsplit
from .DataProviderABC import DataProviderABC

# these are a few of utility functions that might be better stored elsewhere
def eight_bit_to_float(im, dtype=np.uint8):
    imax = np.iinfo(dtype).max  # imax = 255 for uint8
    im = im / imax
    return im

def load_h5(h5_path, image_channels):
    f = h5py.File(h5_path, 'r')
    image = f['image'].value[image_channels, ::]
    image = eight_bit_to_float(image)
    return torch.from_numpy(image)


# this is an augmentation to PyTorch's Dataset class that our Dataprovider below will use
class dataframeDataset(Dataset):
    """dataframe Dataset."""

    def __init__(self,
                 df,
                 image_root_dir='/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/',
                 image_path_col='save_h5_reg_path',
                 image_channels=(3, 4, 2),
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path'):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the image relative locations and target data
            image_root_dir (string): full path to the directory containing all the images
            image_path_col (string): column name in dataframe containing the paths to the h5 files to be used as input data.
            image_channels (tuple of integers): which channels in the input image do you want to keep as data
	    target_col (string): column name in the dataframe containing the data to be used as prediction targets
                                 column contents must be a type that can be converted to a pytorch tensor, eg np.float32 
            unique_id_col (string): which column in the dataframe file to use as a unique identifier for each data point
        """
        opts = locals()
        opts.pop('self')
        opts.pop('df')
        self.opts = opts

        # check that all files we need are present in the df
        good_rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc='scanning files'):
            image_path = os.path.join(image_root_dir, df.ix[idx, image_path_col])
            if os.path.isfile(image_path):
                good_rows += [idx]
        df = df.iloc[good_rows]
        df = df.reset_index(drop=True)

        # save df
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if not isinstance(idx,list):
            idx = [idx]

        # get the path to the image
        image_paths = [os.path.join(self.opts['image_root_dir'], self.df.ix[i, self.opts['image_path_col']]) for i in idx]

        # load the image
        data = [load_h5(ipath, self.opts['image_channels']) for ipath in image_paths]
        data = torch.stack(data)
 
        # load the target
        target = self.df.ix[idx, self.opts['target_col']].values
        target = torch.from_numpy(np.array([target])).transpose_(0,1)
        
        # load the unique id
        unique_id = list(self.df.ix[idx, self.opts['unique_id_col']].values)
        
        if len(image_paths) == 1:
            data = torch.squeeze(data, dim=0)
            target = torch.squeeze(target, dim=0)
            unique_id = unique_id[0]

        # collate the sample and return
        sample = (data,target,unique_id)
        return sample


# This is the dataframe-image dataprovider
class dataframeDataProvider(DataProviderABC):
    """dataframe dataprovider"""
    def __init__(self,
                 df,
                 image_root_dir='/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/',
                 image_path_col='save_h5_reg_path',
                 image_channels=(3, 4, 2),
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path',
                 split_fracs={'train': 0.8, 'test': 0.2},
                 batch_size=32,
                 shuffle=True,
                 split_seed=1,
                 num_workers=4,
                 pin_memory=True):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the relative image locations and target data
            image_root_dir = (string): full path to the directory containing all the HDF5 images
            image_path_col (string): column name in dataframe containing the paths to the files to be used as input data
            image_channels (tuple of integers): which channels in the input image do you want to keep as data
            target_col (string): column name in the dataframe containing the data to be used as prediction targets
                                 column contents must be a type that can be converted to a pytorch tensor, eg np.float32
            unique_id_col (string): which column in the dataframe to use as a unique identifier for each data point
            split_fracs (dict): names of splits desired, and fracion of data in each split
            batch_size (int): minibatch size for iterating through dataset
            shuffle (bool): shuffle the data every epoch, default True
            split_seed (int): random seed/salt for splitting has function
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

        # load up all the data, before splitting it -- the data gets checked in this call
        self._datasets = {split:dataframeDataset(df_s,
                                                 image_root_dir=image_root_dir,
                                                 image_path_col=image_path_col,
                                                 image_channels=image_channels,
                                                 target_col=target_col,
                                                 unique_id_col=unique_id_col) for split,df_s in dfs.items()}

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
        
        # find split each uid is in, get corresponding dfs, get inds in those dfs, and finaaly get data points
        splits = [self._ids2splits[unique_id] for unique_id in unique_ids]
        dfs = [self.dfs[split] for split in splits]
        df_inds = [df.index[df[self.opts['unique_id_col']] == unique_id].tolist()[0] for unique_id,df in zip(unique_ids,dfs)]
        data_points = [self._datasets[split][df_ind] for df_ind,split in zip(df_inds,splits)]
        
        # if more than one data point, group by type rather than by data point
        if len(data_points) == 1:
            return data_points[0]
        else:
            data_points_grouped = list(zip(*data_points))
            
            # x and y each get stacked as tensors
            for i in (0,1):
                data_points_grouped[i] = torch.stack(data_points_grouped[i])

            return data_points_grouped

