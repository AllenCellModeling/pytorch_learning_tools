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

# this is an augmentation to PyTorch's Dataset class that our Dataprovider below will use
class dataframeDataset(Dataset):
    """dataframe Dataset."""

    def __init__(self,
                 df,
                 feat_col_pattern='feat_',
                 feat_dtype_coerce=torch.FloatTensor,
                 target_col='structureProteinName',
                 target_dtype_coerce=torch.LongTensor,
                 unique_id_col='save_h5_reg_path'):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the image relative locations and target data
            feat_col_pattern (string): pattern to match for cols to be used as features
            target_col (string): column name in the dataframe containing the data to be used as prediction targets (no paths).
            unique_id_col (string): which column in the dataframe file to use as a unique identifier for each data point. If None, df index is used.
        """
        opts = locals()
        opts.pop('self')
        opts.pop('df')
        self._opts = opts

        master_inds = np.array(df.index.tolist())
        df = df.reset_index(drop=True)

        # construct X
        feat_cols = df.columns[df.columns.str.contains(pat=feat_col_pattern)]
        X = df[feat_cols].values
        X = torch.from_numpy(X)
        X = X.type(feat_dtype_coerce)
        self._X = X

        #construct y
        y = df[target_col].values
        y = torch.from_numpy(y)
        y = y.type(target_dtype_coerce)
        self._y = y

        # construct unique ids
        if unique_id_col is not None:
            self._u = df[unique_id_col].values
        else:
            self._u = master_inds

        # save df
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not isinstance(idx,list):
            idx = [idx]

        data = self._X[idx]
        target = self._y[idx]
        unique_id = list(self._u[idx])

        sample = (data,target,unique_id)

        return sample


# This is the dataframe-features dataprovider
class dataframeDataProvider(DataProviderABC):
    """dataframe dataprovider"""
    def __init__(self,
                 df,
                 feat_col_pattern='feat_',
                 feat_dtype_coerce=torch.FloatTensor,
                 target_col='structureProteinName',
                 target_dtype_coerce=torch.LongTensor,
                 unique_id_col='save_h5_reg_path',
                 batch_size=32,
                 shuffle=True,
                 split_fracs={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 num_workers=4,
                 pin_memory=True):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the image relative locations and target data
            feat_col_pattern (string): pattern to match for cols to be used as features
            feat_dtype_coerce (type): pytorch tensor type to return input data as
            target_col (string): column name in the dataframe containing the data to be used as prediction targets (no paths).
            target_dtype_coerce (type): pytorch tensor type to return target data as
            unique_id_col (string): which column in the dataframe file to use as a unique identifier for each data point. If None, df index is used.
            batch_size (int): minibatch size for iterating through dataset
            shuffle (bool): shuffle the data every epoch, default True
            split_fracs (dict): names of splits desired, and fracion of data in each split
            split_seed (int): random seed/salt for splitting has function
            num_workers (int): number of cpu cores to use loading data
            pin_memory (Bool): should be True unless you're getting gpu memory errors
        """

        # save the input options a la greg's style
        opts = locals()
        opts.pop('self')
        opts.pop('df')
        self.opts = opts

        df = df.reset_index(drop=True)

        # split the data into the different sets: test, train, valid, whatever
        if unique_id_col is None:
            unique_id_col = 'index_as_uid'
            df[unique_id_col] = df.index
            self.opts['unique_id_col'] = unique_id_col

        self._split_inds = hashsplit(df[unique_id_col],
                                     splits=split_fracs,
                                     salt=split_seed)

        # split dataframe by split inds
        dfs = {split:df.iloc[inds] for split,inds in self._split_inds.items()}

        # load up all the data by split
        self._datasets = {split:dataframeDataset(df_s,
                                                 feat_col_pattern=feat_col_pattern,
                                                 feat_dtype_coerce=feat_dtype_coerce,
                                                 target_col=target_col,
                                                 target_dtype_coerce=target_dtype_coerce,
                                                 unique_id_col=unique_id_col) for split,df_s in dfs.items()}

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

        # if more than one data point, group by type rather than by data point
        if len(data_points) == 1:
            return data_points[0]
        else:
            data_points_grouped = list(zip(*data_points))

            # x and y each get stacked as tensors
            for i in (0,1):
                data_points_grouped[i] = torch.stack(data_points_grouped[i])

            return data_points_grouped

