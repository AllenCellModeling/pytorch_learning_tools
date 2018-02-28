import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample

import torch
from torch.utils.data import Dataset, DataLoader

from ..utils.hashsplit import hashsplit
from .DataProviderABC import DataProviderABC

# this is an augmentation to PyTorch's Dataset class that our Dataprovider below will use
class dataframeDataset(Dataset):
    """feature dataframe Dataset."""

    def __init__(self,
                 df,
                 feat_col_pattern='feat_',
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path'):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the image relative locations and target data
            feat_col_pattern (string): pattern to match for cols to be used as features
            target_col (string): column name in the dataframe containing the data to be used as prediction targets
            unique_id_col (string): which column in the dataframe file to use as a unique identifier for each data point. If None, df index is used
        """
        opts = locals()
        opts.pop('self')
        opts.pop('df')
        self.opts = opts

        master_inds = np.array(df.index.tolist())
        df = df.reset_index(drop=True)
        self.df = df

        # construct X
        feat_cols = df.columns[df.columns.str.contains(pat=feat_col_pattern)]
        X = df[feat_cols].values
        X = torch.from_numpy(X)
        self._X = X

        #construct y
        y = df[target_col].values
        y = torch.from_numpy(y)
        self._y = y

        # construct unique ids
        self._u = df[self.opts['unique_id_col']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not isinstance(idx,list):
            idx = [idx]

        data = torch.squeeze(self._X[idx], dim=0)
        target = self._y[idx]
        unique_id = list(self._u[idx]) if len(idx)>1 else self._u[idx][0]

        return (data,target,unique_id)


# This is the dataframe-features dataprovider
class dataframeDataProvider(DataProviderABC):
    """feature dataframe dataprovider"""
    def __init__(self,
                 df,
                 feat_col_pattern='feat_',
                 target_col='structureProteinName',
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
            target_col (string): column name in the dataframe containing the data to be used as prediction targets (no paths).
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

        # if no unique id column to dataframe if none supplied
        if unique_id_col is None:
            unique_id_col = 'index_as_uid'
            df[unique_id_col] = df.index
            self.opts['unique_id_col'] = unique_id_col
            print('No unique IDs supplied, adding column "index_as_uid" and using input dataframe indices as unique ID via that column')

        # split the data into the different sets: test, train, valid, whatever
        self._split_inds = hashsplit(df[unique_id_col],
                                     splits=split_fracs,
                                     salt=split_seed)

        # split dataframe by split inds
        dfs = {split:df.iloc[inds] for split,inds in self._split_inds.items()}

        # load up all the data by split
        self._datasets = {split:dataframeDataset(df_s,
                                                 feat_col_pattern=feat_col_pattern,
                                                 target_col=target_col,
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

        # save a map from unique ids to splits + inds
        splits2indsuids = {split:tuple(zip(df_s[self.opts['unique_id_col']],df_s.index)) for split,df_s in self.dfs.items()}
        self._uids2splitsinds = {uid:(split,ind) for split in splits2indsuids for (uid,ind) in splits2indsuids[split]}

    @property
    def splits(self):
        return self._split_inds

    @property
    def classes(self):
        unique_classes = np.union1d(*[df_s[self.opts['target_col']].unique() for split,df_s in self.dfs.items()])
        if np.any(np.isnan(unique_classes)):
            unique_classes = np.append(unique_classes[~np.isnan(unique_classes)], np.nan)
        return unique_classes.tolist()

    def __getitem__(self, unique_ids):

        # everything is a list
        if not isinstance(unique_ids,list):
            unique_ids = [unique_ids]

        # get (split,ind) list from uid list, then get datapoints from datasets
        splitsinds = [self._uids2splitsinds[uid] for uid in unique_ids]
        datapoints = [self._datasets[split][ind] for split,ind in splitsinds]

        # if more than one data point, group by type rather than by data point
        if len(datapoints) == 1:
            return datapoints[0]
        else:
            datapoints_grouped = list(zip(*datapoints))
            for i in (0,1):
                datapoints_grouped[i] = torch.stack(datapoints_grouped[i])
            datapoints_grouped[2] = list(datapoints_grouped[2])
            return datapoints_grouped

