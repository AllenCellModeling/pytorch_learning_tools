import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample

import torch
from torch.utils.data import Dataset, DataLoader

from ..utils.hashsplit import hashsplit
from ..utils.data_utils import collate_data
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
        self.opts = locals()
        self.opts.pop('self')
        self.opts.pop('df')

        self.df = df.reset_index(drop=True)

        feat_cols = df.columns[df.columns.str.contains(pat=feat_col_pattern)]
        self._X = torch.from_numpy(df[feat_cols].values)
        self._y = torch.from_numpy(df[target_col].values)
        self._u = df[self.opts['unique_id_col']].values

    def __len__(self):
        return len(self.df)

    def _get_single_item(self, idx):
        data = self._X[idx]
        target = self._y[idx]
        unique_id = self._u[idx]
        return (data,target,unique_id)

    def __getitem__(self, idx):
        return collate_data([self._get_single_item(i) for i in idx]) if isinstance(idx,list) else self._get_single_item(idx)

# This is the dataframe-features dataprovider
class dataframeDataProvider(DataProviderABC):
    """feature dataframe dataprovider"""
    def __init__(self, df,
                 split_fracs={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 dataset_kwargs={split:{'feat_col_pattern':'feat_',
                                        'target_col':'targetNumeric',
                                        'unique_id_col':'save_h5_reg_path'} for split in ('train', 'test')},
                 dataloader_kwargs={split:{'batch_size':128,
                                           'shuffle':True,
                                           'drop_last':True,
                                           'num_workers':4,
                                           'pin_memory':True} for split in ('train', 'test')}):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the relative image locations and target data
            split_fracs (dict): names of splits desired, and fracion of data in each split
            split_seed (int): random seed/salt for splitting has function
            dataset_kwargs: dict of keyword args for dataset (see dataframeDataSet for docs)
            dataloader_kwargs: dict of keyword args for Pytorch DataLoader class
        """
        self.opts = locals()
        self.opts.pop('self')
        self.opts.pop('df')

        df = df.reset_index(drop=True)

        # make sure unique id and target columns are consistent across datasets and save
        uniq_id_cols = set([kwargs['unique_id_col'] for split,kwargs in dataset_kwargs.items()])
        assert len(uniq_id_cols) == 1, "unique id cols must be identical across datasets"
        self._unique_id_col = uniq_id_cols.pop()
        target_cols = set([kwargs['target_col'] for split,kwargs in dataset_kwargs.items()])
        assert len(target_cols) == 1, "target cols must be identical across datasets"
        self._target_col = target_cols.pop()

        # split the data into the different sets: test, train, valid, whatever
        # use dict of list of explicit inds if provided, otherwise split randomly
        if all([isinstance(v,list) for k,v in split_fracs.items()]):
            self._split_inds = split_fracs
        else:
            self._split_inds = hashsplit(df[self._unique_id_col], splits=split_fracs, salt=split_seed)

        # split dataframe by split inds
        dfs = {split:df.iloc[inds].reset_index(drop=True) for split,inds in self._split_inds.items()}

        # load up all the datasets
        self._datasets = {split:dataframeDataset(df_s, **dataset_kwargs[split]) for split,df_s in dfs.items()}

        # save dfs as an accessable dict, and toal (concat) df in a hidden one
        self.dfs = {split:dset.df for split,dset in self._datasets.items()}
        self._df = pd.concat([dset.df for dset in self._datasets.values()], ignore_index=True)

        # create data loaders to efficiently iterate though the random samples
        self.dataloaders = {split:DataLoader(dset, **dataloader_kwargs[split]) for split, dset in self._datasets.items()}

        # save a map from unique ids to splits + inds
        splits2indsuids = {split:tuple(zip(df_s[self._unique_id_col],df_s.index)) for split,df_s in self.dfs.items()}
        self._uids2splitsinds = {uid:(split,ind) for split in splits2indsuids for (uid,ind) in splits2indsuids[split]}

    @property
    def splits(self):
        return self._split_inds

    @property
    def classes(self):
        unique_classes = np.union1d(*[df_s[self._target_col].unique() for split,df_s in self.dfs.items()])
        unique_classes = np.append(unique_classes[~np.isnan(unique_classes)], np.nan) if np.any(np.isnan(unique_classes)) else unique_classes
        return unique_classes.tolist()

    def _get_single_item(self, unique_id):
        split, ind = self._uids2splitsinds[unique_id]
        return self._datasets[split]._get_single_item(ind)

    def __getitem__(self, uids):
        return collate_data([self._get_single_item(u) for u in uids]) if isinstance(uids,list) else self._get_single_item(uids)
