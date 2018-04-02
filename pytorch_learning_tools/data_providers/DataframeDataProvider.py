import os
from collections import Iterable
from functools import partial

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate as collate

from .DataProviderABC import DataProviderABC
from .DataframeDataset import DataframeDataset
from ..utils.hashsplit import hashsplit


class DataframeDataProvider(DataProviderABC):
    """PIL image dataframe dataprovider"""
    def __init__(self, df,
                 datasetClass=DataframeDataset,
                 split_fracs={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 uniqueID='uniqueID',
                 dataset_kwargs={split:{'tabularData':{'outName1':'colName1', 'outName2':'colName2', 'uniqueID':'uniqueIDcol'},
                                        'imageData':{'image1':{'cols':['channel1colName', 'channel2colName']}, 'image2':{'cols':['RGBimagecolName']}}}
                                 for split in ('train', 'test')},
                 dataloader_kwargs={split:{'batch_size':8, 'shuffle':True, 'drop_last':True, 'num_workers':4, 'pin_memory':True}
                                    for split in ('train', 'test')}):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the relative image locations and target data
            datasetClass (DataframeDataset*): specific class you want to wrap in a dataprovider
            split_fracs (dict): names of splits desired, and either a list of indices per split or a fracion of data in each split
            split_seed (int): random seed/salt for splitting has function, ignored if lists of inds are provided
            uniqueID (string): column name of column in df containing a unique id for each row
            dataset_kwargs: dict of keyword args for dataset; keys/values change depending on which type of DataframeDataset is desired
            dataloader_kwargs: dict of keyword args for Pytorch DataLoader class

        Notes: - The onus is on the user to make sure the data is actually present and in good shape
               - you MUST have a uniqueID column in the dataframe
        """
        self.opts = locals()
        self.opts.pop('self')
        self.opts.pop('df')

        df = df.reset_index(drop=True)
        self._unique_id_col = uniqueID

        # split the data into the different sets: test, train, valid, whatever
        # use dict of list of explicit inds if provided, otherwise split by hashing
        if all([isinstance(v,list) for k,v in split_fracs.items()]):
            self._split_inds = split_fracs
        else:
            self._split_inds = hashsplit(df[self._unique_id_col], splits=split_fracs, salt=split_seed)

        # split dataframe by split inds
        dfs = {split:df.iloc[inds].reset_index(drop=True) for split,inds in self._split_inds.items()}

        # load up all the datasets
        self._datasets = {split:datasetClass(df_s, **dataset_kwargs[split]) for split,df_s in dfs.items()}

        # save dfs as an accessable dict
        self.dfs = {split:dset.df for split,dset in self._datasets.items()}

        # create data loaders to efficiently iterate though the split datasets
        self.dataloaders = {split:DataLoader(dset, **dataloader_kwargs[split]) for split, dset in self._datasets.items()}

        # save a map from unique ids to splits + inds
        splits2indsuids = {split:tuple(zip(df_s[self._unique_id_col],df_s.index)) for split,df_s in self.dfs.items()}
        self._uids2splitsinds = {uid:(split,ind) for split in splits2indsuids for (uid,ind) in splits2indsuids[split]}

    @property
    def splits(self):
        return self._split_inds

    def _get_single_item(self, unique_id):
        split, ind = self._uids2splitsinds[unique_id]
        return self._datasets[split]._get_single_item(ind)

    def __getitem__(self, uids):
        return collate([self._get_single_item(u) for u in uids]) if (isinstance(uids, Iterable) and not isinstance(L, str)) else self._get_single_item(uids)
