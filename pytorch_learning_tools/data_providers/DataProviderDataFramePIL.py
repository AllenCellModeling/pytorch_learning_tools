import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..utils.hashsplit import hashsplit
from ..utils.data_utils import collate_data
from ..utils.data_loading_utils import pick_image_loader, load_rgb_img, load_greyscale_tiff
from .DataProviderABC import DataProviderABC


# this is an augmentation to PyTorch's Dataset class that our Dataprovider below will use
class dataframeDataset(Dataset):
    """PIL image dataframe Dataset."""

    def __init__(self, df,
                 image_root_dir='/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/',
                 image_path_col='save_flat_reg_path',
                 image_type='png',
                 image_channels=(0,1,2),
                 image_transform=transforms.Compose([transforms.ToTensor()]),
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path'):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the image relative locations and target data
            image_root_dir (string): full path to the directory containing all the images
            image_path_col (string): column name in dataframe containing the paths to the images to be used as input data.
            image_type (string): jpeg, png, greyscale tiff, etc (only basic image type support is implemented)
            image_channels (tuple of integers): which channels in the input image do you want to keep as data
            image_transform (callable): torchvision transform to be applied on a sample, default is transforms.Compose([transforms.ToTensor()])
                                        to keep only the channels you want, add e.g. lambda x: x[(0,2),:,:] to the Compose
            target_col (string): column name in the dataframe containing the data to be used as prediction targets
                                 column contents must be a type that can be converted to a pytorch tensor, eg np.float32 
            unique_id_col (string): which column in the dataframe file to use as a unique identifier for each data point
        """
        self.opts = locals()
        self.opts.pop('self')
        self.opts.pop('df')

        self.df = df.reset_index(drop=True)
        self._imgloader = pick_image_loader(self.opts['image_type'])
        self._trans = self.opts['image_transform']

    def __len__(self):
        return len(self.df)

    def _get_single_item(self, idx):
        image_path = os.path.join(self.opts['image_root_dir'], self.df[self.opts['image_path_col']][idx])
        image = self._trans(self._imgloader(image_path, self.opts['image_channels']))
        target = torch.from_numpy(np.array([self.df[self.opts['target_col']][idx]]))
        unique_id = self.df[self.opts['unique_id_col']][idx]
        return (image, target, unique_id)

    def __getitem__(self, idx):
        idx = [idx] if not isinstance(idx,list) else idx
        return collate_data([self._get_single_item(i) for i in idx])

# This is the dataframe-image dataprovider
class dataframeDataProvider(DataProviderABC):
    """PIL image dataframe dataprovider"""
    
    dataset_kwargs={split:{'image_root_dir':'/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/',
                           'image_path_col':'save_flat_reg_path',
                           'image_type':'png',
                           'image_channels':(0,2),
                           'target_col':'targetNumeric',
                           'unique_id_col':'save_h5_reg_path'} for split in ('train', 'test')}
    dataloader_kwargs={split:{'batch_size':128,
                              'shuffle':True,
                              'drop_last':True,
                              'num_workers':4,
                              'pin_memory':True} for split in ('train', 'test')}

    def __init__(self, df,
                 split_fracs={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 dataset_kwargs=dataset_kwargs,
                 dataloader_kwargs=dataloader_kwargs):
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
        
        # make sure unique id column is consistent across datasets and save
        uniq_id_cols = set([kwargs['unique_id_col'] for split,kwargs in dataset_kwargs.items()])
        assert len(uniq_id_cols) == 1, "unique id cols must be identical across datasets"
        self._unique_id_col = uniq_id_cols.pop()
 
        # split the data into the different sets: test, train, valid, whatever
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
        unique_classes = np.union1d(*[df_s[self.opts['target_col']].unique() for split,df_s in self.dfs.items()])
        unique_classes = np.append(unique_classes[~np.isnan(unique_classes)], np.nan) if np.any(np.isnan(unique_classes)) else unique_classes
        return unique_classes.tolist()

    def _get_single_item(self, unique_id):
        split, ind = self._uids2splitsinds[unique_id]
        return self._datasets[split]._get_single_item(ind)    
        
    def __getitem__(self, unique_ids):
        unique_ids = [unique_ids] if not isinstance(unique_ids,list) else unique_ids
        return collate_data([self._get_single_item(u) for u in unique_ids])

