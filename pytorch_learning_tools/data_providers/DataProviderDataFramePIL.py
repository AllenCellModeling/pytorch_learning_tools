import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..utils.hashsplit import hashsplit
from ..utils.data_loading_utils import load_rgb_img, load_greyscale_tiff
from .DataProviderABC import DataProviderABC


# this is an augmentation to PyTorch's Dataset class that our Dataprovider below will use
class dataframeDataset(Dataset):
    """PIL image dataframe Dataset."""

    def __init__(self,
                 df,
                 image_root_dir='/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/',
                 image_path_col='save_flat_reg_path',
                 image_type='png',
                 image_channels=(0,1,2),
                 image_transform=transforms.Compose([transforms.ToTensor()]),
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path',
                 preload_data_in_memory=False):
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
            preload_data_in_memory (bool) load up all the data in main memory or not
        """
        opts = locals()
        opts.pop('self')
        opts.pop('df')
        self.opts = opts

        # save df
        df = df.reset_index(drop=True)
        self.df = df

        # pick an image loader
        if self.opts['image_type'] in ['jpg','JPG','jpeg','JPEG','png','PNG','ppm','PPM','bmp','BMP']:
            self._imgloader = load_rgb_img
        elif self.opts['image_type'] in ['tif', 'TIF', 'tiff', 'TIFF']:
            self._imgloader = load_greyscale_tiff
        else:
            raise ValueError('image_type {} is not supported, only basic images'.format(self.image_type))

        # preload data to main memory or not
        if self.opts['preload_data_in_memory']:
            image_paths = [os.path.join(self.opts['image_root_dir'], self.df.ix[i, self.opts['image_path_col']])
                               for i in tqdm(df.index, desc='Loading images into main memory', unit='images')]
            self._X = torch.stack([self._imgloader(ipath, self.opts['image_channels']) for ipath in image_paths])
            self._y = torch.from_numpy(df[target_col].values)
            self._u = df[self.opts['unique_id_col']].values

        # save the transform as a callable function
        self._trans = self.opts['image_transform']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # everything is a list
        if not isinstance(idx,list):
            idx = [idx]

        # if we preloaded, just grab the data
        if self.opts['preload_data_in_memory']:
            data = self._X[idx]
            data = torch.stack([self._trans(d) for d in data])
            target = self._y[idx]
            unique_id = self._u[idx]
        else:
            # get the path to the image
            image_paths = [os.path.join(self.opts['image_root_dir'], self.df.ix[i, self.opts['image_path_col']]) for i in idx]

            # load the image
            data = [self._imgloader(ipath, self.opts['image_channels']) for ipath in image_paths]

            # transform the image
            data = torch.stack([self._trans(d) for d in data])

            # load the target
            target = self.df.ix[idx, self.opts['target_col']].values
            target = torch.from_numpy(np.array([target])).transpose_(0,1)

            # load the unique id
            unique_id = list(self.df.ix[idx, self.opts['unique_id_col']].values)

        # if only one thing requested, don't wrap it in a list
        if len(idx) == 1:
            data = torch.squeeze(data, dim=0)
            target = torch.squeeze(target, dim=0)
            unique_id = unique_id[0]

        # collate the sample and return
        return (data,target,unique_id)


# This is the dataframe-image dataprovider
class dataframeDataProvider(DataProviderABC):
    """PIL image dataframe dataprovider"""
    def __init__(self,
                 df,
                 image_root_dir='/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/',
                 image_path_col='save_flat_reg_path',
                 image_type='png',
                 image_channels=(0, 1, 2),
                 image_transform=transforms.Compose([transforms.ToTensor()]),
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path',
                 split_fracs={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 dataloader_kwargs={'batch_size':32,
                                    'shuffle':True,
                                    'num_workers':4,
                                    'pin_memory':True}):
        """
        Args:
            df (pandas.DataFrame): dataframe containing the relative image locations and target data
            image_root_dir = (string): full path to the directory containing all the images
            image_path_col (string): column name in dataframe containing the paths to the files to be used as input data
            image_type (string): png, jpg, greyscale tiff, etc (only common image format support is implemented)
            image_channels (tuple of integers): which channels in the input image do you want to keep as data
            image_transform (callable): torchvision transform to be applied on a sample, default is transforms.Compose([transforms.ToTensor()])
                                        to keep only the channels you want, add e.g. lambda x: x[(0,2),:,:] to the Compose
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
        
        # add unique id column to dataframe if none supplied
        if unique_id_col is None:
            unique_id_col = 'index_as_uid'
            df[unique_id_col] = df.index
            self.opts['unique_id_col'] = unique_id_col

        # split the data into the different sets: test, train, valid, whatever
        self._split_inds = hashsplit(df[unique_id_col],
                                     splits=split_fracs,
                                     salt=split_seed)

        # split dataframe by split inds
        dfs = {split:df.iloc[inds].reset_index(drop=True) for split,inds in self._split_inds.items()}

        # if transform isn't a dict over splits, make it one:
        if not isinstance(image_transform, dict):
            image_transform = {split:image_transform for split in self._split_inds.keys()}

        # load up all the data, before splitting it -- the data gets checked in this call
        self._datasets = {split:dataframeDataset(df_s,
                                                 image_root_dir=image_root_dir,
                                                 image_path_col=image_path_col,
                                                 image_type=image_type,
                                                 image_channels=image_channels,
                                                 image_transform=image_transform[split],
                                                 target_col=target_col,
                                                 unique_id_col=unique_id_col) for split,df_s in dfs.items()}

        # save filtered dfs as an accessable dict
        self.dfs = {split:dset.df for split,dset in self._datasets.items()}
        self._df = pd.concat([dset.df for dset in self._datasets.values()], ignore_index=True)

        # create data loaders to efficiently iterate though the random samples
        self.dataloaders = {split:DataLoader(dset, **dataloader_kwargs) for split, dset in self._datasets.items()}

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
            
            # x and y each get stacked as tensors
            for i in (0,1):
                datapoints_grouped[i] = torch.stack(datapoints_grouped[i])

            return datapoints_grouped

