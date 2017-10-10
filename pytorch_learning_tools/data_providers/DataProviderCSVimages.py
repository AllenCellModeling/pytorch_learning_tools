import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torchvision import transforms, utils


from ..utils.hashsplit import hashsplit
from .DataProviderABC import DataProviderABC


# these are a couple of utility functions that might be better stored elsewhere
def eight_bit_to_float(im, dtype=np.uint8):
    imax = np.iinfo(dtype).max  # imax = 255 for uint8
    im = im / imax
    return im
def load_h5(h5_path, channel_inds):
    f = h5py.File(h5_path, 'r')
    image = f['image'].value[channel_inds, ::]
    image = eight_bit_to_float(image)
    return image


# modification of PyTorch's SubsetRandomSampler to remove randomness
class SubsetSampler(Sampler):
    """Samples elements sequentially, always in the same order
       from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# this is an augmentation to PyTorch's Dataset class that our Dataprovider below will use
class csvDataset(Dataset):
    """csv Dataset."""

    def __init__(self,
                 root_dir,
                 csv_name='data_jobs_out.csv',
                 data_path_col='save_h5_reg_path',
                 data_type='hdf5',
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path',
                 channel_inds=(3, 4, 2),
                 transform=None):
        """
        Args:
            root_dir = (string): full path to the directory containing csv_name.
            csv_name (string): csv file with annotations, just the base name, not the full path.
            data_path_col (string): column name in csv file containing the paths to the files to be used as input data.
            data_type (string): hdf5, png, etc (only hdf5 support is implemented)
            target_col (string): column name in the csv file containing the data to be used as prediction targets (no paths).
            unique_id_col (string): which column in the csv file to use as a unique identifier for each data point
            channel_inds (tuple of integers): 0: cell segmentation
                                              1: nuclear segmentation
                                              2: DNA channel
                                              3: membrane channel
                                              4: structure channel
                                              5: bright-field
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        opts = locals()
        opts.pop('self')
        self._opts = opts
        
        df = pd.read_csv(os.path.join(root_dir, csv_name))

        # check that all files we need are present in the df
        good_rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc='scanning files'):
            data_path = os.path.join(root_dir, df.ix[idx, data_path_col])
            if os.path.isfile(data_path):
                good_rows += [idx]
        print('dropping {0} rows out of {1} from {2}'.format(len(df) - len(good_rows), len(df), csv_name))
        df = df.iloc[good_rows]
        df = df.reset_index()

        # save df
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_path = os.path.join(self._opts['root_dir'], self.df.ix[idx, self._opts['data_path_col']])
        if self._opts['data_type'] == 'hdf5':
            data = load_h5(data_path, self._opts['channel_inds'])
        else:
            raise ValueError('data_type {} is not supported, only hdf5'.format(self.data_type))

        target = self.df.ix[idx, self._opts['target_col']]
        unique_id = self.df.ix[idx, self._opts['unique_id_col']]

        sample = {'data': data, 'target': target, 'unique_id': unique_id}

        if self._opts['transform']:
            sample = self._opts['transform'](sample)

        return sample

    
# This is the csv-image dataprovider
class DataProvider(DataProviderABC):
    """csv dataprovider"""
    def __init__(self,
                 root_dir,
                 batch_size=32,
                 csv_name='data_jobs_out.csv',
                 data_path_col='save_h5_reg_path',
                 data_type='hdf5',
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path',
                 channel_inds=(3, 4, 2),
                 split_fracs={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 inspection_split='test',
                 insection_data_size=16,
                 transform=None,
                 num_workers=4,
                 pin_memory=True):
        """
        Args:
            root_dir (string): full path to the directory containing csv_name.
            batch_size (int): minibatch size for iterating through dataset
            csv_name (string): csv file with annotations, just the base name, not the full path.
            data_path_col (string): column name in csv file containing the paths to the files to be used as input data.
            data_type (string): hdf5, png, etc (only hdf5 support is implemented)
            target_col (string): column name in the csv file containing the data to be used as prediction targets (no paths).
            unique_id_col (string): which column in the csv file to use as a unique identifier for each data point
            channel_inds (tuple of integers): 0: cell segmentation
                                              1: nuclear segmentation
                                              2: DNA channel
                                              3: membrane channel
                                              4: structure channel
                                              5: bright-field
            split_fracs (dict): names of splits desired, and fracion of data in each split
            split_seed (int): random seed/salt for splitting has function
            inspection_split (string): set from which to draw inspectino data
            insection_data_size (int): size of inspection data set
            transform (callable, optional): Optional transform to be applied on a sample.
            num_workers (int): number of cpu cores to use loading data
            pin_memory (Bool): should be True unless you're getting gpu memory errors
        """

        # save the input options a la greg's style
        opts = locals()
        opts.pop('self')
        self.opts = opts

        # load up all the data, before splitting it -- the data gets checked in this call
        self.dataset = csvDataset(root_dir,
                               csv_name=csv_name,
                               data_path_col=data_path_col,
                               target_col=target_col,
                               unique_id_col=unique_id_col,
                               data_type=data_type,
                               channel_inds=channel_inds,
                               transform=transform)
        
        
        # split the data into the different sets: test, train, valid, whatever
        split_inds = hashsplit(self.dataset.df[self.opts['unique_id_col']],
                               splits=split_fracs,
                               salt=split_seed)
        self._split_inds=split_inds
        
        # create random samplers to access only random samples from the appropriate indices
        samplers = {}
        for split,inds_in_split in self._split_inds.items():
            samplers[split] = SubsetRandomSampler(inds_in_split)
        self._samplers = samplers
        
        # create data loaders to efficiently iterate though the random samples
        dataloaders = {}
        for split, sampler in self._samplers.items():
            dataloaders[split] = DataLoader(self.dataset,
                                            batch_size=batch_size,
                                            sampler=sampler,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory)
        self.dataloaders = dataloaders
        
        # create determinisitic loaders to always sample ceratin inds --
        # useful for inspecting training progress on the same images every time
        # hard coded to sample the first few images
        deterministic_samplers = {}
        for split,inds_in_split in self._split_inds.items():
            deterministic_samplers[split] = SubsetSampler(sample(inds_in_split,batch_size))
        self._deterministic_samplers = deterministic_samplers
        
        deterministic_dataloaders = {}
        for split, deterministic_sampler in self._deterministic_samplers.items():
            deterministic_dataloaders[split] = DataLoader(self.dataset,
                                                          batch_size=batch_size,
                                                          sampler=deterministic_sampler,
                                                          num_workers=4, # TODO: make an option?
                                                          pin_memory=True) # TODO: make an option?
        self._deterministic_dataloaders = deterministic_dataloaders
    
    @property
    def splits(self):
        return self._split_inds
    
    @property
    def unique_targets(self):
        return np.unique(self.dataset.df[self.opts['target_col']]).tolist()

    # WARNING: get_data_points is an inefficient way to interact with the data,
    # mosty useful for inspecting good/bad predictions post-hoc.
    # To efficiently iterate thoguh the data, use somthing like:
    #
    # for epoch in range(N_epochs):
    #     for phase, dataloader in dp.dataloaders.items():
    #         for i_mb,sample_mb in enumerate(dataloader):
    #             data_mb, targets_mb, unique_ids_mb = sample_mb['data'], sample_mb['target'], sample_mb['unique_id']
    def get_data_points(self, unique_ids):
        df_inds = self.dataset.df.index[self.dataset.df[self.opts['unique_id_col']].isin(unique_ids)].tolist()
        data_points = [self.dataset[i] for i in df_inds]
        return data_points
    
    def get_inspection_data(self):
        inpection_data = next(iter(self._deterministic_dataloaders[self.opts['inspection_split']]))
        return inpection_data
