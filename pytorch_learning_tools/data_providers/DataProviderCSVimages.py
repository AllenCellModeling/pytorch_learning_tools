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


# modification of SubsetRandomSampler to remove randomness
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
    

# this is a pytorch Dataset augmentation that the Dataprovider below will use
class csvDataset(Dataset):
    """csv Dataset."""

    def __init__(self,
                 root_dir,
                 csv_name='data_jobs_out.csv',
                 data_path_col='save_h5_reg_path',
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path',
                 data_type='hdf5',
                 channel_inds=(3, 4, 2),
                 transform=None):
        """
        Args:
            csv_name (string): csv file with annotations, just the base name, not the full path.
            root_dir = (string): full path to the directory containing csv_name.
            data_path_col (string): column name in csv file containing the paths to the files to be used as input data.
            target_col (string): column name in the csv file containing the data to be used as prediction targets (no paths).
            unique_id_col (string):
            data_type (string):
            channel_inds (tuple of integers): 0: cell segmentation
                                              1: nuclear segmentation
                                              2: DNA channel
                                              3: membrane channel
                                              4: structure channel
                                              5: bright-field
            transform (callable, optional): Optional transform to be applied on a sample.
        """

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

        # save all the instantiation options and junk into the Dataset object
        self.df = df
        self.root_dir = root_dir
        self.data_path_col = data_path_col
        self.target_col = target_col
        self.unique_id_col = unique_id_col
        self.data_type = data_type
        self.channel_inds = channel_inds
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.df.ix[idx, self.data_path_col])
        if self.data_type == 'hdf5':
            data = load_h5(data_path, self.channel_inds)
        else:
            raise ValueError('data_type {} is not supported, only hdf5'.format(self.data_type))

        target = self.df.ix[idx, self.target_col]
        unique_id = self.df.ix[idx, self.unique_id_col]

        sample = {'data': data, 'target': target, 'unique_id': unique_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

# This is the csv-image dataprovider
class DataProvider(DataProviderABC):

    def __init__(self,
                 root_dir,
                 batch_size=32,
                 csv_name='data_jobs_out.csv',
                 data_path_col='save_h5_reg_path',
                 data_type='hdf5',
                 target_col='structureProteinName',
                 unique_id_col='save_h5_reg_path',
                 channel_inds=(3, 4, 2),
                 splits={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 transform=None):

        opts = locals()
        opts.pop('self')
        self.opts = opts

        # load up all the data, before splitting it
        self.data = csvDataset(root_dir,
                               csv_name=csv_name,
                               data_path_col=data_path_col,
                               target_col=target_col,
                               unique_id_col=unique_id_col,
                               data_type=data_type,
                               channel_inds=channel_inds,
                               transform=transform)
        
        
        # split the data into the different sets: test, train, valid, whatever
        split_inds = hashsplit(self.data.df[self.data.unique_id_col],
                               splits=splits,
                               salt=split_seed)
        self.split_inds=split_inds
        
        # create random samplers to access only random samples from the appropriate indices
        samplers = {}
        for split,inds_in_split in self.split_inds.items():
            samplers[split] = SubsetRandomSampler(inds_in_split)
        self.samplers = samplers
        
        # create data loaders to efficiently iterate though the random samples
        dataloaders = {}
        for split, sampler in self.samplers.items():
            dataloaders[split] = DataLoader(self.data,
                                            batch_size=batch_size,
                                            sampler=sampler,
                                            num_workers=4, # TODO: make an option?
                                            pin_memory=True) # TODO: make an option?
        self.dataloaders = dataloaders
        
        # create determinisitic loaders to always sample ceratin inds --
        # useful for inspecting training progress on the same images every time
        # hard coded to sample the first few images
        deterministic_samplers = {}
        for split,inds_in_split in self.split_inds.items():
            deterministic_samplers[split] = SubsetSampler(sample(inds_in_split,batch_size))
        self.deterministic_samplers = deterministic_samplers
        
        deterministic_dataloaders = {}
        for split, deterministic_sampler in self.deterministic_samplers.items():
            deterministic_dataloaders[split] = DataLoader(self.data,
                                                          batch_size=batch_size,
                                                          sampler=deterministic_sampler,
                                                          num_workers=4, # TODO: make an option?
                                                          pin_memory=True) # TODO: make an option?
        self.deterministic_dataloaders = deterministic_dataloaders
    
    # This could really be a static attribute?
    def data_len(self, split):
        return len(self.split_inds[split])        
    
    # Same here
    def target_cardinality(self):
        return len(np.unique(self.data.df[self.data.target_col]))        

    # get_data_points is an inefficient way to interact with the data,
    # mosty useful for inspecting good/bad predictions post-hoc.
    # To efficiently iterate thoguh th data, use somthing like:
    #
    # for epoch in range(N_epochs):
    #     for phase, dataloader in dp.dataloaders.items():
    #         for i_minibatch, samples in enumerate(dataloader):
    #             unique_ids_mb, data_mb, targets_mb = samples['unique_id'], sample['data'] ,samples['target']
    #
    def get_data_points(self, inds, split):
        data = [self.data[i] for i in inds]
        return data
