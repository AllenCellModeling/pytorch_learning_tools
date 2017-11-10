import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
# from random import sample

from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torchvision import transforms, utils

from ..utils.hashsplit import hashsplit
from .DataProviderABC import DataProviderABC


# this is an augmentation to PyTorch's TensorDataset class that our Dataprovider below will use
class TensorDatasetWithIDs(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        id_array (numpy string array): contains sample unique ids.
    """

    def __init__(self, data_tensor, target_tensor, id_array):
        assert data_tensor.size(0) == target_tensor.size(0)
        assert data_tensor.size(0) == len(id_array)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.id_array = id_array

    def __getitem__(self, index):
        sample = {'data': self.data_tensor[index],
                  'target': self.target_tensor[index],
                  'unique_id': self.id_array[index]}
        return sample

    def __len__(self):
        return self.target_tensor.size(0)

    
# This is the csv-image dataprovider
class DataProvider(DataProviderABC):
    """csv dataprovider"""
    def __init__(self,
                 root_dir,
                 batch_size=32,
                 csv_name='feats_out_mitotic_annotations.csv',
                 feat_col_pattern='feat_',
                 target_col='MitosisLabel',
                 convert_target_to_string=True,
                 labelencode=True,
                 fill_NA=0,
                 unique_id_col='save_h5_reg_path',
                 split_fracs={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 num_workers=4,
                 pin_memory=True):
        """
        Args:
            root_dir (string): full path to the directory containing csv_name.
            batch_size (int): minibatch size for iterating through dataset
            csv_name (string): csv file with annotations, just the base name, not the full path.
            feat_col_pattern (string): pattern to match for cols to be used as features
            target_col (string): column name in the csv file containing the data to be used as prediction targets (no paths).
            fill_NA (anything): if not None, what to replace NAs with in the data
            convert_target_to_string (bool): force conversion of target column contents to string type
            unique_id_col (string): which column in the csv file to use as a unique identifier for each data point
            split_fracs (dict): names of splits desired, and fracion of data in each split
            split_seed (int): random seed/salt for splitting has function
            num_workers (int): number of cpu cores to use loading data
            pin_memory (Bool): should be True unless you're getting gpu memory errors
        """

        # save the input options a la greg's style
        opts = locals()
        opts.pop('self')
        self.opts = opts

        # load up all the data
        df = pd.read_csv(os.path.join(root_dir, csv_name))
        if fill_NA is not None:
            df = df.fillna(fill_NA)
        
        # parse data into feature matrix (X), target vector(y), and unique ids (t)
        y = df[target_col]
        X = df[df.columns[df.columns.str.contains(pat = feat_col_pattern)]].values
        t = df[unique_id_cols].astype(str).values
        
        if labelencode:
            le = LabelEncoder()
            le.fit(y)
            y = le.transform(y)
        else:
            y = y.values
        
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        
        self._dataset = TensorDatasetWithIDs(X,y,t)
        self.df = self._dataset.df
        
        # split the data into the different sets: test, train, valid, whatever
        split_inds = hashsplit(self.df[self.opts['unique_id_col']],
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
            dataloaders[split] = DataLoader(self._dataset,
                                            batch_size=batch_size,
                                            sampler=sampler,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory)
        self.dataloaders = dataloaders
        
    
    @property
    def splits(self):
        return self._split_inds
    
    @property
    def classes(self):
        return np.unique(self.df[self.opts['target_col']]).tolist()

    # WARNING: get_data_points is an inefficient way to interact with the data,
    # mosty useful for inspecting good/bad predictions post-hoc.
    # To efficiently iterate thoguh the data, use somthing like:
    #
    # for epoch in range(N_epochs):
    #     for phase, dataloader in dp.dataloaders.items():
    #         for i_mb,sample_mb in enumerate(dataloader):
    #             data_mb, targets_mb, unique_ids_mb = sample_mb['data'], sample_mb['target'], sample_mb['unique_id']
    def get_data_points(self, unique_ids):
        df_inds = self.df.index[self.df[self.opts['unique_id_col']].isin(unique_ids)].tolist()
        data_points = [self._dataset[i] for i in df_inds]
        return data_points
