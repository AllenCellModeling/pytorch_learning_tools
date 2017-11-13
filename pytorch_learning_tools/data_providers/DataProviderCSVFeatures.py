import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, scale
from sklearn.feature_selection import VarianceThreshold

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torchvision import transforms, utils

from ..utils.hashsplit import hashsplit
from .DataProviderABC import DataProviderABC


# This is the csv-features dataprovider, which uses a TensorDataset -- data point unique ids are the data itself
class DataProvider(DataProviderABC):
    """csv dataprovider"""
    def __init__(self,
                 root_dir,
                 batch_size=32,
                 csv_name='feats_out_mitotic_annotations.csv',
                 feat_col_pattern='feat_',
                 feat_remove_zero_var=True,
                 feat_z_score=True,
                 target_col='MitosisLabel',
                 target_type_coerce=int,
                 target_transform='one_hot',
                 dataset_dtype=torch.FloatTensor,
                 fill_NA=0,
                 split_fracs={'train': 0.8, 'test': 0.2},
                 split_seed=1,
                 num_workers=4,
                 pin_memory=True,
                 verbose=True,
                 drop_last=True):
        """
        Args:
            root_dir (string): full path to the directory containing csv_name.
            batch_size (int): minibatch size for iterating through dataset
            csv_name (string): csv file with annotations, just the base name, not the full path.
            feat_col_pattern (string): pattern to match for cols to be used as features
            feat_z_score (bool): mean center and unit variance on feature cols
            feat_remove_zero_var (bool): if True, remove features with zero variance
            target_col (string): column name in the csv file containing the data to be used as prediction targets (no paths).
            target_type_coerce (numpy dtype): enforce target data type if not None, eg int string, float, etc
            target_transform (string): one of [None, 'label_encode', 'one_hot']
            dtype (torch tensor datatype): one of torch. FloatTensor, DoubleTensor, HalfTensor, ByteTensor, CharTensor, ShortTensor, IntTensor, LongTensor
            fill_NA (anything): if not None, what to replace NAs with in the data
            split_fracs (dict): names of splits desired, and fracion of data in each split
            split_seed (int): random seed/salt for splitting has function
            num_workers (int): number of cpu cores to use loading data
            pin_memory (bool): should be True unless you're getting gpu memory errors
            verbose (bool): run verbosely or quietly
            drop_last (bool): drop the last minibatch if smaller than the rest
        """

        # save the input options a la greg's style
        opts = locals()
        opts.pop('self')
        self.opts = opts

        # load up all the data
        if verbose:
            print("loading {}".format(os.path.join(root_dir, csv_name)))
        df_original = pd.read_csv(os.path.join(root_dir, csv_name))
        if fill_NA is not None:
            df_modified = df_original.fillna(fill_NA)
        
        # parse data into feature matrix (X), target vector (y)
        if verbose:
            print("splitting csv dataframe into X and y")
        y = df_modified[target_col].values
        if target_type_coerce is not None:
            if verbose:
                print("coercing y to {}".format(target_type_coerce))
            y = y.astype(target_type_coerce)
            
        feat_cols = df_modified.columns[df_modified.columns.str.contains(pat = feat_col_pattern)]
        X = df_modified[feat_cols].values
        if feat_z_score:
            if verbose:
                print("centering and scaling X to zero mean and unit variance")
            X = scale(X)
            
        # compute class weights
        c, w = np.unique(y, return_counts=True)
        w = 1/w
        w = w/np.sum(w)
        class_weights = torch.from_numpy(w).type(dataset_dtype)
            
        # save out modified dataframe
        df_modified[target_col] = y
        df_modified[feat_cols] = X
        
        # remove features with zero variance
        if feat_remove_zero_var:
            selector = VarianceThreshold()
            X_nzv = selector.fit_transform(X)
            if verbose:
                print("removed {0} features with zero variance".format(X.shape[1] - X_nzv.shape[1]))
            X = X_nzv
        
        # encode target as ints or one-hots and save a mapping to original names
        if target_transform == 'label_encode':
            if verbose:
                print("transforming y using label encoder")
            le = LabelEncoder()
            le.fit(y)
            y_le = le.transform(y)
            label_map = {c:le.transform([c])[0] for c in le.classes_}
            y = y_le
        elif target_transform == 'one_hot':
            if verbose:
                print("transforming y using one hot encoding")
            lb = LabelBinarizer()
            y_lb = lb.fit_transform(y)
            label_map = {y_el:lb.transform(np.array([y_el]))[0] for y_el in np.unique(y)}
            y = y_lb
        else:
            pass

        # convert X and y to torch tensors and save as dataset
        if verbose:
            print("creating TensorDataset from X and y")
        X = torch.from_numpy(X).type(dataset_dtype)
        y = torch.from_numpy(y).type(dataset_dtype)
        dataset = TensorDataset(X,y)
        
        # save out dimensions of data
        data_dims = {}
        data_dims['X'] = X.shape
        data_dims['y'] = y.shape
        
        # split the data into the different sets: test, train, valid, whatever
        if verbose:
            print("splitting dataset into {}".format(split_fracs))
        split_inds = hashsplit(df_original.astype(str).sum(axis=1).values,
                               splits=split_fracs,
                               salt=split_seed)
        
        # create random samplers to access only random samples from the appropriate indices
        if verbose:
            print("creating samplers")
        samplers = {}
        for split,inds_in_split in split_inds.items():
            samplers[split] = SubsetRandomSampler(inds_in_split)
        
        # create data loaders to efficiently iterate though the random samples
        if verbose:
            print("creating data loaders")
        dataloaders = {}
        for split, sampler in samplers.items():
            dataloaders[split] = DataLoader(dataset,
                                            batch_size=batch_size,
                                            sampler=sampler,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory,
                                            drop_last=drop_last)
        
        # save local variables into object
        self.df_original = df_original
        self.df_modified = df_modified
        
        self._dataset = dataset
        self.class_weights = class_weights
        self.data_dims = data_dims
        self._split_inds = split_inds
        self._samplers = samplers
        self.dataloaders = dataloaders
        if target_transform:
            self.label_map = label_map

    @property
    def splits(self):
        return self._split_inds
    
    @property
    def classes(self):
        return np.unique(self.df_modified[self.opts['target_col']]).tolist()

    # WARNING: get_data_points is an inefficient way to interact with the data,
    # mosty useful for inspecting good/bad predictions post-hoc.
    # To efficiently iterate through the data, use somthing like:
    #
    # for epoch in range(N_epochs):
    #     for phase, dataloader in dp.dataloaders.items():
    #         for i_mb,sample_mb in enumerate(dataloader):
    #             data_mb, targets_mb = sample_mb
    
    def get_data_points(self, df_indices):
        data_points = [self._dataset[i] for i in df_indices]
        data_points_X = torch.stack([d[0] for d in data_points])
        data_points_y = torch.stack([d[1] for d in data_points])
        return data_points_X, data_points_y
