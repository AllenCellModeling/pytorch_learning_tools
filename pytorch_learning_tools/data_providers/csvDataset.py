import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def eight_bit_to_float(im, dtype=np.uint8):
    imax = np.iinfo(dtype).max  # imax = 255 for uint8
    im = im / imax
    return (im)


def load_h5(h5_path, channel_inds):
    f = h5py.File(h5_path, 'r')
    image = f['image'].value[channel_inds, ::]
    image = eight_bit_to_float(image)
    return image


class csvDataset(Dataset):
    """csv Dataset."""

    def __init__(self,
                 csv_file_name,
                 root_dir,
                 data_path_col,
                 target_col,
                 unique_id_col,
                 data_type='hdf5',
                 channel_inds=(3, 4, 2),
                 transform=None):
        """
        Args:
            csv_file_name (string): csv file with annotations, just the base name, not the full path.
            root_dir = (string): full path to the directory containing csv_file_name.
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

        df = pd.read_csv(os.path.join(root_dir, csv_file_name))

        # check that all files we need are present in the df
        good_rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc='scanning files'):
            data_path = os.path.join(root_dir, df.ix[idx, data_path_col])
            if os.path.isfile(data_path):
                good_rows += [idx]
        print('dropping {0} rows out of {1} from {2}'.format(len(df) - len(good_rows), len(df), csv_file_name))
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
