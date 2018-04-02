import os
from collections import Iterable
from functools import partial

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate as collate

from ..utils.data_loading_utils import loadPILImages, loadH5images

class DataframeDataset(Dataset):
    """general dataframe Dataset class"""

    def __init__(self, df,
                 tabularData={'outName1':'colName1', 'outName2':'colName2'},
                 imageData={'image1':{'cols':['channel1colName', 'channel2colName']}, 'image2':{'cols':['RGBimagecolName']}}):
        """
        Args:
            df (pandas.DataFrame): dataframe containing tabular data and/or the absolute paths to image locations
            tabularData: dict whos values are column names in df whose entries will be returned for each data point, the names when returned will be the keys of this dict.
                         Groups of colum names can be used for a value (eg generating an 'X' matrix) but should only be used for groups of numeric columns, not strings
                         Columns should contain either strings or numeric types convertable to torch rensors (int, np.float, etc)
            imageData: dict whose keys each specify the name of a returned image, and whose values specify how that image is loaded/constructed, via kewargs: 'cols', 'mode' 'transform', and 'aggregate'.
                       'cols' (list) specifies which columns in the df contain paths to images that will be transformed/aggregated to construct an output images
                       'mode' (string) specifies which mode PIL/Pillow load the image as, eg 'L' (greyscale), 'RGB', etc
                       'aggregate' (function) specifies how each individual tensor from the transformed images will be aggregated into one image tensor, e.g. torch.stack, torch.cat, etc
                       'transform' (function) specifies how aggregated image found in 'cols' will be transformed.
                           mostly useful for one or three channel images using transforms.ToPILImage(), your transforms, and then transforms.ToTensor() inside transforms.Compose([])
                           lambda transforms like transforms.Lambda(lambda x: mask*x) should work directly on tensors of any number of channels
        """
        self.opts = locals()
        self.opts.pop('self')
        self.opts.pop('df')
        self.df = df.reset_index(drop=True)

        for name,kwargs in self.opts['imageData'].items():
            defaultkwargs = {'loader':partial(loadPILImages, mode='L'), 'aggregate':torch.stack, 'transform':transforms.Compose([])}
            defaultkwargs.update(self.opts['imageData'][name]); self.opts['imageData'][name] = defaultkwargs

    def __len__(self):
        return len(self.df)

    def _get_item(self, idx):
        images = {name:kw['transform'](kw['aggregate'](kw['loader'](list(self.df.loc[idx,kw['cols']]))).squeeze()) for name,kw in self.opts['imageData'].items()}
        tabular = {name:self.df[cols].iloc[idx] for name,cols in self.opts['tabularData'].items()}
        tabular = {name:(torch.from_numpy(value.values) if isinstance(value, pd.Series) else value) for name,value in tabular.items()}
        return {**images, **tabular}

    def __getitem__(self, idx):
        return collate([self._get_item(i) for i in idx]) if isinstance(idx,Iterable) else self._get_item(idx)

class DatasetFeatures(DataframeDataset):
    def __init__(self, df, ycol='yCol', Xcols=['Xcol1', 'Xcol2', 'Xcol3']):
        kwargs = locals()
        tabularData={'X':kwargs['target'],
                     'y':kwargs['target']},
        DataframeDataset.__init__(self, df, tabularData=tabularData, imageData={})

class DatasetH5ToTarget(DataframeDataset):
    def __init__(self, df, target='targetCol', image='h5FilePathCol', channels=(3,2,4), imageTransform=transforms.Compose([])):
        kwargs = locals()
        tabularData={'target':kwargs['target']},
        imageData={'image':{'cols':[kwargs['image']], 'loader':partial(loadH5images, channels=kwargs['channels']), 'transform':kwargs['imageTransform']}}
        DataframeDataset.__init__(self, df, tabularData=tabularData, imageData=imageData)

class DatasetSingleRGBImageToTarget(DataframeDataset):
    def __init__(self, df, target='targetCol', image='imageFilePathCol', imageTransform=transforms.Compose([])):
        kwargs = locals()
        tabularData={'target':kwargs['target']}
        imageData={'image':{'cols':[kwargs['image']], 'loader':partial(loadPILImages, mode='RGB'), 'transform':kwargs['imageTransform']}}
        DataframeDataset.__init__(self, df, tabularData=tabularData, imageData=imageData)

class DatasetSingleGreyScaleImagetoTarget(DataframeDataset):
    def __init__(self, df, target='targetCol', image='imageFilePathCol', imageTransform=transforms.Compose([])):
        kwargs = locals()
        tabularData={'target':kwargs['target']}
        imageData={'image':{'cols':[kwargs['image']], 'loader':partial(loadPILImages, mode='L'), 'transform':kwargs['imageTransform']}}
        DataframeDataset.__init__(self, df, tabularData=tabularData, imageData=imageData)

class DatasetMultipleGreyScaleImagestoTarget(DataframeDataset):
    def __init__(self, df, target='targetCol', images=['imageFilePathCol1','imageFilePathCol2'], imageTransform=transforms.Compose([])):
        kwargs = locals()
        tabularData={'target':kwargs['target']}
        imageData={'image':{'cols':kwargs['images'], 'loader':partial(loadPILImages, mode='L'), 'transform':kwargs['imageTransform']}}
        DataframeDataset.__init__(self, df, tabularData=tabularData, imageData=imageData)

class DatasetHPA(DataframeDataset):
    def __init__(self, df, seqCol='antigenSequence', metadatacols=['EnsemblID','proteinID','antibodyName'], inputImageCols =['microtubuleChannel', 'nuclearChannel'], targetImageCol='antibodyChannel',
                 inputImageTransform=transforms.Compose([]), targetImageTransform=transforms.Compose([])):
        kwargs = locals()
        tabularData={'sequence':kwargs['seqCol'], **dict(zip(metadatacols,metadatacols))}
        imageData={'inputImage':{'cols':kwargs['inputImageCols'], 'loader':partial(loadPILImages, mode='L'), 'transform':kwargs['inputImageTransform']},
                   'targetImage':{'cols':[kwargs['targetImageCol']], 'loader':partial(loadPILImages, mode='L'), 'transform':kwargs['targetImageTransform']}}
        DataframeDataset.__init__(self, df, tabularData=tabularData, imageData=imageData)
