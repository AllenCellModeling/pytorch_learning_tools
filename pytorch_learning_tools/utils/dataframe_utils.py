import os
import torch
import pandas as pd
from tqdm import tqdm

def filter_dataframe(input_df, image_root_dir, image_path_col):
    """filter dataframe to include only rows where files in im_col are present"""
    df = input_df.copy()
    good_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='scanning files'):
        image_path = os.path.join(image_root_dir, df.loc[idx, image_path_col])
        if os.path.isfile(image_path):
            good_rows += [idx]
    df = df.iloc[good_rows]
    df = df.reset_index(drop=True)
    return df
