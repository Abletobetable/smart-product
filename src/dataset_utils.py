"""
couple dataset functions for preprocessing
"""


import os
import json
import pandas as pd
from typing import Literal

def expand_text_fields(df: pd.DataFrame()) -> pd.DataFrame():
    """
    expand every category in text fields in single column

    Parameters
    ----------
        df (pd.DataFrame()): 
            dataframe to process

    Return
    ------
        new_df (pd.DataFrame()): 
            processed dataframe
    """
    new_df = df

    title = []
    description = []
    attributes = []
    custom_characteristics = []
    defined_characteristics = []
    filters = []

    for d_text in df['text_fields']:

        d = json.loads(d_text)

        title.append(d['title'])
        description.append(d['description'])
        attributes.append(d['attributes'])
        custom_characteristics.append(d['custom_characteristics'])
        defined_characteristics.append(d['defined_characteristics'])
        filters.append(d['filters'])

    new_df['title'] = title
    new_df['description'] = description
    new_df['attributes'] = attributes
    new_df['custom_characteristics'] = custom_characteristics
    new_df['defined_characteristics'] = defined_characteristics
    new_df['filters'] = filters

    return new_df

def add_images_path(path_df: pd.DataFrame(), expanded_df: pd.DataFrame(), split: Literal['train', 'test']) -> pd.DataFrame():
    """
    add to dataframe path to images
    inside: get path with os.listdir() and then pd.merge()

    Parameters
    ----------
        path_df (pd.DataFrame()): 
            dataframe with raw path

        expanded_df (pd.DataFrame()): 
            dataframe with information about products

        split (Literal['train', 'test']): 
            split for choosing right folder with images

    Return
    ------
        new_df (pd.DataFrame()): 
            full dataframe
    """

    new_df = path_df

    df_images_id = []
    df_images_path = []
    df_images_id = []
    df_images_path = []

    for path in new_df['raw_path']:

        image_id = os.path.splitext(path)[0]
        df_images_id.append(int(image_id))

        df_images_path.append(f'images/{split}/{path}')

    new_df['product_id'] = df_images_id
    new_df['path'] = df_images_path
    new_df.drop(['raw_path'], axis=1, inplace=True)

    new_df = pd.merge(expanded_df, path_df, on="product_id", how="right")

    return new_df