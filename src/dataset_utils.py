"""
couple dataset functions for preprocessing
"""


import os
import cv2
import json
import pandas as pd
from typing import Literal

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

MAGIC_SEED = len('DS Internship 2023 | KazanExpress')

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

def add_images_path(path_df: pd.DataFrame(), 
                    expanded_df: pd.DataFrame(), 
                    split: Literal['train', 'test']) -> pd.DataFrame():
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

def create_labels_mapping(preprocessed_dataset: pd.DataFrame()) -> dict():
    """
    create label2id and id2label dicts 
    for mapping between categories and labels

    Parameters
    ----------
        preprocessed_dataset (pd.DataFrame()): 
            dataframe with all content inside
        
    Return
    ------
        label2id (dict()): 
            label <-> id mapping

        id2label (dict()): 
            id <-> label mapping
    """

    labels = sorted(list(set(preprocessed_dataset['category_id'])))
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    print('Number of labels:', len(labels))

    return label2id, id2label

class ProductsDataset(Dataset):
    def __init__(self, imgs: list, transform=None):
        """
        Parameters
        ----------
            imgs (list): 
                list of zip of pairs (path, category)
            transform:
                transforms for images (augmentation)
        """
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_filepath = self.imgs[idx][0]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        category = self.imgs[idx][1]

        if self.transform is not None:
            image = self.transform(image=image)['image']
        return {'pixel_values': image, 'label': category}

def create_image_datasets(preprocessed_dataset_train: pd.DataFrame(), 
                          preprocessed_dataset_pred: pd.DataFrame()):
    """
    create pytorch datasets: train, valid and predict

    Parameters
    ----------
        preprocessed_dataset_train (pd.DataFrame()): 
            train dataframe with all content inside
        
        preprocessed_dataset_predict (pd.DataFrame()): 
            predict dataframe with all content inside
        
    Return
    ------

        dataset (ProductsDataset): 
            unsplitted version of dataset

        train_dataset (ProductsDataset):
            train split

        valid_dataset (ProductsDataset):
            valid split   

        pred_dataset (ProductsDataset):
            part of dataset for prediction
    """

    train_transform = A.Compose([
            A.augmentations.geometric.resize.Resize(224, 224), 
            A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
    ])

    test_transform = A.Compose([
            A.augmentations.geometric.resize.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
    ])

    # To make it easier for the model to get the label name from the label id, 
    # create a dictionary that maps the label name to an integer and vice versa
    label2id, id2label = create_labels_mapping(preprocessed_dataset_train)

    # train / valid split
    train_df, valid_df = train_test_split(preprocessed_dataset_train, test_size=0.2, 
                                          random_state=MAGIC_SEED)

    print('len train split:', len(train_df))
    print('len valid split:', len(valid_df))

    # unsplitted dataset
    unsplitted_labels = [torch.tensor(int(label2id[l])) for l in preprocessed_dataset_train['category_id']]
    unsplitted_set = list(zip(preprocessed_dataset_train['path'], unsplitted_labels))
    unsplitted_dataset = ProductsDataset(unsplitted_set, test_transform)

    # splitted datasets
    train_labels = [torch.tensor(int(label2id[l])) for l in train_df['category_id']]
    valid_labels = [torch.tensor(int(label2id[l])) for l in valid_df['category_id']]
    pred_labels = [torch.tensor(0) for i in range(len(preprocessed_dataset_pred))]

    trainset = list(zip(train_df['path'], train_labels))
    validset = list(zip(valid_df['path'], valid_labels))
    predset = list(zip(preprocessed_dataset_pred['path'], pred_labels))

    train_dataset = ProductsDataset(trainset, train_transform)
    valid_dataset = ProductsDataset(validset, test_transform)
    pred_dataset = ProductsDataset(predset, test_transform)

    return unsplitted_dataset, train_dataset, valid_dataset, pred_dataset, label2id, id2label
    