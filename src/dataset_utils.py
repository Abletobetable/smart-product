"""
couple dataset functions for preprocessing
"""


import os
import cv2
import json
import numpy as np
import pandas as pd
from typing import Literal

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import Dataset
from transformers import AutoTokenizer

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

def create_labels_mapping(dataset: pd.DataFrame()) -> dict():
    """
    create label2id and id2label dicts 
    for mapping between categories and labels

    Parameters
    ----------
        dataset (pd.DataFrame()): 
            dataframe with all content inside
        
    Return
    ------
        label2id (dict()): 
            label <-> id mapping

        id2label (dict()): 
            id <-> label mapping
    """

    labels = sorted(list(set(dataset['category_id'])))
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
    train_df, valid_df = stratified_train_test_split_df(preprocessed_dataset_train)

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

def stratified_train_test_split_numpy(X_train: np.array) -> np.array:
    """

    the same as stratified_train_test_split_pd, but works with np.ndarray()

    duplicate single objects for stratified split
    after duplicating apply train_test_split from sklearn

    Parameters
    ----------
        X_train (np.array()):
            dataset to perfome splitting

    Return
    ------
        X_train, y_train, X_valid, y_valid (np.array())
    """
    
    # get category_id
    categories = pd.Series(X_train[:, 0])

    # count unpopular values
    cat_count = pd.DataFrame(categories.value_counts(), columns=['count'])

    # get index = category
    unpopular_categ = list(cat_count[cat_count['count'] == 1].index)
    print('rare products:', unpopular_categ)

    # duplicate
    X_duplicated = X_train
    for categ in unpopular_categ:
        new_row = X_train[np.where(X_train == categ)[0][0], :].reshape(1, -1)
        X_duplicated = np.concatenate((X_duplicated, new_row), axis=0)

    X_train_splitted, X_valid_splitted = train_test_split(X_duplicated, 
                                            test_size=0.2, 
                                            random_state=MAGIC_SEED, 
                                            stratify=X_duplicated[:, 0])

    X_train, y_train = X_train_splitted[:, 2:], X_train_splitted[:, 0]
    X_valid, y_valid = X_valid_splitted[:, 2:], X_valid_splitted[:, 0]
                                        
    return X_train, y_train, X_valid, y_valid

def stratified_train_test_split_df(X_train: pd.DataFrame()) -> pd.DataFrame():
    """

    the same as stratified_train_test_split_numpy, but works with pd.DataFrame()

    duplicate single objects for stratified split
    after duplicating apply train_test_split from sklearn

    Parameters
    ----------
        X_train (pf.DataFrame()):
            dataset to perfome splitting

    Return
    ------
        X_train_splitted, X_valid_splitted (pd.DataFrame())
    """
    
    # get category_id
    categories = X_train['category_id']

    # count unpopular values
    cat_count = pd.DataFrame(categories.value_counts())

    # get index = category
    unpopular_categ = list(cat_count[cat_count['category_id'] == 1].index)
    print('rare categories:', unpopular_categ)

    # duplicate
    X_duplicated = X_train
    for categ in unpopular_categ:

        # get row for duplicate
        idx = X_train[X_train['category_id'] == categ].index[0]
        new_row = X_train.iloc[idx, :].to_list()

        # append new row
        X_duplicated.loc[len(X_duplicated.index)] = new_row

    X_train_splitted, X_valid_splitted = train_test_split(X_duplicated, 
                                            test_size=0.2, 
                                            random_state=MAGIC_SEED, 
                                            stratify=X_duplicated['category_id'])
                                        
    return X_train_splitted, X_valid_splitted

def create_text_datasets(prep_train_dataset: pd.DataFrame(), 
                         prep_predict_dataset: pd.DataFrame(), 
                         tokenizer_checkpoint: str) -> pd.DataFrame():
    """
    prepare text datasets using huggingface 'datasets' and 'tokenizers'

    Parameters
    ----------
        prep_train_dataset (pf.DataFrame()):
            train part of dataset

        prep_predict_dataset (pf.DataFrame()):
            predict part of dataset. 
            Needed to convert data in tensors

        tokenizer_checkpoint (str):
            path where to get tokenizer for processing datasets

    Return
    ------
        unsplitted_dataset:
            unsplitted train dataset for extracting features

        train_dataset:
            train converted in tensors
        
        valid_dataset:
            validation converted in tensors

        predict_dataset:
            predict dataset for feature exctraction
            
        label2id, id2label:
            mapping for getting label <-> id and vice versa
    """

    # split
    train_dataset, valid_dataset = stratified_train_test_split_df(prep_train_dataset)

    # to huggingface format
    train_dataset = Dataset.from_pandas(train_dataset)
    valid_dataset = Dataset.from_pandas(valid_dataset)
    unsplitted_dataset = Dataset.from_pandas(prep_train_dataset)
    predict_dataset = Dataset.from_pandas(prep_predict_dataset)

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    # tokenize
    # train split
    train_dataset = train_dataset.map(lambda examples: tokenizer(
        examples["text"], 
        padding="max_length", 
        max_length=512, 
        truncation=True), batched=True
    )
        
    # validation split
    valid_dataset = valid_dataset.map(lambda examples: tokenizer(
        examples["text"], 
        padding="max_length", 
        max_length=512, 
        truncation=True), batched=True
    )

    # unsplitted
    unsplitted_dataset = unsplitted_dataset.map(lambda examples: tokenizer(
        examples["text"], 
        padding="max_length", 
        max_length=512, 
        truncation=True), batched=True
    )

    # predict
    predict_dataset = predict_dataset.map(lambda examples: tokenizer(
        examples["text"], 
        padding="max_length", 
        max_length=512, 
        truncation=True), batched=True
    )
    
    # get label <-> id mapping
    label2id, id2label = create_labels_mapping(train_dataset)

    return unsplitted_dataset, train_dataset, valid_dataset, \
           predict_dataset, label2id, id2label


