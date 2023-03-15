"""
functions for preprocessing final data: resample and split
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

MAGIC_SEED = len('DS Internship 2023 | KazanExpress')

def under_sample(
    X_train: np.array, 
    y_train: np.array, 
    max_count: int = 100) -> np.array():
    """
    resample given dataset using undersample method

    Parameters
    ----------
        X_train (np.array()):
            dataset to perfome splitting

        X_train (np.array()):
            dataset to perfome splitting

        max_count (int):
            maximum possible amount of samples in every class

    Return
    ------
        X_train, y_train, X_valid, y_valid (np.array())
    """

    indexes = pd.DataFrame(pd.Series(y_train).value_counts(), 
                           columns=['count']).iloc[:, 0].index

    counts = pd.DataFrame(pd.Series(y_train).value_counts(), 
                          columns=['count']).iloc[:, 0]

    # sampling stratagy for RandomUnderSampler
    weights = dict()
    for (idx, cnt) in list(zip(indexes, counts)):
        if cnt > max_count:
            weights[idx] = max_count    

    rus = RandomUnderSampler(
        random_state=MAGIC_SEED, 
        sampling_strategy=weights,
    )

    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    print('Shape before under-sampling:', X_train.shape)
    print('Shape after under-sampling:', X_resampled.shape)

    return X_resampled, y_resampled

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