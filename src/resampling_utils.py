"""
functions for preprocessing final data: resample and split
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import ParameterGrid

from metric_learn import LFDA

MAGIC_SEED = len('DS Internship 2023 | KazanExpress')

def reduce_dimension(
    X_train: np.array,
    y_train: np.array, 
    X_valid: np.array,
    X_predict: np.array,
    num_features: int = 512) -> np.array:
    """
    reduce dimension of features using Local Fisher Discriminant Analysis
    then transform X_train and X_valid

    Parameters
    ----------
        X_train (np.array):
            dataset for under-sampling

        y_train (np.array):
            targets for supervised reduction

        X_valid (np.array):
            dataset for under-sampling

        X_predict (np.array):
            dataset for under-sampling

        num_features (int):
            num_features for reducing

    Return
    ------
        X_reduced_train, X_reduced_valid (np.array)
    """

    metric_model = LFDA(n_components=num_features)
    metric_model.fit(X_train, y_train)
    X_reducted_train = metric_model.transform(X_train)
    X_reducted_valid = metric_model.transform(X_valid)
    X_reducted_predict = metric_model.transform(X_predict)

    return X_reducted_train, X_reducted_valid, X_reducted_predict

def under_sample(
    X_train: np.array,
    y_train: np.array,
    max_count: int = 100) -> np.array:
    """
    resample given dataset using undersample method

    Parameters
    ----------
        X_train (np.array()):
            dataset for under-sampling

        y_train (np.array()):
            dataset for under-sampling

        max_count (int):
            maximum possible amount of samples in every class

    Return
    ------
        X_resampled, y_resampled (np.array())
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

def over_sample(
    X_train: np.array,
    y_train: np.array,
    min_count: int = 25) -> np.array:
    """
    resample given dataset using oversample method

    Parameters
    ----------
        X_train (np.array()):
            dataset for over sampling

        y_train (np.array()):
            dataset for over sampling

        max_count (int):
            minimum possible amount of samples in every class

    Return
    ------
        X_resampled, y_resampled (np.array())
    """

    indexes = pd.DataFrame(pd.Series(y_train).value_counts(),
                           columns=['count']).iloc[:, 0].index

    counts = pd.DataFrame(pd.Series(y_train).value_counts(),
                          columns=['count']).iloc[:, 0]

    # sampling stratagy for RandomOverSampler
    weights = dict()
    for (idx, cnt) in list(zip(indexes, counts)):
        if cnt < min_count:
            weights[idx] = min_count 

    rus = RandomOverSampler(
        random_state=MAGIC_SEED,
        sampling_strategy=weights,
    )

    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    print('Shape before over-sampling:', X_train.shape)
    print('Shape after over-sampling:', X_resampled.shape)

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

def stratified_train_test_split_df(X_train: pd.DataFrame) -> pd.DataFrame:
    """

    the same as stratified_train_test_split_numpy, but works with pd.DataFrame

    duplicate single objects for stratified split
    after duplicating apply train_test_split from sklearn

    Parameters
    ----------
        X_train (pf.DataFrame()):
            dataset to perfome splitting

    Return
    ------
        X_train_splitted, X_valid_splitted (pd.DataFrame)
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

def grid_search(X_train, y_train,
                X_valid, y_valid,
                params: dict(), lfda_components, clf) -> dict():
    """
    grid search for best combination of parameters:
    dimension reduction + under-sample + over-sample

    Parameters
    ----------
        params (dict):
            dict with parameters for grid search

        lfda_components (list):
            number of components for lfda algorithm

        clf :
            estimator for searching best parameters
    Return
    ------
        dict with combinations and scores
    """

    scores = dict()

    for num_comp in tqdm(lfda_components):

        X_reduced_train, X_reduced_valid = reduce_dimension(
                    X_train,
                    y_train,
                    X_valid,
                    num_features = num_comp
                )

        for pair in list(ParameterGrid(params)):

            # no resampling
            if pair['lower_bound'] == 1 and pair['upper_bound'] == 2134:

                clf.fit(X_resampled, y_resampled)
                pred = clf.predict(X_reduced_valid)
                f1 = f1_score(y_valid, pred, average='weighted')
                scores[(pair['lower_bound'], pair['upper_bound'],
                        num_comp)] = f1
                print(pair, ':', f1)
                print()

            # only undersampling
            elif pair['lower_bound'] == 1:

                X_resampled, y_resampled = under_sample(X_reduced_train, y_train,
                                                        pair['upper_bound'])
                clf.fit(X_resampled, y_resampled)
                pred = clf.predict(X_reduced_valid)
                f1 = f1_score(y_valid, pred, average='weighted')
                scores[(pair['lower_bound'], pair['upper_bound'],
                        num_comp)] = f1
                print(pair, ':', f1)
                print()

            # only oversampling
            elif pair['upper_bound'] == 2134:

                X_resampled, y_resampled = over_sample(X_reduced_train, y_train,
                                                        pair['lower_bound'])
                clf.fit(X_resampled, y_resampled)
                pred = clf.predict(X_reduced_valid)
                f1 = f1_score(y_valid, pred, average='weighted')
                scores[(pair['lower_bound'], pair['upper_bound'],
                        num_comp)] = f1
                print(pair, ':', f1)
                print()

            # reduce and resample
            else:

                X_resampled, y_resampled = under_sample(X_reduced_train, y_train, pair['upper_bound'])
                X_resampled, y_resampled = over_sample(X_resampled, y_resampled, pair['lower_bound'])

                # fit and validate estimator
                clf.fit(X_resampled, y_resampled)
                pred = clf.predict(X_reduced_valid)
                f1 = f1_score(y_valid, pred, average='weighted')
                scores[(pair['lower_bound'], pair['upper_bound'],
                        num_comp)] = f1
                print(pair, ':', f1)
                print()

    return scores
