"""
function for calculating text features
"""

import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Literal

def create_average_navec_embed(
    navec_model, 
    sentences: pd.Series(), 
    category_ids: pd.Series(), 
    product_ids: pd.Series() = None, 
    split: Literal['train', 'test'] = 'train',
) -> np.array:
    """
    get average embeddings for shop titles 
    from pretrained navec embdeddings

    Parameters
    ----------
        navec_model: 
            model with pretrained embeddings

        sentences (pd.Series()): 
            iterable with sentences to perfome average embeddings

        category_ids (pd.Series()): 
            iterable with categories
        
        product_ids (pd.Series()): 
            iterable with categories

        split ('train', 'test'):
            flag for function to know if there are categories in dataset

    Return
    ------
        processed_attributes (pd.Series()): 
            cleaned and processed attributes
    """

    if split == 'train':
        X = np.zeros((len(sentences), 300+2))
    else:
        X = np.zeros((len(sentences), 300+1))

    # init tokenizer
    tokenizer = nltk.WordPunctTokenizer()

    # main loop
    for i, (title, cat, prod) in enumerate(tqdm(list(zip(sentences, 
                                                       category_ids, 
                                                       product_ids)))):

        # get average embed
        title_embed = []
        for token in tokenizer.tokenize(title.lower()):
            if token in navec_model:
                title_embed.append(navec_model[token])
        
        if len(title_embed) > 0: # get average embedding
            embed = np.mean(title_embed, axis = 0)

        else: # init with zeros
            embed = np.zeros((300, ))

        # write in array
        if split == 'train':
            X[i, 0] = cat
            X[i, 1] = prod
            X[i, 2:] = embed

        else:
            X[i, 0] = prod
            X[i, 1:] = embed

    return X

def preprocess_attributes(attributes: pd.Series()) -> pd.Series():
    """
    parse attributes column and preprocess it

    Parameters
    ----------
        attributes (pd.Series()): 
            iterable for parse and processing attributes

    Return
    ------
        processed_attributes (pd.Series()): 
            cleaned and processed attributes
    """

    processed_attributes = []

    # main loop
    for text in tqdm(attributes):
        
        if len(text) > 2: # means string not empty

            # cleaning list
            lst = text.strip('][').split(', ')
            lst = [re.sub("'", "", elem.strip()) for elem in lst]

            # join in meaningful string
            string = ', '.join(lst)
        
        else:
            string = ''

        processed_attributes.append(string)

    return pd.Series(processed_attributes, name='attributes')

def filter_description(descriptions: pd.Series()) -> pd.Series(): 
    """
    filter using regular expressions description field

    Parameters
    ----------
        descriptions (pd.Series()): 
            iterable for parse and processing descriptions

    Return
    ------
        processed_descriptions (pd.Series()): 
            cleaned and processed descriptionas
    """

    filtered = []

    # main loop
    for text in tqdm(descriptions):

        if len(str(text)) > 3: # so not including nan values

            text = re.sub(r'<[^<>]+>', '', text)
            text = re.sub(r'&[^;]+;', ';', text)

        else:
            text = ''

        filtered.append(text)

    return pd.Series(filtered, name='description')




