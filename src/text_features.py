"""
function for clculating text features
"""

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
    split = Literal['train', 'test'] = 'train',
) -> np.array:
    """
    get average embeddings for shop titles 
    from pretrained navec embdeddings
    """

    if split == 'train':
        X = np.zeros((len(sentences), 300+2))
    else:
        X = np.zeros((len(sentences), 300+1))

    # init tokenizer
    tokenizer = nltk.WordPunctTokenizer()

    # main loop
    for i, title, cat, prod in enumerate(tqdm(list(zip(sentences, 
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






