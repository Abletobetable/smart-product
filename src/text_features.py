"""
function for calculating text features
"""

import re
import json
import nltk
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Literal

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from datasets import load_metric

import torch
from torch.utils.data import DataLoader

def create_average_navec_embed(
    navec_model,
    sentences: pd.Series,
    category_ids: pd.Series,
    product_ids: pd.Series = None,
    split: Literal['train', 'test'] = 'train',
) -> np.array:
    """
    get average embeddings for shop titles 
    from pretrained navec embdeddings

    Parameters
    ----------
        navec_model: 
            model with pretrained embeddings

        sentences (pd.Series): 
            iterable with sentences to perfome average embeddings

        category_ids (pd.Series): 
            iterable with categories
        
        product_ids (pd.Series): 
            iterable with categories

        split ('train', 'test'):
            flag for function to know if there are categories in dataset

    Return
    ------
        processed_attributes (pd.Series): 
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

def preprocess_attributes(attributes: pd.Series) -> pd.Series:
    """
    parse attributes column and preprocess it

    Parameters
    ----------
        attributes (pd.Series): 
            iterable for parse and processing attributes

    Return
    ------
        processed_attributes (pd.Series): 
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

def filter_description(descriptions: pd.Series) -> pd.Series:
    """
    filter using regular expressions description field

    Parameters
    ----------
        descriptions (pd.Series): 
            iterable for parse and processing descriptions

    Return
    ------
        processed_descriptions (pd.Series): 
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

def filter_characteristics(df: pd.DataFrame) -> pd.Series():
    """
    every field in 
    ['custom_characteristics', 'defined_characteristics', 'filters']
    (if it not empty) is dict, so I will get every key from fields 
    and concatenate in one string

    Parameters
    ----------
        df (pd.DataFrame): 
            iterable for parse and processing characteristics
    Return
    ------
        processed_characteristics (pd.Series): 
            cleaned and processed characteristics
    """

    filtered = []

    # custom
    for (cust, defin, filt) in tqdm(list(zip(df['custom_characteristics'],
                                             df['defined_characteristics'],
                                             df['filters']))):

        row = []

        # custom
        if cust != '{}':
            try:
                cust = cust.replace("\'", "\"")
                d = json.loads(cust)
                row = []
                for key in d.keys():
                    row.append(key.strip().lower())
            except:
                pass

        # defined
        if defin != '{}':
            try:
                defin = defin.replace("\'", "\"")
                d = json.loads(defin)
                for key in d.keys():
                    row.append(key.strip().lower())
            except:
                pass

        # filters
        if filt != '{}':
            try:
                filt = filt.replace("\'", "\"")
                d = json.loads(filt)
                for key in d.keys():
                    row.append(key.strip().lower())
            except:
                pass

        if len(row) > 0:
            filtered.append(', '.join(row))
        else:
            filtered.append('')

    return pd.Series(filtered, name='characteristics')

def concatenate_text_fields(categories: pd.DataFrame,
                            prep_title: pd.Series,
                            prep_attrib: pd.Series,
                            prep_descrip: pd.Series) -> pd.DataFrame:

    """
    conctenate all text content using ". " between fields

    Parameters
    ----------
        categories (pd.DataFrame): 
            categories ids and product ids(if train part)

        prep_title (pd.Series): 
            titles

        prep_attributes (pd.Series): 
            attributes

        prep_descrip (pd.Series): 
            descriptions

    Return
    ------
        concated_text (pd.Series): 
            dataframe with concated text content
    """

    concat = []

    for (t, a, d) in list(zip(prep_title, prep_attrib, prep_descrip)):

        if t[-1] not in string.punctuation:
            text = t + ". " + a
        else:
            text = t + " " + a

        if len(a) > 0 and a[-1] not in string.punctuation:
            text += ". " + d
        else:
            text += " " + d

        if len(d) > 0 and d[-1] not in string.punctuation:
            text += "."

        concat.append(text)

    concated_text = pd.DataFrame(categories)
    concated_text['text'] = concat

    return concated_text

def compute_metrics(eval_pred):
    """
    this function called by trainer and compute given metric
    """

    metric = load_metric('f1')

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions,
                          references=labels, average='weighted')

def create_model_and_trainer(model_checkpoint: str,
                             train_dataset, valid_dataset,
                             num_epochs: int, batch_size: int,
                             freeze: bool, num_labels: int,
                             label2id: dict(), id2label: dict(),
                             report_to: Literal['wandb', 'local'],
                             push_to_hub: bool):
    """
    1. init model for training from model_checkpoint

    2. init trainer with datasets and other params

    Parameters
    ----------
        model_checkpoint (str): 
            repo in huggingface hub with model, tokinzer abd config
        
        freeze (bool):
            if True, set require_grad = False in feature_extraction layers, 
            only classifier module will train

        push_to_hub (bool):
            if need to push model on huggingface model hub
            (must be logined in huggingface-cli)

    Return
    ------
        model

        trainer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
                                                  num_labels=num_labels,
                                                  id2label=id2label,
                                                  label2id=label2id,
                                                  ignore_mismatched_sizes=True,)

    # freeze feature extractor params
    # (only classifier are trainable)
    if freeze:
        for param in model.bert.parameters():
            param.requires_grad = False

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="text_feature_extractor",
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=1000,
        learning_rate=5e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        push_to_hub=push_to_hub,
        report_to=report_to
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return model, trainer

def get_text_features(dataset, model, device: str) -> np.ndarray:
    """
    get features for image dataset from model provided
        Parameters
    ----------
        dataset : 
            dataset with images

        model : 
            model with weights
        
        device (str):
            cpu or gpu use for extraction
            
        model_type (['ViT', 'CNN']):
            set model for feature extraction
    Return
    ------
        X (np.array): array with features 
        and
        1 column: product ids
        2 column: category ids
    """
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=2)

    model.to(device)

    num_features = 768

    # in predict dataset we dont have category_id column
    if 'category_id' in dataset[0].keys():
        X = np.zeros((len(dataset), num_features+2))
    else:
        X = np.zeros((len(dataset), num_features+1))

    model.eval()
    for i, batch in enumerate(tqdm(loader)):

        with torch.no_grad():

            output = model.bert(batch['input_ids'].to(device)).pooler_output.cpu()

            if 'category_id' in dataset[0].keys():
                X[i, 2:] = output
                X[i, 0] = batch['category_id']
                X[i, 1] = batch['product_id']
            else:
                X[i, 1:] = output
                X[i, 0] = batch['product_id']

    return X
