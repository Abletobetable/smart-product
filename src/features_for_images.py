"""
functions for initialisation model and trainer 
"""

import numpy as np
from tqdm import tqdm

from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import TrainingArguments, Trainer, DefaultDataCollator

from datasets import load_metric

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_metrics(eval_pred):

    metric = load_metric('f1')

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

def create_model_and_trainer(model_checkpoint: str, 
                             train_dataset, valid_dataset, 
                             num_epochs: int, batch_size: int,
                             freeze: bool, num_labels: int, 
                             label2id: dict(), id2label: dict()):
    """
    1. init model for training from model_checkpoint

    2. init trainer with datasets and other params

    Parameters
    ----------
        model_checkpoint (str): 
            repo in huggingface hub with model an processor
        
        freeze (bool):
            if True, set require_grad = False in feature_extraction layers, 
            only classifier module will train

    Return
    ------
        model

        trainer
    """

    processor = BeitImageProcessor.from_pretrained(model_checkpoint)
    model = BeitForImageClassification.from_pretrained(model_checkpoint, 
                                                    num_labels=num_labels,
                                                    id2label=id2label,
                                                    label2id=label2id,
                                                    ignore_mismatched_sizes=True,)



    model.classifier = nn.Linear(in_features=768, out_features=num_labels, bias=True)

    # freeze feature extractor params 
    # (only classifier are trainable)
    if freeze:
        for param in model.beit.parameters():
            param.requires_grad = False

    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir="image_feature_extractor",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    return model, trainer

def get_image_features(dataset, model, device: str) -> np.ndarray:
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

    Return
    ------
        X (np.ndarray()): array with features
    """
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=2)

    model.to(device)

    X = np.zeros((len(dataset), 768))

    model.eval()
    for i, batch in enumerate(tqdm(loader)):

        with torch.no_grad():

            # beit output: last_hidden_state, pooler_output
            X[i, :] = model.beit(batch['pixel_values'].to(device)).pooler_output.cpu()

    return X






