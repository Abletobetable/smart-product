# smart-product
Predict category of product by it's description, image and other attributes.

## Content

- [Stack of technologies](#Stack-of-technologies)
- [Proposed solution](#proposed-solution)
- [Process and results]
- [How to improve?]

## Stack of technologies

- Python
- PyTorch
- HuggingFace
- Transformers
- Timm
- Wandb
- sklearn, metric-learn, unbalanced-learn
- navec


## Proposed solution:

0. [EDA](#Exploratory-data-analysis) to know the data

1. [Preprocessing dataset]

2. [get features]

    2.1. feature extractor for images

    2.2. feature extractor for descriptions, title and attributes

    2.3. get embeddings for shop title and other fields

3. concatenate all features and perfome metric learning and data resampling

4. [train final classifier]

    4.1. Classic ML model

    4.2. NN classifier

## Process and results

### Exploratory data analysis
(notebooks/EDA_kazan2023.ipynb)

Dataset consists of products that are characterized by it's image, name, description, store and some product attributes. Target value is category of product.

The most important finding of the EDA is that the data are not balanced by class. Some classes are present in a single instance. 

While some fields seemingly are completely undiversified: rating and sale indicator.

Every image has shape (512, 512). All shops and products have titles.

custom_characteristics, defined_characteristics and filters are just dictionaries.

description field in html format

### Preprocessing
(main part in notebooks/TextProcessing_kazan2023.ipynb)

First, I expand text fields in separate columns and in DataFrame column with path to images.
Then, I clean text with nltk.tokenizer and regural expressions. 

Title, description and attributes I join together, because I think that bert can extract more complete information and maybe catch some hidden connection between this fields, than if I gave him separate sentences.

I tokenized in words shop titles and keys from dictionary-like fields.

For images I prepare transform pipeline before training and feature extraction: Resize to (224, 224), normalisation and for training also augmentation: flip and random rotation.

### Feature exctraction:
#### From images
(notebooks/Image_kazan2023.ipynb)

I used pretrained [beit](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k) (Vision Transformer) and [EfficientNet](https://huggingface.co/timm/tf_efficientnetv2_b3.in21k_ft_in1k) for exctracting features from huggingface transfomers and timm. I also add option for finetuning models. Here are finetuned models for [beit](https://huggingface.co/abletobetable/image_feature_extractor) and [EfficientNet](https://huggingface.co/abletobetable/smart-product-EfficientNet-v1).

Beit plots:

<img src="img/beit_train_loss.png" alt="drawing" width="325"/> <img src="img/beit_lr.png" alt="drawing" width="325"/> <img src="img/beit_eval_f1_score.png" alt="drawing" width="325"/>

EfficientNet plots:

<img src="img/efficientnet_train_loss.png" alt="drawing" width="325"/> <img src="img/efficientnet_lr.png" alt="drawing" width="325"/> <img src="img/efficientnet_eval_f1.png" alt="drawing" width="325"/>

So, training only on images I get eval f1 score 0.61 with beit and 0.49 with EfficientNet.

I extract features with trained nets and not for futher experimenting with both variants.

#### From Text
(notebooks/TextFeaturing_kazan2023.ipynb)

### details:
- unbalanced dataset:

    try undersampling or oversampling techniques
    
    or use metric learning for better separation between classes
    
- beit alone can't classify categories:

    try CNN, for example or use image features in combination with others
    
- add option for train all feature extractors at the same time:

    image feature extractor loss, text feature extractor loss + final classifier loss

- add metric learning for features
    
    so pipeline will be: pretrained_model -> metric-learm model -> final classifier
    
- apply adapters: 

    for efficient fine-tuning of big (large) models adapters may be great solution
