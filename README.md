# smart-product
Predict category of product by it's description, image and other attributes

## Porposed solution:

1. get features from every possible source(column in df)

    1.1. feature extractor for images

    1.2. feature extractor for descriptions and others text fields

    1.3. get embedding for shop title
(word2vec algorithm)

2. metric learning on every block of features for better separation between classes

3. concatenate all features and train classifier on them

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
