# smart-product
Predict category of product by it's description, image and other attributes

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
