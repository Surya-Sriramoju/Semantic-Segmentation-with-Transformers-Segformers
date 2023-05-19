# Semantic-Segmentation-with-Transformers-Segformers

### Dependencies ###
* torch
* matplotlib
* einops
* albumentations
* cityscapesscripts
* opencv
* torchmetrics
* pillow

### steps to train and perform inference ###
* Enter into segformers directory
* add the path to the dataset in the main file, for the variable root_dir
* Uncomment the training code snippet and run the main.py file
* The weights will be stored in a folder called weights
* For inference testing, comment the training snippet and uncomment the testing snippet.
* Provide the path to the weights and run the main.py file