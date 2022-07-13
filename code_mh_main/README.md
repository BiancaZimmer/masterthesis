# Explaining Image Classifications with Near Misses, Near Hits and Prototypes: Supporting Domain Experts in Understanding Decision oundaries
___
###  Marvin Herchenbach, Dennis Müller, Stehphan Scheele, Ute Schmid
___

## Abstract

We propose a method for explaining the results of black box image classifiers to domain experts and end users, combining two example-based explanatory approaches: Firstly, prototypes as representative data points for classes, and secondly, contrastive example comparisons in the form of near misses and near hits. A prototype globally explains the relevant characteristics for a whole class, whereas near hit and near miss examples explain the local decision boundary of a specific prediction. The proposed approaches are evaluated with respect to parameter selection and suitability on two different data sets – the well-known MNIST as well as a real world data set from industrial quality control. Finally, it is shown how global and local example-based can be combined and realized within a demonstrator system.

### Structure

```  
├── README.md
├── classify.py
├── cnn_model.py
├── data                        || Folder of data
├── dataentry.py
├── dataset.py
├── feature_extractor.py
├── flaskutil.py
├── helpers.py
├── kernels.py
├── mmd_critic.py
├── near_miss_hits_selection.py
├── offline_feat_extraction.py
├── prototype_selection.py
├── requirements.txt
├── static                      || Folder for static data - e.g. data, embeddings, models - of the FLASK app
├── templates                   || Folder for html files of the FLASK app
├── utils.py
└── xai_demo.py

```

## Ressources

- Quality Dataset: https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- Original Code: https://gitlab.cc-asp.fraunhofer.de/sees/vis-ml2022-mh

## Contact

* **Marvin Herchenbach** - If you have questions you can contact me under marv.her@t-online.de

# By Bianca

## Setup
- change the utils.py so the global variables fit your data
  - DATA_DIR should contain the data in a folder with name <data_name>
  - inside should be folders named "test", "validation" and "train" wherein your images in separate folders for the classes shall be found
  - change DATA_DIR_FOLDERS to the name of your data folder(s) you want to use
- run cnn_model.py to fit your model - important: for this you'll have to adjust the boolean switches in \_\_main__
- run dataentry.py to get all feature embeddings - important: change the suffix_path in \_\_main__ to whatever the name of your model
- run prototype_selection.py - when first running this you might have to evaluate the best number of prototypes via screeplot
- run xai_demo.py - this will run a flask app

Alternative to running the flask app:
- run near_miss_hits_selection.py - to get near miss, hits and prototypes of a random (or fixed) test image

## Test settings can be found in the following modules:
- dataset.py
- dataentry.py

## Main programs can be found in:
- xai_demo: main file!!!
- cnn_model: trains a cnn model with training data, evaluates the misclassified data, prints plots

## Things I changed fundamentally:
- added an option "DATA_DIR_FOLDERS" with which one can choose which data folder(s) should be used
- added an option "BINARY" which, if set to True, runs the original code with a binary CNN, if set to False a
multi-class CNN will be used. Option was incorporated in all python files and flask app
- added code so that a multi-class CNN can be used by only using the BINARY=False switch in the utils.py file
- added "TOP_N_NMNH" in utils.py where one can set the number of near miss/hits to be selected globally
- changed the html/flask app:
  - made the footer flexible
  - changed paths to pictures so they would be shown
- added preprocessing.py which is a complete pipeline to preprocess a data set for the 
flask app
- added the crop_to_square() function in helpers.py