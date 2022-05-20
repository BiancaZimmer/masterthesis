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
- Change the utils.py so the global variables fit your data
- DATA_DIR should contain the data in a folder with name <data_name>
- Inside should be two folders named "test" and "train" wherein your images in separate folders for the classes shall be found