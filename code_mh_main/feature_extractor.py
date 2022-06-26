import os
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model


from PIL import Image, ImageOps
from numpy.core.records import array
from utils import *


from cnn_model import *

class FeatureExtractor():
    """Feature Extractor model
    """
    def __init__(self, loaded_model=None, use_flatten: bool = False) -> None:
        """Initialisation of the feature extraction model based on either the VGG16 or the given CNN model.

        :param loaded_model: A simple CNN model with 2 FC layers and 2 Activation layers as well as a Flatten layer, defaults to None
        :type loaded_model: tf.keras.Sequential(), optional
        :param use_flatten: Set to True if the feature vector should be retrieved from the Flatten of the simple CNN, otherwise it is retrieved from the first FC layer, defaults to False
        :type use_flatten: bool, optional
        """
        if loaded_model is None:
            ## Define the Base Model
            model = VGG16(weights='imagenet', include_top=True)
            ## Create Feature Extractor Model based on the Base Model
            self.fe_model = Model(inputs=model.input, outputs=model.get_layer("flatten").output, name = "VGG16")
        # TODO: add options where we can use a model outside of CNN
        else:
            if BINARY & use_flatten:        # never used
                self.fe_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[-5].output,
                                      name="SimpleCNN_flatten")
            elif (not BINARY) & use_flatten:  # never used
                self.fe_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[-5].output,
                                      name="MultiCNN_flatten")
            elif not BINARY:
                self.fe_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[-3].output,
                                      name="MultiCNN")
            else:
                self.fe_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[-3].output,
                                      name="SimpleCNN")


    def load_preprocess_img(self, path):
        """Returns the loaded and preprocessed image based on the current feature selector in PIL format as well as in a 4-dim numpy array. 
        The latter element can be used for CNN-based predictions.

        :param path: The path including the file name of the desired image
        :type path: str
        :return: 
            - **img_PIL** (`PIL.Image`) - The loaded, converted and resized image in repsect to the current Feature Selector by using PIL
            - **x** (`numpy.ndarray`) - A 4-dim pre-processed numpy array containing that single image which can used for prediction
        """
        img_PIL = None
        x = None

        if self.fe_model.name == 'VGG16':
            img_PIL = Image.open(path).convert('RGB')
            img_PIL = img_PIL.resize(self.fe_model.input_shape[1:3])
            x = np.array(img_PIL)
            #print("1: ", np.shape(x))
            x = np.expand_dims(x, axis=0)
            #print(np.shape(x))
            x = preprocess_input(x)

        elif self.fe_model.name in ['SimpleCNN', 'MultiCNN']:
            img_PIL = Image.open(path).convert('L')
            img_PIL = img_PIL.resize(self.fe_model.input_shape[1:3])
            x = np.array(img_PIL, dtype=np.float64)
            #print("1: ", np.shape(x))
            x = np.expand_dims(x, -1)
            x = np.expand_dims(x, axis=0)
            #print(np.shape(x))
            x /= 255. 

        return img_PIL, x
    
    def extract_features(self, x):
        """Create the feature vector for given image (numpy array)

        :param x: 4-dim numpy array referring to the image
        :type x: numpy.ndarray
        :return: *self* (`numpy.ndarray`) - One-dimensional feature vector
        """
        return self.fe_model.predict(x)[0]


if __name__ == "__main__":

    from dataset import DataSet
    from dataset import get_dict_datasets

    ## LOAD the DATASETS
    use_CNN_feature_embeddding = False
    use_all_datasets = True
    if len(DATA_DIR_FOLDERS) > 0:
        use_all_datasets = False
    dict_datasets = get_dict_datasets(use_CNN_feature_embeddding, use_all_datasets)

    print(f'Possible dataset: {dict_datasets.keys()}')

    sel_model = CNNmodel(dict_datasets[DATA_DIR_FOLDERS[0]])  # TODO: careful always takes first data set!
    sel_model.load_model()
    sel_model._preprocess_img_gen()

    fe = FeatureExtractor(loaded_model=sel_model.model)

    print(fe.fe_model.name)
    fe.load_preprocess_img(dict_datasets[sel_model.selected_dataset].data_t[99].img_path)


