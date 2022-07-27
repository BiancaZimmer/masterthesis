import os
import ntpath
import numpy as np
from lazy import lazy
from PIL import Image, ImageOps
import math
import time

from utils import *


class DataEntry:
    """DataEntry object that refers to a data sample (image) and contains all its information
    """
    img_path : str
    feature_file : str
    ground_truth_label : str

    def __init__(self,fe ,dataset:str,img_path : str):
        """Initialise DataEntry object, preferably via DataSet initialisation with List Comprehension of DataEntry objects. 

        :param fe: FeatureExtractor model used for feature extraction (Note: A name view Keras Model is mandatory)
        :type fe: FeatureExtractor
        :param dataset: Name of the `DataSet` in order to refer each DataEntry object to it.
        :type dataset: str
        :param img_path: Local path of the image file (including file name)
        :type img_path: str
        """

        #extract all avaiable folders (classes) of the selected dataset 
        #and the corresponding of the selected image path
        DIR_TRAIN_DATA = os.path.join(DATA_DIR, dataset, 'train')
        avaiable_classes = [item for item in os.listdir(DIR_TRAIN_DATA) if os.path.isdir(os.path.join(DIR_TRAIN_DATA,item))]
        folder_name = ntpath.basename(ntpath.dirname(img_path))

        #entire path to the image
        self.img_path = img_path 
        #image file name
        self.img_name = ntpath.basename(img_path) 
        #class of the image referring to the folder name
        self.ground_truth_label = folder_name 
        #extract index of folder name/class -> int value for the class
        self.y = avaiable_classes.index(folder_name) 
        self.fe = fe #selected FeatureExtractor 

        # construct PATH where the feature vector is saved
        DIR_FEATURE_EMBEDDING_DATASET = os.path.join(DIR_FEATURES, fe.fe_model.name, dataset)
        # create the folder for feature vectors if it is not created yet
        if os.path.exists(DIR_FEATURE_EMBEDDING_DATASET) == False:
            os.makedirs(DIR_FEATURE_EMBEDDING_DATASET)

        pre, _ = os.path.splitext(os.path.join(DIR_FEATURE_EMBEDDING_DATASET,ntpath.basename(img_path)))
        #entire path to the feature vector of the images
        self.feature_file = pre + '.npy' 
        pass

    def initiate_feature_embedding(self):
        """Lazy function for extracting the Feature Vector of a single image for the first time

        :return: *self* (`numpy.ndarray`) - One-dimensional feature vector
        """
        if os.path.exists(self.feature_file):
            # check is implemented whether there are pictures with the same names
            print("WARNING: feature file of", self.img_path, " already existed.")
            return np.load(self.feature_file, allow_pickle=True)
        else:
            _, x = self.__compute_image
            feat = self.fe.extract_features(x)
            np.save(self.feature_file, feat)
            # print(self.feature_file)
            return feat


    @lazy
    def feature_embedding(self):
        """Lazy function for extracting or loading the Feature Vector of a single image

        :return: *self* (`numpy.ndarray`) - One-dimensional feature vector
        """
        if os.path.exists(self.feature_file):
            return np.load(self.feature_file, allow_pickle=True)
        else:
            _, x = self.__compute_image
            feat = self.fe.extract_features(x)
            np.save(self.feature_file, feat)
            print(self.feature_file)
            return feat

    @lazy
    def __compute_image(self):
        """Lazy function for loading and pre-processing a single image

        :return: *self* (`numpy.ndarray`) - The loaded and preprocessed image based on the current `FeatureExtractor`
        """
        return self.fe.load_preprocess_img(self.img_path)

    def image_numpy(self, img_size: int = None, mode='L'):
        """Function for loading and converting a single image into a numpy array. If img_size is given it will be resized.

        :param img_size: Select a image size (commonly used `128`), defaults to None
        :type img_size: int, optional
        :param mode: Color mode of the output image. use 'L' for grayscalale and 'RGB' for RGB, defaults to 'L'

        :return: *self* (`numpy.ndarray`) - Normalized image as numpy array
        """
        x = Image.open(self.img_path).convert(mode)
        if img_size is not None:
            x = x.resize((img_size, img_size))
           
        x = np.array(x, dtype=np.float64)

        #Normalize to [0,1]
        x /= 255.

        # print(np.shape(x))
        return x
       

def code_from_dataentry(dataset, suffix_path=''):
    from cnn_model import get_CNNmodel
    from feature_extractor import FeatureExtractor

    # CODE FROM DATAENTRY.PY
    print("----- CREATION OF FEATURE EMBEDDINGS -----")
    new_embedding = True
    feature_embeddings_to_initiate = "current"

    while new_embedding:
        tic = time.time()
        # set feature extractor
        if feature_embeddings_to_initiate == "current":
            # gets FE from a loaded CNN with the dataset name and a suffix
            fe = FeatureExtractor(loaded_model=get_CNNmodel(dataset, suffix_path=suffix_path))
        else:
            # Standard FE for general model:
            fe = FeatureExtractor()  # loaded_model=VGG16(weights='imagenet', include_top=True)
        print("Initiating ", fe.fe_model)

        # get all data entries
        image_path = os.path.join(DATA_DIR, dataset)
        data = [DataEntry(fe, dataset, os.path.join(path, file)) for path, _, files in os.walk(image_path) for file in
                files if file != ".DS_Store"]

        # create feature embeddings for all data entries
        for count, d in enumerate(data):
            d.initiate_feature_embedding()
            if count % 100 == 0:
                print(count, " feature embeddings created")
        print("")

        toc = time.time()
        print("Creating Feature Embeddings needed: ",
              "{}h {}min {}sec ".format(round(((toc - tic) / (60 * 60))), math.floor(((toc - tic) % (60 * 60)) / 60),
                                        ((toc - tic) % 60)))

        a = input("Do you want to train another feature embedding? [y/n]")
        if a == "n":
            new_embedding = False
        else:
            feature_embeddings_to_initiate = input("Which feature extractor would you like to train next? [VGG16/...]")


# python code_mh_main/dataentry.py #works as of 20/05/2022
if __name__ == '__main__':
    # run this to generate all feature embeddings
    # needs a trained model for the feature embeddings else VGG16 is used
    dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information.")
    while dataset_to_use == "help":
        print("We need the folder name of a data set that is saved in your DATA_DIR. Usually that would be"
              "one of the names you specified in the DATA_DIR_FOLDERS list. e.g. 'mnist'")
        dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information.")
    suffix_path = input("What is the suffix of your cnn_model? Type a string. e.g. '_testcnn'")
    code_from_dataentry(dataset_to_use, suffix_path)

#### OLD CODE #####
    # import time
    # from feature_extractor import *
    # tic = time.time()
    #
    # dataset = DATA_DIR_FOLDERS[0]  # careful takes first data set hardcoded!
    #
    # # set feature extractor
    # # gets FE from a loaded CNN with the dataset name and a suffix
    # # fe = FeatureExtractor(loaded_model=get_CNNmodel(dataset, suffix_path="_multicnn"))
    # # Standard FE for general model:
    # # fe = FeatureExtractor()
    # # fe.set_femodel("OCT_retrained_graph_2.pb", "retina1")
    #
    # fe = FeatureExtractor()  # loaded_model=VGG16(weights='imagenet', include_top=True)
    # # VGG16(weights='imagenet', include_top=True).input needs to be possible
    #
    # # get all data entries
    # image_path = os.path.join(DATA_DIR, dataset)
    # data = [DataEntry(fe, dataset, os.path.join(path, file)) for path, _, files in os.walk(image_path) for file in files if file != ".DS_Store"]
    #
    # # create feature embeddings for all data entries
    # for count, d in enumerate(data):
    #     # print(d.feature_embedding)
    #     d.initiate_feature_embedding()
    #     if count % 100 == 0:
    #         print(count, " feature embeddings created")
    # print("")
    #
    # toc = time.time()
    # print("{}h {}min {}sec ".format(round(((toc - tic) / (60 * 60))), math.floor(((toc - tic) % (60 * 60)) / 60),
    #                                 ((toc - tic) % 60)))
