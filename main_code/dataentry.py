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

    Attributes
    ---------
    fe: FeatureExtractor
        FeatureExtractor model used for feature extraction
    dataset: str
        name of the dataset the image belongs to
    img_path: str
        path to the image
    img_name: str
        name of the image inferred from img_path
    ground_truth_label: str
        label of the folder of the image inferred from img_path
    y: int
        label of the folder of the image encoded by the LabelEncoder
    feature_file: str
        path of the feature embedding

    Methods
    ---------
    initiate_feature_embedding()
        function for extracting the Feature Vector of a single image for the first time
    image_numpy(self, img_size: int = None, mode='L')
        Function for loading and converting a single image into a numpy array. If img_size is given it will be resized.
    dataentry_to_nparray()
        transforms image to numpy array according to the FeatureExtractor
    feature_embedding()
        Lazy function for extracting or loading the Feature Vector of a single image
    __compute_image()
        Lazy function for loading and pre-processing a single image
    """
    img_path: str
    feature_file: str
    ground_truth_label: str

    def __init__(self, fe, dataset: str, img_path: str):
        """Initialise DataEntry object, preferably via DataSet initialisation with List Comprehension of DataEntry objects. 

        :param fe: FeatureExtractor model used for feature extraction (Note: A name view Keras Model is mandatory)
        :type fe: FeatureExtractor
        :param dataset: Name of the `DataSet` in order to refer each DataEntry object to it.
        :type dataset: str
        :param img_path: Local path of the image file (including file name)
        :type img_path: str
        """

        # extract all avaiable folders (classes) of the selected dataset
        # and the corresponding of the selected image path
        DIR_TRAIN_DATA = os.path.join(DATA_DIR, dataset, 'train')
        avaiable_classes = [item for item in os.listdir(DIR_TRAIN_DATA) if os.path.isdir(os.path.join(DIR_TRAIN_DATA,item))]
        folder_name = ntpath.basename(ntpath.dirname(img_path))

        # entire path to the image
        self.img_path = img_path 
        # image file name
        self.img_name = ntpath.basename(img_path)
        # class of the image referring to the folder name
        self.ground_truth_label = folder_name 
        # extract index of folder name/class -> int value for the class
        self.y = avaiable_classes.index(folder_name) 
        self.fe = fe  # selected FeatureExtractor

        # construct PATH where the feature vector is saved
        DIR_FEATURE_EMBEDDING_DATASET = os.path.join(DIR_FEATURES, fe.fe_model.name, dataset)
        # create the folder for feature vectors if it is not created yet
        if not os.path.exists(DIR_FEATURE_EMBEDDING_DATASET):
            os.makedirs(DIR_FEATURE_EMBEDDING_DATASET)

        pre, _ = os.path.splitext(os.path.join(DIR_FEATURE_EMBEDDING_DATASET, ntpath.basename(img_path)))
        #entire path to the feature vector of the images
        self.feature_file = pre + '.npy'

    def initiate_feature_embedding(self):
        """ function for extracting the Feature Vector of a single image for the first time

        :return: *self* (`numpy.ndarray`) - One-dimensional feature vector
        """
        if os.path.exists(self.feature_file):
            # check is implemented whether there are pictures with the same names
            print("WARNING: feature file of", self.img_path, " already existed.")
            # return np.load(self.feature_file, allow_pickle=True)
        else:
            _, x = self.fe.load_preprocess_img(self.img_path)
            feat = self.fe.extract_features(x)
            np.save(self.feature_file, feat)
            # print(self.feature_file)
            # return feat


    @lazy
    def feature_embedding(self):
        """Lazy function for extracting or loading the Feature Vector of a single image

        :return: *self* (`numpy.ndarray`) - One-dimensional feature vector
        """
        if os.path.exists(self.feature_file):
            return np.load(self.feature_file, allow_pickle=True)
        else:
            _, x = self.fe.load_preprocess_img(self.img_path)
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

    def image_numpy(self, img_size: int = None, mode='L', lrp: bool = False):
        """Function for loading and converting a single image into a numpy array. If img_size is given it will be resized.

        :param img_size: Select a image size (commonly used `128`), defaults to None
        :type img_size: int, optional
        :param mode: Color mode of the output image. use 'L' for grayscalale and 'RGB' for RGB, defaults to 'L'
        :param lrp: bool, if True, path for lrp heatmap is generated and this image is converted to numpy array

        :return: *self* (`numpy.ndarray`) - Normalized image as numpy array
        """

        # TODO check occurrences, cuz might be same as fe.load_preprocess_img(self, path)
        # part answer: not the same since axes are not expanded
        if lrp:
            datasetname = str.split(self.img_path, "/")[-4]
            if str.split(self.img_path, "/")[-3] == "test":
                heatmap_directory = os.path.join(STATIC_DIR, 'heatmaps', self.fe.fe_model.name, datasetname, "test",
                                                 self.ground_truth_label,
                                                 os.path.splitext(self.img_name)[0] + "_heatmap.png")
            else:
                heatmap_directory = os.path.join(STATIC_DIR, 'heatmaps', self.fe.fe_model.name, datasetname,
                                                 self.ground_truth_label,
                                                 os.path.splitext(self.img_name)[0] + "_heatmap.png")
            x = Image.open(heatmap_directory).convert(mode)
        else:
            x = Image.open(self.img_path).convert(mode)
        if img_size is not None:
            x = x.resize((img_size, img_size))
        x = np.array(x, dtype=np.float64)
        # Normalize to [0,1]
        x /= 255.
        # print(np.shape(x))
        return x

    def dataentry_to_nparray(self):
        """ transforms image to numpy array according to the FeatureExtractor

        :return: numpy array
        """
        _, x = self.fe.load_preprocess_img(self.img_path)
        return np.squeeze(x, axis=0)
       

def code_from_dataentry(dataset, suffix_path='', feature_embeddings_to_initiate='current', type_of_model="cnn"):
    """

    :param dataset: name of dataset: str
    :param suffix_path: str, suffix path under which the model for the feature extractor is saved
    :param feature_embeddings_to_initiate: str, shall the current FE be used or another one (like VGG16)
    :param type_of_model: str, vgg or cnn
    """
    from modelsetup import load_model_from_folder, get_output_layer
    from feature_extractor import FeatureExtractor

    # CODE FROM DATAENTRY.PY
    print("----- CREATION OF FEATURE EMBEDDINGS -----")
    new_embedding = True

    while new_embedding:
        tic = time.time()
        # set feature extractor
        if feature_embeddings_to_initiate == "current":
            # gets FE from a loaded CNN with the dataset name and a suffix
            sel_model = load_model_from_folder(dataset, suffix_path=suffix_path)
            fe = FeatureExtractor(loaded_model=sel_model,
                                  options_cnn=True if type_of_model == "cnn" else False,
                                  feature_model_output_layer=get_output_layer(sel_model, type_of_model))
        else:
            # Standard FE for general model:
            fe = FeatureExtractor()  # loaded_model=VGG16(weights='imagenet', include_top=True)
        print("Initiating ", fe.fe_model.name)

        # get all data entries
        image_path = os.path.join(DATA_DIR, dataset)
        data = [DataEntry(fe, dataset, os.path.join(path, file)) for path, _, files in os.walk(image_path) for file in
                files if file != ".DS_Store"]

        # create feature embeddings for all data entries
        for count, d in enumerate(data):
            d.initiate_feature_embedding()
            if count % 1000 == 0:
                print(count, " feature embeddings created")
        print("")

        toc = time.time()
        print("Creating Feature Embeddings needed: ",
              "{}h {}min {}sec ".format(math.floor(((toc - tic) / (60 * 60))), math.floor(((toc - tic) % (60 * 60)) / 60),
                                        ((toc - tic) % 60)))

        a = input("Do you want to create the feature embeddings for the general VGG16? [y/n]")
        if a == "n":
            new_embedding = False
        else:
            feature_embeddings_to_initiate = "VGG16"


if __name__ == '__main__':
    # run this to generate all feature embeddings
    # needs a trained model for the feature embeddings else VGG16 is used
    dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information. ")
    while dataset_to_use == "help":
        print("We need the folder name of a data set that is saved in your DATA_DIR. Usually that would be"
              "one of the names you specified in the DATA_DIR_FOLDERS list. e.g. 'mnist'")
        dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information. ")
    suffix_path = input("What is the suffix of your model? Type a string. e.g. '_testcnn' ")
    code_from_dataentry(dataset_to_use, suffix_path, feature_embeddings_to_initiate="vgg16")
