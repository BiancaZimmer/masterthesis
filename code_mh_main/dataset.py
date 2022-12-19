import os
import sys
import ntpath
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

from dataentry import *
from utils import *
from feature_extractor import *
from modelsetup import *
# from cnn_model import *

# Set seed
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(RANDOMSEED)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(RANDOMSEED)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(RANDOMSEED)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
from tensorflow.random import set_seed
set_seed(RANDOMSEED)


class DataSet():
    """DataSet object for the image dataset containing list of DataEntries

    Attributes
    ---------
    name: str
        name of dataset to use
    fe: FeatureExtractor
        FE to use for images
    path_datasets:str
        usually global variable DATA_DIR, this is where the data is
    DIR_TRAIN_DATA: str
        path to folder of train data
    DIR_VAL_DATA: str
        path to folder of validation data
    DIR_TEST_DATA: str
        path to folder of test data
    data: list of DataEntry
        list of DataEntries from the train data
    data_t: list of DataEntry
        list of DataEntries from the test data
    available_classes: list of str
        class labels inferred from DIR_TRAIN_DATA

    Methods
    ---------
    apply_elbow_method(self, use_image_embeddings:bool =True, sel_size:int = 128, components_PCA: int =None)
        Elbow method that is applied to the feature embedding or the raw data of this DataSet
    apply_sil_method(self, use_image_embeddings:bool =True, sel_size:int = 128, components_PCA: int =None)
        Silhouette method that is applied to the feature embedding or the raw data of this DataSet
    """
    def __init__(self, name: str, fe: FeatureExtractor, path_datasets=DATA_DIR):
        """Initialization of the DataSet

        :param name: Dataset name which should refer to the folder name
        :type name: str
        :param fe: FeatureExtractor model that is used for the feature embedding, e.g. VGG16 for model-agnostic feature extraction
        :type fe: FeatureExtractor
        :param path_datasets: Local path where the dataset(s) is stored, defaults to DATA_DIR
        :type path_datasets: str, optional
        """

        if fe is None:
            # Initialize Feature Extractor Instance 
            self.fe = FeatureExtractor(loaded_model=fe)
        else:
            self.fe = fe

        self.name = name

        self.path_datasets = path_datasets

        # Image path of the dataset
        self.DIR_TRAIN_DATA = os.path.join(self.path_datasets, self.name, 'train')
        self.DIR_VAL_DATA = os.path.join(self.path_datasets, self.name, 'val')
        self.DIR_TEST_DATA = os.path.join(self.path_datasets, self.name, 'test')

        self.data = [DataEntry(self.fe, self.name, os.path.join(path, file)) for path, _, files in
                     os.walk(self.DIR_TRAIN_DATA) for file in files if file.endswith(tuple(image_extensions))]
        self.data_t = [DataEntry(self.fe, self.name, os.path.join(path, file)) for path, _, files in
                       os.walk(self.DIR_TEST_DATA) for file in files if file.endswith(tuple(image_extensions))]

        self.available_classes = [item for item in os.listdir(self.DIR_TRAIN_DATA) if
                                  os.path.isdir(os.path.join(self.DIR_TRAIN_DATA, item))]

        print('\n==============')
        print(f'Current Dataset: {self.name}')
        print(f'Available Classes: {self.available_classes}')

        print(f'Length of Train Data: {len(self.data)}')
        num_val_data = len([path for path, _, files in os.walk(self.DIR_VAL_DATA)
                   for file in files if file.endswith(tuple(image_extensions))])
        print(f'Length of Validation Data: {num_val_data}')
        print(f'Length of Test Data: {len(self.data_t)}')
        print('==============\n')

    def apply_elbow_method(self, use_image_embeddings:bool =True, sel_size:int = 128, components_PCA: int =None):
        """Elbow method that is applied to the feature embedding or the raw data of this DataSet

        :param use_image_embeddings: Set to False if this Elbow method should be applied to the raw data, defaults to True
        :type use_image_embeddings: bool, optional
        :param sel_size: Image size which is utilized if this method is applied to the raw data, defaults to 128
        :type sel_size: int, optional
        :param components_PCA: Optional a PCA  can be performed in advance to reduce dimensions, defaults to None
        :type components_PCA: int, optional
        """
        data_per_class = {}

        for available_class in self.available_classes:
            distortions = []
            inertias = []

            K = range(1, 16)

            data_per_class[available_class] = list(filter(lambda x: x.ground_truth_label == available_class, self.data))

            if use_image_embeddings:
                X = np.array([item.feature_embedding.flatten().astype(float) for index, item in enumerate(data_per_class[available_class])])
                X_img = np.array([item.img_path for _, item in enumerate(data_per_class[available_class])])

            else:
                X = np.array([item.image_numpy(img_size=sel_size).flatten() for index, item in enumerate(data_per_class[available_class])])
                X_img = np.array([item.img_path for _, item in enumerate(data_per_class[available_class])])

            if components_PCA is not None:
                print(f"Components before PCA: {X.shape[1]}")
                print(f"-- Components PCA: {pca.n_components}")
                print(f"Components after PCA: {pca.n_components}")
                pca = PCA(n_components=100, random_state=22)
                pca.fit(X)
                X = pca.transform(X)

            for k in K:
                # Building and fitting the model
                kmeanModel = KMeans(n_clusters=k).fit(X)
                kmeanModel.fit(X)

                distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                                    'euclidean'), axis=1)) / X.shape[0])
                #inertias.append(kmeanModel.inertia_)

                if k < 0:
                    x_data = [i for i in range(X.shape[1])]
                    plt.title(f'{k} cluster fitted')
                    plt.scatter(x_data,kmeanModel.cluster_centers_[0], color = 'red',alpha=0.2,s=70)
                    plt.scatter(x_data,kmeanModel.cluster_centers_[1] , color = 'blue',alpha=0.2,s=50)
                    plt.show()

            plt.plot(K, distortions, 'bx-', markersize=10, color='#1F497D', linewidth=3)
            plt.xlabel('Number of Prototypes', fontsize=12, fontweight = 'medium')
            plt.ylabel('Distortion', fontsize=12, fontweight = 'medium')
            plt.title(f'Class: {available_class} - The Elbow Method using distortions', fontsize=14, fontweight = 'demibold')
            plt.show()

            # plt.plot(K, inertias, 'bx-', markersize=10, color='#1F497D', linewidth=3)
            # plt.xlabel('Number of Prototypes', fontsize=12, fontweight = 'medium')
            # plt.ylabel('Inertia', fontsize=12, fontweight = 'medium')
            # plt.title(f'Class: {available_class} - The Elbow Method using Inertia', fontsize=14, fontweight = 'demibold')
            # plt.show()

    def apply_sil_method(self, use_image_embeddings:bool =True, sel_size:int = 128, components_PCA: int =None):
        """Silhouette method that is applied to the feature embedding or the raw data of this DataSet

        :param use_image_embeddings: Set to False if this Silhouette method should be applied to the raw data, defaults to True
        :type use_image_embeddings: bool, optional
        :param sel_size: Image size which is utilized if this method is applied to the raw data, defaults to 128
        :type sel_size: int, optional
        :param components_PCA: Optional a PCA  can be performed in advance to reduce dimensions, defaults to None
        :type components_PCA: int, optional
        """
        data_per_class = {}

        for available_class in self.available_classes:
            sil = []

            K = range(2, 16)

            data_per_class[available_class] = list(filter(lambda x: x.ground_truth_label == available_class, self.data))

            if use_image_embeddings:
                X = np.array([item.feature_embedding.flatten().astype(float) for index, item in enumerate(data_per_class[available_class])])
                X_img = np.array([item.img_path for _, item in enumerate(data_per_class[available_class])])
            else:
                X = np.array([item.image_numpy(img_size=sel_size).flatten() for index, item in enumerate(data_per_class[available_class])])
                X_img = np.array([item.img_path for _, item in enumerate(data_per_class[available_class])])


            if components_PCA is not None:
                print(f"Components before PCA: {X.shape[1]}")
                print(f"-- Components PCA: {pca.n_components}")
                print(f"Components after PCA: {pca.n_components}")
                pca = PCA(n_components=100, random_state=22)
                pca.fit(X)
                X = pca.transform(X)

            sil_score_max =  -1

            for k in K:
                # Building and fitting the model
                kmeanModel = KMeans(n_clusters=k).fit(X)
                kmeanModel.fit(X)

                labels = kmeanModel.labels_
                sil_score =silhouette_score(X, labels, metric = 'euclidean')
                sil.append(sil_score)
                if sil_score > sil_score_max:
                    print("The average silhouette score for %i clusters is %0.2f" %(k,sil_score))
                    sil_score_max = sil_score
                    best_n_clusters = k

                if k < 0:
                    x_data = [i for i in range(X.shape[1])]
                    plt.title(f'{k} cluster fitted')
                    plt.scatter(x_data,kmeanModel.cluster_centers_[0], color = 'red',alpha=0.2,s=70)
                    plt.scatter(x_data,kmeanModel.cluster_centers_[1] , color = 'blue',alpha=0.2,s=50)
                    plt.show()


            print('Best number of Clusters by Silhouette:', best_n_clusters)

            plt.plot(K, sil, 'bx-', markersize=10, color='#1F497D', linewidth=3)
            plt.xlabel('Number of Prototypes', fontsize=12, fontweight = 'medium')
            plt.ylabel('Silhouette Scores', fontsize=12, fontweight = 'medium')
            plt.title(f'Class: {available_class} - The Silhouette Method', fontsize=14, fontweight = 'demibold')
            plt.show()

    def statistical_overview(self):
        # TODO: implement method to do some descriptive statistics
        pass


def get_available_dataset(path_datasets =DATA_DIR):
    """Function to get a list of local available datasets.

    :param path_datasets: Local path where the datasets are stored, defaults to DATA_DIR
    :type path_datasets: str, optional
    
    :return: *self* (`list`) - List of local available datasets
    """
    return next(os.walk(path_datasets))[1]


def get_available_featureembeddings(path_features =DIR_FEATURES):
    """Function to get a list of local available feature embeddings.

    :param path_features: Local path where the pre-computed feature embeddings are stored, defaults to DIR_FEATURES
    :type path_features: str, optional

    :return: *self* (`list`) - List of local available feature embeddings
    """
    return next(os.walk(path_features))[1]


def get_dict_datasets_with_all_embeddings():
    """Function to get a dictionary where the keys are the avaiable DataSets and their values another dictionary with the available feature embeddings.

    :return: *self* (`dict`) - Dictionary with all feature embeddings in respect to the avaiable datasets
    """
    from modelsetup import load_model_from_folder
    # LOAD the DATASETS
    if len(DATA_DIR_FOLDERS) > 0:
        dataset_list = DATA_DIR_FOLDERS
    else:
        dataset_list = get_available_dataset()
    dict_datasets_embeddings = {}
    for dataset_name in dataset_list:
        dict_embeddings = {}
            
        fe_CNNmodel = FeatureExtractor(loaded_model=load_model_from_folder(sel_dataset=dataset_name, suffix_path='_cnn'))
        dict_embeddings[fe_CNNmodel.fe_model.name] = DataSet(name = dataset_name, fe=fe_CNNmodel)
        fe_VGG16 = FeatureExtractor(loaded_model=None)
        dict_embeddings[fe_VGG16.fe_model.name] = DataSet(name = dataset_name, fe=fe_VGG16)

        dict_datasets_embeddings[dataset_name] = dict_embeddings

    print(f'Possible dataset: {dict_datasets_embeddings.keys()}')
    print(f'Possible embeddings: {dict_embeddings.keys()}')
    return dict_datasets_embeddings


def get_dict_datasets(use_CNN_feature_embeddding:bool, use_all_datasets: bool):
    """Function to get a dictionary with available datasets where the keys are the names and the values are the DataSet objects.

    :param use_CNN_feature_embeddding: Set to True in order to save the model-specific (CNN-based) feature embedding instead the model-agnostic feature embedding (VGG16)
    :type use_CNN_feature_embeddding: bool
    :param use_all_datasets: Set to True in order to walk through all data sets in the DATA_DIR; if set to false DATA_DIR_FOLDER is used
    :type bool

    :return: *self* (`dict`) - Dictionary with avaiable DataSets objects
    """
    # LOAD the DATASETS
    if use_all_datasets:
        dataset_list = get_available_dataset()
    else:
        dataset_list = DATA_DIR_FOLDERS
    dict_datasets = {}
    for dataset_name in dataset_list:
        if use_CNN_feature_embeddding:
            fe_CNNmodel = FeatureExtractor(loaded_model=load_model_from_folder(sel_dataset=dataset_name))
            dict_datasets[dataset_name] = DataSet(name = dataset_name, fe = fe_CNNmodel)
        else:
            fe_VGG16 = FeatureExtractor(loaded_model=None)
            dict_datasets[dataset_name] = DataSet(name = dataset_name, fe = fe_VGG16)
    print(f'Possible dataset: {dict_datasets.keys()}')
    return dict_datasets


def get_quality(fe =None):
    """Function to get the DataSet of quality data.

    :param fe: FeatureExtractor model that is used for the feature embedding, e.g. VGG16 for model-agnostic feature extraction
    :type fe: FeatureExtractor

    :return: *self* (`DataSet`) - DataSet object of the quality data.
    """
    return DataSet(name = 'quality', fe =fe) 


def get_mnist(fe =None):
    """Function to get the DataSet of MNIST data.

    :param fe: FeatureExtractor model that is used for the feature embedding, e.g. VGG16 for model-agnostic feature extraction
    :type fe: FeatureExtractor

    :return: *self* (`DataSet`) - DataSet object of the MNIST data.
    """

    return DataSet(name = 'mnist', fe =fe) 


if __name__ == "__main__":

    # -- Setup -------------------------------------------------------------------   

    # give name of data set folder
    dataset_name = DATA_DIR_FOLDERS[0]

    use_image_embeddings = True
    sel_size = 128

    ## Select feature Extractor
    ## Simple or MultiCNN (set BINARY to False)
    fe = FeatureExtractor(loaded_model=load_model_from_folder(dataset_name))
    ## VGG16 -> loaded_model = None
    # fe = FeatureExtractor(loaded_model=None)

    ## Load Dataset
    dataset = DataSet(name = dataset_name, fe =fe)

    # -- Apply elbow method ------------------------------------------------------

    # dataset.apply_elbow_method(use_image_embeddings=use_image_embeddings, components_PCA=None)
    # dataset.apply_sil_method(use_image_embeddings=use_image_embeddings, components_PCA=None)


    # -- Apply elbow method ------------------------------------------------------

    # print(dataset.data_t[0].img_name)
    # print(dataset.data_t[0].y)
    # print('numpy')
    # print(dataset.data_t[0].image_numpy())
    # print(np.shape(dataset.data_t[0].image_numpy()))
    # print(np.shape(dataset.data_t[0].image_numpy(img_size=sel_size).flatten()))


    # if use_image_embeddings:
    #     X_test = np.array([item.feature_embedding.flatten().astype(float) for item in dataset.data_t])
    # else:
    #     X_test = np.array([item.image_numpy(img_size=sel_size).flatten() for item in dataset.data_t])

    # y_test = np.array([item.y for item in dataset.data_t])

    # print('X_test')
    # print(X_test[0])
    # print(X_test[0].shape)

    # print(y_test[0])
    # print(y_test[0].shape)
