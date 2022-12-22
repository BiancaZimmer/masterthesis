import math
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics.pairwise import rbf_kernel as rbf

from tensorflow.keras.models import load_model

from sklearn_extra.cluster import KMedoids

from feature_extractor import FeatureExtractor
from dataentry import DataEntry
from mmd_critic import select_prototypes, compute_rbf_kernel

from classify import *
from dataset import *
from kernels import * 
from dataentry import *

import json

import warnings
# this ignores the warnings induced by the numpy version > 1.19
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

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


class PrototypesSelector:
    """Main prototype selector class from which the respective prototype_selector classes inherit
    """
    def __init__(self, dataset,
                 num_prototypes: int = 3,
                 use_image_embeddings: bool = True,
                 use_lrp: bool = False,
                 make_plots: bool = True,
                 verbose: int = 0,
                 sel_size = -1):
        """Initalizes super-class PrototypeSelector; infers FeatureExtractor and DataSet from input parameters

        :param dataset: An instance of type DataSet with the data to use
        :param num_prototypes: number of prototypes which shall be generated
        :param use_image_embeddings: bool, if True prototypes are calculated on basis of the feature embeddings
        :param use_lrp: bool, if True prototypes are calculated on basis of the LRP heatmaps
        :param make_plots: bool, if True plots are generated
        :param verbose: 0, 1, 2 depending on the verbose output; 0 == no output
        """

        # only one of feature embeddings, lrp heatmaps or raw data can be used as reference point
        self.use_image_embeddings = use_image_embeddings
        self.use_lrp = use_lrp if not self.use_image_embeddings else False
        self.use_raw = False if any([self.use_image_embeddings, self.use_lrp]) else True

        self.num_prototypes = num_prototypes
        self.make_plots = make_plots
        self.verbose = verbose

        self.prototypes_per_class = None
        self.mmd2_per_class = None

        self.dataset = dataset
        self.available_classes = self.dataset.available_classes
        self.sel_size = sel_size

    def plot_prototypes(self):
        """Plot selected prototypes for each available  class.
        """
        assert self.prototypes_per_class is not None, "No Prototypes selected yet! Please, fit the Selector first!"

        for available_class in self.available_classes:
            proto = self.prototypes_per_class[available_class]
            
            num_cols = 8
            num_rows = math.ceil(self.num_prototypes / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, num_rows * 3.0))
            for i, axis in enumerate(axes.ravel()):
                if i >= self.num_prototypes:
                    axis.axis('off')
                    continue
                # axis.imshow(prototypes_sorted[i].reshape(resize_size,resize_size,3))
                # axis.imshow(prototypes_sorted[i].reshape(resize_size,resize_size), cmap='gray')
                axis.imshow(Image.open([item.img_path for _, item in enumerate(proto)][i]), cmap='gray')
                axis.axis('off')

            fig.suptitle(f'{self.num_prototypes} Prototypes - for class: {available_class}')
            plt.show()

    def get_prototypes(self):
        """Return selected prototypes for each available  class.

        :return: **prototypes_per_class** (`dict`) - Dictionary of prototype per class, where the keys refer to the class and values to the :py:class:`DataEntry` objects of the prototypes
        """
        assert self.prototypes_per_class is not None, "No Prototypes selected yet! Please, fit the Selector first!"

        return self.prototypes_per_class

    def get_prototypes_img_files(self):
        """Return the image file names of the selected prototypes for each available class. These are used as an indicator for loading the DataEntry object in the online phase (running xai demo).

        :return: **prototypes_per_class_img_files** (`dict`) - Dictionary of image file names of prototypes per class, where the keys refer to the class and values to the image file name of the prototypes
        """
        assert self.prototypes_per_class is not None, "No Prototypes selected yet! Please, fit the Selector first!"

        proto_classes = self.prototypes_per_class.keys()
        prototypes_per_class_img_files = {}
        for proto_class in proto_classes:
            prototypes_per_class_img_files[proto_class] = [entry.img_name for entry in self.prototypes_per_class[proto_class]]
            
        return prototypes_per_class_img_files

    def _calc_mmd2(self, gamma=None, unbiased: bool = False):
        """Calculate maximum mean discrepancy (MMD) based on Gaussian kernel function for keras models (theano or tensorflow backend) based on 'A kernel method for the two-sample-problem.' from 'Gretton, Arthur, et al. (2007)'.

        :param gamma: Kernel coefficient for RBF kernel, defaults to None
        :type gamma: float, optional
        :param unbiased: Set to false, if the biased MMD2 should be calculated, defaults to True
        :type unbiased: bool, optional
        :return: **mmd2_per_class** (`dict`) - Dictionary of MMD2 values on the selected prototypes per class, where the keys refer to the class and values to the MMD2 value
        """
        assert self.prototypes_per_class is not None, "No Prototypes selected yet! Please, fit the Selector first!"

        eval_classes = self.prototypes_per_class.keys()
        mmd2_per_class = {}

        for available_class in eval_classes:
            # print("... Calculate MMD for class: ", available_class)

            if self.use_image_embeddings:
                X1 = np.array([item.feature_embedding.flatten().astype(float) for index, item in enumerate(list(filter(lambda x: x.ground_truth_label == available_class, self.dataset.data)))])
                X2 = np.array([item.feature_embedding.flatten().astype(float) for index, item in enumerate(self.prototypes_per_class[available_class])])

            else:
                X1 = np.array([item.image_numpy(img_size=self.sel_size).flatten() for index, item in enumerate(list(filter(lambda x: x.ground_truth_label == available_class, self.dataset.data)))])
                X2 = np.array([item.image_numpy(img_size=self.sel_size).flatten() for index, item in enumerate(self.prototypes_per_class[available_class])])
            
            if gamma is None:
                gamma_mmd = default_gamma(X1)
            else:
                gamma_mmd = gamma
                print(f'Chosen gamma={gamma}')

            x1x1 = rbf(X1, X1, gamma_mmd)
            x1x2 = rbf(X1, X2, gamma_mmd)
            x2x2 = rbf(X2, X2, gamma_mmd)

            if unbiased:
                m = x1x1.shape[0]
                n = x2x2.shape[0]
                # assert n > 1, "Not possible to calculate! At least 2 prototypes are needed for the calculation of the unbiased MMD!"

                # set mmd2 to 1 for one selected prototype, since no unbiased mmd2 can be against one single data point
                if n == 1:
                    mmd2 = None
                else:
                    mmd2 = ((x1x1.sum() - m) / (m * (m - 1))
                        + (x2x2.sum() - n) / (n * (n - 1))
                        - 2 * x1x2.mean())
            else:
                mmd2 = x1x1.mean() + x2x2.mean() - 2 * x1x2.mean()
            
            mmd2_per_class[available_class] = mmd2

        return mmd2_per_class

    def _test_1NN(self):
        """Implementation of the 1-NN classifier which is run over the range of the selected number of prototypes and returns a list for each metric.

        :param verbose: Set to `1` to print accuracy and recall, or even set to `2` to get further classifcation results, defaults to 0
        :type verbose: int, optional
        :return: 
            - list_errorm (`list`) - List of error rates 
            - list_accuracym (`list`) - List of accuracy
            - list_recallm (`list`) - List of recall
        """

        assert self.prototypes_per_class is not None, "No Prototypes selected yet! Please, fit the Selector first!"

        test_num_prototypes = list(range(1, self.num_prototypes+1))

        eval_classes = self.prototypes_per_class.keys()

        if self.use_image_embeddings:
            X_train = np.array([item.feature_embedding.flatten().astype(float) for available_class in eval_classes
                                for item in self.prototypes_per_class[available_class]])
            X_test = np.array([item.feature_embedding.flatten().astype(float) for item in self.dataset.data_t])
        else:
            X_train = np.array([item.image_numpy(img_size=self.sel_size).flatten() for available_class in eval_classes
                                for item in self.prototypes_per_class[available_class]])
            X_test = np.array([item.image_numpy(img_size=self.sel_size).flatten() for item in self.dataset.data_t])

        y_train = np.array([item.y for available_class in eval_classes
                            for item in self.prototypes_per_class[available_class]])
        y_test = np.array([item.y for item in self.dataset.data_t])

        # print("len of X_train:" , np.shape(X_train))
        # print("len of y_train:" , np.shape(y_train))
        # print("len of X_test:" , np.shape(X_test))
        # print("len of y_test:" , np.shape(y_test))

        list_errorm = []
        list_accuracym = []
        list_recallm = []

        for testm in test_num_prototypes:
            classifier = Classifier()
            classifier.build_model(X_train[list(range(0,testm)) + list(range(self.num_prototypes, self.num_prototypes+testm)), :],
                                   y_train[list(range(0, testm)) + list(range(self.num_prototypes,self.num_prototypes+testm))],
                                   self.verbose)
            accuracym, errorm, recallm = classifier.classify(X_test, y_test, self.verbose)

            list_errorm.append(errorm)
            list_accuracym.append(accuracym)
            list_recallm.append(recallm)

            if self.verbose > 1 and testm == self.num_prototypes:
                print('########################## RESULTS for 【 m=%d 】Prototype ############# '% (testm))
                if self.verbose > 2:
                    print('Number of used prototype per class: ', testm)
                    print("Classified data points: ", len(y_test))
                print(f"Accuracys for [1 to m={self.num_prototypes}] selected prototypes: {list_accuracym}")
                print(f"Recalls for [1 to m={self.num_prototypes}] and prototypes: {list_recallm}")
                print('####################################################################### ')

        return list_errorm, list_accuracym, list_recallm  

    def score(self, X=None, y=None, sel_metric: str = 'recall'):
        """Run 1-NN classifier (over the range of the selected number of prototypes) and calculate the metrics, that is also used for optimizing in GridSearchCV.

        :param X: always None, not used, needs to be there in order for GridSearchcv to work
        :param y: always None, not used, needs to be there in order for GridSearchcv to work
        :param sel_metric: Select a metric among `accuracy`, `error` or `recall`, defaults to recall
        :type sel_metric: str, optional
        :return: *self* (`dict`) - Return the value of selected metric.
        """

        errors, accuracys, recalls = self._test_1NN()
        error = errors[-1]
        accuracy = accuracys[-1]
        recall = np.mean(recalls[-1])
        
        self.mmd2_per_class = self._calc_mmd2()
        
        try:
            global mmd2_tracking
            mmd2_tracking[self.num_prototypes] = self.mmd2_per_class
        except NameError:
            print("No global mmd2_tracking variable found")
                
        if self.verbose > 1:
            print('################## RESULTS | Metric to optimize: '+ str(sel_metric) + ' ###############')
            print("Overall Error: ", error)
            print("Overall Accuracy: ", accuracy)
            print("Overall Recall: ", recall)
            print('MMD2 per class: ', self.mmd2_per_class)
            print('#######################################################################\n\n')

        if sel_metric == 'recall':
            return recall
        elif sel_metric == 'accuracy':
            return accuracy
        elif sel_metric == 'error':
            return error
        else:
            print("Select either 'recall', 'accuracy' or 'error' as metric!")

    def save_prototypes(self):
        if self.use_image_embeddings:
            DIR_PROTOTYPES_DATASET = os.path.join(MAIN_DIR, 'static/prototypes', self.dataset.fe.fe_model.name, self.dataset.name)
        else:
            DIR_PROTOTYPES_DATASET = os.path.join(MAIN_DIR, 'static/prototypes', "rawData", self.dataset.name)

        if not os.path.exists(DIR_PROTOTYPES_DATASET):
            os.makedirs(DIR_PROTOTYPES_DATASET)

        protos_file = os.path.join(DIR_PROTOTYPES_DATASET, str(self.num_prototypes) + '.json')

        protos_img_files = self.get_prototypes_img_files()

        if os.path.exists(protos_file):
            print('[!!!] A file already exists! Please delete this file to save again prototypes of these settings.')
        else:
            print(protos_file)
            # np.save(protos_file, protos_img_files)
            print('Saving ...')
            with open(protos_file, 'w') as fp:
                json.dump(protos_img_files, fp)


class PrototypesSelector_KMedoids(BaseEstimator, PrototypesSelector):
    """K-Medoids based Prototype Selection
    """

    def __init__(self, *args, **kwds):
        """Initialize k-Medoids prototype selector
        """
        super().__init__(*args, **kwds)

    def fit(self, data=None):
        """Run the prototype selection using the k-Medoids algorithm. The prototypes are stored in a dict **prototypes_per_class**.

        :param data: DataSet on which the selection as well as the GridSearchCV is run/fitted , defaults to None
        :type data: DataSet, optional
        """

        if data is None:
            data = self.dataset.data

        data_per_class = {}
        # data_t_per_class = {}
        for available_class in self.available_classes:
            data_per_class[available_class] = list(filter(lambda x: x.ground_truth_label == available_class, data))
            # data_t_per_class[available_class] = list(filter(lambda x: x.ground_truth_label == available_class, self.dataset.data_t))

        print("... selecting k-Medoids Prototypes")    
            
        if self.verbose > 0:
            print('======= Parameters =======')
            print(f'num_prototypes:{self.num_prototypes}')
            print(f'use_image_embeddings:{self.use_image_embeddings}')
            print(f'sel_size:{self.sel_size}')
            print('==========================\n')
            print('======== Classes =========')
            print(f'Prototype for:{self.available_classes}')
            print(f'Samples for classes:{[len(data_per_class[available_class]) for available_class in self.available_classes]}')
            print('==========================\n')

        self.prototypes_per_class = {}

        for available_class in self.available_classes:

            if self.use_image_embeddings:  # TODO maybe errors will rise here
                X = np.array([item.feature_embedding.flatten().astype(float) for index, item in enumerate(data_per_class[available_class])])
            else:
                X = np.array([item.image_numpy(img_size=self.sel_size).flatten() for index, item in enumerate(data_per_class[available_class])])

            kmedoids = KMedoids(n_clusters=self.num_prototypes, random_state=42, metric='euclidean', method='alternate').fit(X)

            # print(kmedoids.labels_)
            # print(kmedoids.cluster_centers_)
            # print(kmedoids.medoid_indices_)
            
            prototype_indices = kmedoids.medoid_indices_       
            prototypes = [data_per_class[available_class][idx] for idx in prototype_indices]        

            if self.verbose > 1: print('Indices: ', prototype_indices, '\nLabels: ', [item.ground_truth_label for _, item in enumerate(prototypes)])

            self.prototypes_per_class[available_class] = prototypes     

            # Visualize
            if self.make_plots:
                num_cols = 8
                num_rows = math.ceil(self.num_prototypes / num_cols)
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, num_rows * 3.0))
                for i, axis in enumerate(axes.ravel()):
                    if i >= self.num_prototypes:
                        axis.axis('off')
                        continue
                    # axis.imshow(prototypes_sorted[i].reshape(resize_size,resize_size,3))
                    # axis.imshow(prototypes_sorted[i].reshape(resize_size,resize_size), cmap='gray')
                    axis.imshow(Image.open([item.img_path for _, item in enumerate(prototypes)][i]), cmap='gray')
                    axis.axis('off')

                fig.suptitle(f'{self.num_prototypes} Prototypes - for class: {available_class}')
                plt.show()
            
        return self
    

class PrototypesSelector_MMD(BaseEstimator, PrototypesSelector):
    """MMD-based Prototype Selection by using parts of orignal implemention of *Been Kim*
    """

    def __init__(self, dataset, gamma: float = None, kernel_type: str = 'global',
                 num_prototypes: int = 3, use_image_embeddings: bool = True, use_lrp: bool = False,
                 make_plots: bool = True, verbose: int = 0, sel_size = -1):
        """Initialize MMD2 prototype selector

        :param gamma: Kernel coefficient for RBF kernel, defaults to None, defaults to None
        :type gamma: float, optional
        :param kernel_type: type of Kernel, currently only "global" is supported
        :type kernel_type: str, optional
        --- inherited from PrototypeSelector() ---
        :param num_prototypes: number of prototypes which shall be generated
        :param use_image_embeddings: bool, if True prototypes are calculated on basis of the feature embeddings
        :param use_lrp: bool, if True prototypes are calculated on basis of the LRP heatmaps
        :param make_plots: bool, if True plots are generated
        :param verbose: 0, 1, 2 depending on the verbose output; 0 == no output
        """
        self.gamma = gamma
        self.kernel_type = kernel_type
        super().__init__(dataset=dataset, num_prototypes = num_prototypes, use_image_embeddings = use_image_embeddings,
                         use_lrp = use_lrp, make_plots = make_plots, verbose = verbose, sel_size = sel_size)
        # no *args **kwds here because we can not use it with gridsearchcv then :/

    def fit(self, data=None):
        """Run the prototype selection using the MMD2 algorithm. The prototypes are stored in a dict **prototypes_per_class**.

        :param data: DataSet on which the selection as well as the GridSearchCV is run/fitted , defaults to None
        :type data: DataSet, optional
        :raises KeyError: The kernel_type must be either `global` or `local`, defaults to local

        """
        if data is None:
            data = self.dataset.data

        data_per_class = {}
        # data_t_per_class = {}
        for available_class in self.available_classes:
            data_per_class[available_class] = list(filter(lambda x: x.ground_truth_label == available_class, data))
            # data_t_per_class[available_class] = list(filter(lambda x: x.ground_truth_label == available_class, self.dataset.data_t))

        print("... selecting MMD Prototypes")    

        if self.verbose > 0:
            print('======= Parameters =======')
            print(f'num_prototypes:{self.num_prototypes}')
            print(f'gamma:{self.gamma}')
            print(f'use_image_embeddings:{self.use_image_embeddings}')
        if self.verbose > 1:
            print('======= Parameters =======')
            print(f'num_prototypes:{self.num_prototypes}')
            print(f'gamma:{self.gamma}')
            print(f'kernel_type:{self.kernel_type}')
            print(f'use_image_embeddings:{self.use_image_embeddings}')
            print(f'sel_size:{self.sel_size}')
            print('==========================\n')
            print('======== Classes =========')
            print(f'Prototype for:{self.available_classes}')
            print(f'Samples for classes:{[len(data_per_class[available_class]) for available_class in self.available_classes]}')
            print('==========================\n')

        self.prototypes_per_class = {}

        # find prototypes for every class
        for available_class in self.available_classes:

            # get feature embeddings in an array
            if self.use_image_embeddings:  # TODO errors may arise
                X = np.array([item.feature_embedding.flatten().astype(float)
                              for index, item in enumerate(data_per_class[available_class])])
            else:
                X = np.array([item.image_numpy(img_size=self.sel_size).flatten()
                              for index, item in enumerate(data_per_class[available_class])])

            # compute Kernel
            if self.gamma is None: 
                self.gamma = default_gamma(X)
                print(f'Setting default gamma={self.gamma} since no gamma value specified')

            # ToDo local-Kernel
            # In original setting of Been Kim, a local kernel in respect to the classes is also applied, which is not
            # feasible here since near hits and misses already show class relation.
            # However, this could be used for further exploration, maybe sub-classes.
            if self.kernel_type == 'global':
                K = compute_rbf_kernel(X, self.gamma)
            else:
                raise KeyError('kernel_type must be either "global" or "local"')
              
            if self.verbose >= 2:
                print('Shape of X: ', np.shape(X), "\nKernel Shape:", np.shape(K))

            # select Prototypes
            if self.num_prototypes > 0:
                prototype_indices = select_prototypes(K, self.num_prototypes)
                prototypes = [data_per_class[available_class][idx] for idx in prototype_indices]        

                if self.verbose > 1:
                    print('Indices: ', prototype_indices, '\nLabels: ', [item.ground_truth_label for _, item in enumerate(prototypes)])

                self.prototypes_per_class[available_class] = prototypes     

                # Visualize
                if self.make_plots:
                    num_cols = 8
                    num_rows = math.ceil(self.num_prototypes / num_cols)
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, num_rows * 3.0))
                    for i, axis in enumerate(axes.ravel()):
                        if i >= self.num_prototypes:
                            axis.axis('off')
                            continue
                        # axis.imshow(prototypes_sorted[i].reshape(resize_size,resize_size,3))
                        # axis.imshow(prototypes_sorted[i].reshape(resize_size,resize_size), cmap='gray')
                        axis.imshow(Image.open([item.img_path for _, item in enumerate(prototypes)][i]), cmap='gray')
                        axis.axis('off')

                    fig.suptitle(f'{self.num_prototypes} Prototypes - for class: {available_class}')
                    plt.show()

                    # plt.savefig(output_dir / f'{self.num_prototypes}_prototypes_imagenet_{class_name}.svg')

        return self


def scree_plot_MMD2(mmd2_tracking, available_classes, save_path='test'):

    assert mmd2_tracking is not None, "No MMD2 values were tracked. Please init the dict 'mmd2_tracking' for storing the values while selection!"

    print("=========== MMD2 values ===========")
    print(mmd2_tracking)

    for available_class in available_classes:
        mmd2_per_proto_and_class = []
        num_proto = []
        for k in mmd2_tracking.keys():
            mmd2_per_proto_and_class.append(mmd2_tracking[k][available_class])
            num_proto.append(k)

        fig1 = plt.figure()
        plt.plot(num_proto, mmd2_per_proto_and_class, 'x-', markersize=10, color='#1F497D', linewidth=3)
        plt.xlabel('Number of Prototypes', fontsize=12, fontweight = 'medium')
        plt.ylabel('MMD2 Scores', fontsize=12, fontweight = 'medium')
        plt.title(f'Class: {available_class} - The Scree-plot on MMD2', fontsize=14, fontweight = 'demibold')
        plt.tight_layout()
        fig1.savefig(save_path+"_"+str(available_class)+".png", bbox_inches='tight')
        # plt.show()

        # print(available_class,': -> ', num_proto, '=>>=', mmd2_per_proto_and_class)


def gridsearch_crossval_forMMD(dataset_to_use: str, suffix_path: str, type_of_model: str = 'vgg', make_plots: bool = False,
                               scree_params: dict = None, save_path = 'test'):

    if scree_params is None:
        scree_params = {"gamma": [None],
                        "use_image_embeddings": [False],
                        "use_lrp": [False],
                        "num_prototypes": list(range(1, 10))}

    if suffix_path == '':
        model_for_feature_embedding = None  # -> VGG16 is used -> untrained
    else:
        model_for_feature_embedding = load_model(os.path.join(STATIC_DIR, 'models', 'model_history_' + dataset_to_use +
                                                              str(suffix_path) + '.hdf5'))

    # Set up FeatureExtractor
    if type_of_model == "cnn":
        options_cnn = True
    else:
        options_cnn = False
    if model_for_feature_embedding is not None:
        feature_model_output_layer = get_output_layer(model_for_feature_embedding, type_of_model)
    else:
        feature_model_output_layer = None

    fe = FeatureExtractor(loaded_model=model_for_feature_embedding,
                          model_name=str.upper(type_of_model),
                          options_cnn=options_cnn,
                          feature_model_output_layer=feature_model_output_layer)

    # set up dataset for grid search
    dataset = DataSet(name=dataset_to_use, fe=fe)

    method = PrototypesSelector_MMD(dataset=dataset, make_plots = make_plots, verbose = 1)

    if type_of_model == "vgg":  # VGG16 will be used -> needs correct input shape; model_for_feature_embedding is None
        method.sel_size = 224
    else:
        method.sel_size = model_for_feature_embedding.input_shape[1]

    gs = GridSearchCV(method,
                      scree_params,
                      cv=[(slice(None), slice(None))],
                      verbose=1)

    # fitting of grid search plus timing
    tic = time.time()
    gs.fit(X = dataset.data)
    toc = time.time()
    print(
        "{}h {}min {}sec ".format(np.floor(((toc - tic) / (60 * 60))), np.floor(((toc - tic) % (60 * 60)) / 60),
                                  ((toc - tic) % 60)))

    try:
        global mmd2_tracking
        scree_plot_MMD2(mmd2_tracking, method.available_classes, save_path)
        mmd2_tracking = {}
    except NameError:
        print("No global mmd2_tracking variable found")


if __name__ == "__main__":
    # model_history_mnist_1247_cnn_seed3871
    # model_history_oct_cc_cnn_seed3871

    # global variable to save mmd2 values
    mmd2_tracking = {}

    # for the number of prototypes a scree plot must be drawn, since grid search will always use maximum num of prototypes
    # for gamma we can then conduct another search independently

    params = {"gamma": [None],
                    "use_image_embeddings": [False],
                    "use_lrp": [False],
                    "num_prototypes": list(range(1, 16))}

    gridsearch_crossval_forMMD(dataset_to_use="mnist_1247", suffix_path="_cnn_seed3871", type_of_model='cnn', scree_params=params, save_path="static/0000")
    gridsearch_crossval_forMMD(dataset_to_use="mnist_1247", suffix_path="", type_of_model='vgg', scree_params=params, save_path="static/0001")

    gridsearch_crossval_forMMD(dataset_to_use="oct_cc", suffix_path="_cnn_seed3871", type_of_model='cnn', scree_params=params, save_path="static/1000")
    gridsearch_crossval_forMMD(dataset_to_use="oct_cc", suffix_path="", type_of_model='vgg', scree_params=params, save_path="static/1001")

   # With FE
   #  params = {"gamma": [None],
   #                  "use_image_embeddings": [True],
   #                  "use_lrp": [False],
   #                  "num_prototypes": list(range(1, 16))}
   #
   #  gridsearch_crossval_forMMD(dataset_to_use="mnist_1247", suffix_path="_cnn_seed3871", type_of_model='cnn', scree_params=params)
   #  gridsearch_crossval_forMMD(dataset_to_use="mnist_1247", suffix_path="", type_of_model='vgg', scree_params=params)
   #
   #  gridsearch_crossval_forMMD(dataset_to_use="oct_cc", suffix_path="_cnn_seed3871", type_of_model='cnn', scree_params=params)
   #  gridsearch_crossval_forMMD(dataset_to_use="oct_cc", suffix_path="", type_of_model='vgg', scree_params=params)
   #
   #  # With LRP
   #  params = {"gamma": [None],
   #                  "use_image_embeddings": [False],
   #                  "use_lrp": [True],
   #                  "num_prototypes": list(range(1, 16))}
   #
   #  gridsearch_crossval_forMMD(dataset_to_use="mnist_1247", suffix_path="_cnn_seed3871", type_of_model='cnn', scree_params=params)
   #  gridsearch_crossval_forMMD(dataset_to_use="mnist_1247", suffix_path="", type_of_model='vgg', scree_params=params)
   #
   #  gridsearch_crossval_forMMD(dataset_to_use="oct_cc", suffix_path="_cnn_seed3871", type_of_model='cnn', scree_params=params)
   #  gridsearch_crossval_forMMD(dataset_to_use="oct_cc", suffix_path="", type_of_model='vgg', scree_params=params)


    # Find best gamma values


### Marvin's code:
    # -- Screeplot of MMD across num of proto  ---------------------------------------

    # scree_params = {"gamma": [None],
    #                 "use_image_embeddings": [False],
    #                 "num_prototypes": list(range(1,16))}

    # verbose_GSCV = 1

    # method = PrototypesSelector_MMD(dataset, verbose=1, make_plots=False)

    # gs = GridSearchCV(method, 
    #                 scree_params, 
    #                 cv=[(slice(None), slice(None))], 
    #                 verbose=verbose_GSCV)

    # gs.fit(dataset.data)

    # scree_plot_MMD2(mmd2_tracking, dataset.available_classes)

    # -- GridSearchCV to find gamma of RBF kernel -----------------------------------  

    # -- Params for GridSearchCV --------

    tuned_params = {"gamma": [ None, 1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
                    "use_image_embeddings": [True],
                    "num_prototypes": [3]}

    ## mnist - VGG16 - 2st
    # tuned_params = {"gamma": [ 1e-6, 2e-6, 4e-6, 6e-6, 8e-6, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4],
    #                 "use_image_embeddings": [True],
    #                 "num_prototypes": [3]}

    # ## mnist - SimpleCNN - 2st
    # tuned_params = {"gamma": [ 1e-1, 2e-1, 4e-1, 6e-1, 8e-1, 1, 2, 4, 6, 8, 10],
    #                 "use_image_embeddings": [True],
    #                 "num_prototypes": [3]}

    ## mnist - rawData - 2st
    # tuned_params = {"gamma": [ 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3],
    #                 "use_image_embeddings": [False],
    #                 "num_prototypes": [3]}

    # ----------------------------------

    # cv = True
    # verbose_GSCV = 1

    # method = PrototypesSelector_MMD(dataset, verbose=1, make_plots=False)

    # if cv:
    #     gs = GridSearchCV(method, 
    #                     tuned_params, 
    #                     cv=KFold(n_splits=5, shuffle=False), 
    #                     verbose=verbose_GSCV)
    # else:
    #     gs = GridSearchCV(method, 
    #                     tuned_params, 
    #                     cv=[(slice(None), slice(None))], 
    #                     verbose=verbose_GSCV)


    # gs.fit(dataset.data)

    # results = gs.cv_results_

    # # Plot GridSearchCV on gamma
    # plt.plot(tuned_params['gamma'], gs.cv_results_['mean_test_score'],'bx-', markersize=10, color='#1F497D', linewidth=3)
    # plt.xlabel('Gamma values', fontsize=12, fontweight = 'medium')
    # plt.xscale('log',base=10) 
    # plt.ylabel('Recall', fontsize=12, fontweight = 'medium')
    # if tuned_params['use_image_embeddings'][0]:
    #     plt.title(f'GridSearchCV - {fe.fe_model.name} - on {dataset_name} dataset', fontsize=14, fontweight = 'demibold')
    # else:
    #     plt.title(f'GridSearchCV - raw Data - on {dataset_name} dataset', fontsize=14, fontweight = 'demibold')
    # plt.show()

    # print("======= RESULTS")
    # print('Gamma values: {}'.format(tuned_params['gamma']))
    # print('Mean test score: {}'.format(gs.cv_results_['mean_test_score']))
    # print('MMD scores: {}'.format(mmd2_tracking))


    # -- Store prototypes locally  ---------------------------------------

    # gamma_vgg16_quality = 6e-06
    # gamma_simpleCNN_quality = 0.006
    # gamma_rawData_quality = 0.0004
    gamma_vgg16_mnist = 4e-05
    gamma_simpleCNN_mnist = 1
    gamma_rawData_mnist = 0.0001

    # Initialize Prototype Selector
    # tester = PrototypesSelector_MMD(dataset, num_prototypes=3, use_image_embeddings=True, gamma=gamma_simpleCNN_mnist,
    #                                 verbose=1, make_plots=True)
    #
    # tester.fit()
    # tester.score()
    # tester.save_prototypes()


    # -- Final run of both prototype selection algorihtms  ---------------------------------------

    # gamma_vgg16_quality = 6e-06
    # gamma_simpleCNN_quality = 0.006
    # gamma_rawData_quality = 0.0004
    # gamma_vgg16_mnist = 4e-05
    # gamma_simpleCNN_mnist = 1
    # gamma_rawData_mnist = 0.0001

    # tester_KMedoids = PrototypesSelector_KMedoids(dataset, num_prototypes=3, use_image_embeddings=True, verbose=1, make_plots=False)
    # tester_MMD = PrototypesSelector_MMD(dataset, num_prototypes=3, gamma=gamma_simpleCNN_quality, use_image_embeddings=True, verbose=1, make_plots=False)

    # tester_KMedoids.fit()
    # tester_KMedoids.score()

    # tester_MMD.fit()
    # tester_MMD.score()
