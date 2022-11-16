import math
import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

import ssim.ssimlib as pyssim
from skimage.metrics import structural_similarity as ssim

from feature_extractor import FeatureExtractor
from dataentry import DataEntry
from utils import *
from LRP_heatmaps import generate_LRP_heatmap, create_special_analyzer
from modelsetup import get_output_layer
from helpers import normalize_to_onesize

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


def calc_distances_scores_on_fe(class_data_entry_list, feature_vector, top_n: int = 5, dist: str = 'cosine',
                                return_data_entry_ranked: bool = False, plot_idx: bool = False):
    """ This function determines the top n closest DataEntries with respect to a given feature vector of another DataEntry (test input). Thereby, it iterate over
    the given list of DataEntries and calculate the distances (dissimiliarty) and rank them. 

    :param class_data_entry_list: List of DataEntries, note class reference/filter has to be done in advance if required.
    :type class_data_entry_list: list
    :param feature_vector: Feature embedding of a DataEntry (test input)
    :type feature_vector: numpy.ndarray
    :param top_n: Set an integer value how many nearest samples should be selected, defaults to 5
    :type top_n: int, optional
    :param dist: Distance applied in feature embedding, e.g. 'euclidean'/'cosine'/'manhattan', defaults to 'cosine'
    :type dist: str, optional
    :param return_data_entry_ranked: Set True in order to get a list of the DataEntries of the nearest samples, defaults to False
    :type return_data_entry_ranked: bool, optional
    :param plot_idx: Set to True in order to plot the indices of the nearest data samples, defaults to False
    :type plot_idx: bool, optional

    :return: 
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_dataentry** (`list`) - If 'return_data_entry_ranked' set to True, a list of the DataEntries of the nearest samples
    """
    
    # Load feature vector of data sample w.r.t. current class
    feature_embedding = [x.feature_embedding for x in class_data_entry_list]

    if dist == 'euclidean':
        # distances = np.linalg.norm(feature_embedding-feature_vector, axis=1)
        distances = np.array([distance.euclidean(feature_vector, feat) for feat in feature_embedding])
    elif dist == 'cosine':  # carful! 0 means close, 1 means far away
        distances = np.array([distance.cosine(feature_vector, feat) for feat in feature_embedding])
    elif dist == 'manhattan':
        distances = np.array([distance.cityblock(feature_vector, feat) for feat in feature_embedding])

    # Top distances
    if top_n < 1:
        idx_ranked = np.argsort(distances)
    else:
        idx_ranked = np.argsort(distances)[:top_n]
    if plot_idx:
        print("Index of top distances: ", idx_ranked)
        print("DISTANCES OVERVIEW")
        print(pd.Series(distances).describe())

    scores = distances[idx_ranked]
    
    if return_data_entry_ranked:
        return scores, [class_data_entry_list[i] for i in idx_ranked]
    else:
        return scores


def calc_distance_score_on_image(class_data_entry_list, test_data_entry, model, outputlabel, top_n: int = 5,
                                 dist: str = 'SSIM', image: bool = True,
                                 return_data_entry_ranked: bool = False, plot_idx: bool = False):
    """

    :param class_data_entry_list: List of DataEntries, note that class reference/filter has to be done in advance if required.
    :type class_data_entry_list: list
    :param test_data_entry: DataEntry of the image for which distances should be computed
    :param model: A trained model
    :type model: class ModelSetup
    :param outputlabel: the class/label/ouputneuron for which to calculate the difference of the heatmaps. Outputneuron equals the folder name
    :param top_n: Set an integer value how many nearest samples should be selected, defaults to 5
    :type top_n: int, optional
    :param dist: Distance applied to images, e.g. 'SSIM'/'CW-SSIM', defaults to 'SSIM'
    :type dist: str
    :param image: base the distance measurement on the lrp heatmaps (True) or on the raw images (False), defaults to True
    :type image: bool
    :param return_data_entry_ranked: Set True in order to get a list of the DataEntries of the nearest samples, defaults to False
    :type return_data_entry_ranked: bool, optional
    :param plot_idx: Set to True in order to plot the indices of the nearest data samples, defaults to False
    :type plot_idx: bool, optional

    :return:
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_dataentry** (`list`) - If 'return_data_entry_ranked' set to True, a list of the DataEntries of the nearest samples
    """

    def logtrafo(array):
        # trafo of [0, 255] into [0,2] - is redundant but better to understand later
        array = np.divide(array, 128)
        # ensure that you do not take log(0) in a later step
        array[array == 0] = np.min(array[array != 0])
        array[array == 2] = np.max(array[array != 2])
        # Trafo of [0, 2] into ]-inf, inf[
        arrt = np.full_like(array, 1)
        arrt[array <= 1] = np.log(array[array <= 1])
        arrt[array > 1] = -np.log(2 - array[array > 1])
        # trafo back to [0, 255]; values between -inf and 0 get transformed to 0;128 in the same measurement like 0;inf to 128;255
        # cv2.normalize(array[array <= 0], None, alpha=0, beta=128, norm_type=cv2.NORM_MINMAX)
        result = np.full_like(array, -1)
        result[arrt == 0] = 128
        try:
            min_ = np.min(arrt[arrt < 0])
        except ValueError:
            min_ = 0
        try:
            max_ = np.max(arrt[arrt > 0])
        except ValueError:
            max_ = 0

        factor = np.max([-min_, max_])
        try:
            result[arrt < 0] = ((arrt[arrt < 0]/factor)+1)*128
        except ValueError:
            pass  # no number below 0
        try:
            result[arrt > 0] = ((arrt[arrt > 0]/factor)*128)+127
        except ValueError:
            pass  # no number below 0
        return result

    def minmaxtrafo(array):
        array[array < 128] = 0
        array[array > 128] = 255
        return array

    def blurtrafo(array, sigma):
        return cv2.GaussianBlur(array, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

    def thresholdtrafo(array_ori):
        array = array_ori.copy()
        try:
            median_high = np.quantile(array[array > 128], 0.95)
            array[np.logical_and(array > 128, array <= median_high)] = 128
        except IndexError:
            pass
        try:
            median_low = np.quantile(array[array < 128], 0.05)
            array[np.logical_and(array < 128, array >= median_low)] = 128
        except IndexError:
            pass

        return array

    if image:
        dataset_to_use = str.split(class_data_entry_list[0].img_path, "/")[-4]
        heatmap_directory = os.path.join(STATIC_DIR, 'heatmaps', class_data_entry_list[0].fe.fe_model.name, dataset_to_use)
        image_names = [str.split(img.img_name, ".")[0] for img in class_data_entry_list]
        # outputlabel needs to be a string
        try:
            image_paths_list = [os.path.join(heatmap_directory, outputlabel, name+"_heatmap.png") for name in image_names]
        except TypeError:
            image_paths_list = [os.path.join(heatmap_directory, outputlabel[0], name+"_heatmap.png") for name in image_names]
        images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_paths_list]
        # is equal to:
        # images = [np.squeeze(img_to_array(load_img(img, color_mode='grayscale'))) for img in image_paths_list]
        # print(images[0].shape)

        test_image_name = str.split(test_data_entry.img_name, ".")[0]
        test_image_path = os.path.join(heatmap_directory, "test", outputlabel[0], test_image_name + "_heatmap.png")

        if not os.path.exists(test_image_path):  # heatmap not yet created -> create heatmap
            if not os.path.exists(os.path.join(heatmap_directory, "test", outputlabel[0])):
                os.makedirs(os.path.join(heatmap_directory, "test", outputlabel[0]))
            # calculate heatmap for testimage
            analyzer = create_special_analyzer(model.model, dataset_to_use)
            # outputlabel needs to be a string
            try:
                outputneuron = model.labelencoder.transform(outputlabel[0])
                img_test_heatmap = generate_LRP_heatmap(model.img_preprocess_for_prediction(test_data_entry), analyzer,
                                                        outputneuron)
            except ValueError:
                outputneuron = model.labelencoder.transform(outputlabel)
                img_test_heatmap = generate_LRP_heatmap(model.img_preprocess_for_prediction(test_data_entry), analyzer,
                                                        outputneuron)
            plt.imsave(test_image_path, img_test_heatmap)
        # print("saved ", outputneuron, "  label ", outputlabel)
        img_test = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    else:  # if image==False we evaluate on the raw images
        raw_image_paths_list = [img.img_path for img in class_data_entry_list]
        image_paths_list = raw_image_paths_list
        images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_paths_list]
        test_image_path = test_data_entry.img_path
        img_test = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        images, img_test = normalize_to_onesize(img_test, images)

    if dist == 'CW-SSIM':
        # FOR EXPERIMENTS ONLY!
        # Very slow algorithm - up to 50x times slower than SIFT or SSIM.
        # Optimization using CUDA or Cython code should be explored in the future.
        # value between 0;1 with 0 different and 1 same
        # -> trafo to 0;1 with 0 same and 1 different
        pil_test_image = Image.open(test_image_path)

        def calc_cw_ssim(image_path):
            pil = Image.open(image_path)
            result = pyssim.SSIM(pil).cw_ssim_value(pil_test_image)
            pil.close()
            result = 1-result
            return result
        distances = [calc_cw_ssim(img_path) for img_path in image_paths_list]
    elif dist == "SSIM":
        # Default SSIM implementation of Scikit-Image
        # value between -1; 1 ; -1 different, 1 the same
        # -> trafo to structural dissimilarity where value between 0;1 with 0 same and 1 different
        def dssim(img):
            ssim_index = ssim(img, img_test)
            result = (1-ssim_index)/2
            return result
        distances = [dssim(image) for image in images]
    elif dist == "SSIM-pushed":
        # Like SSIM but image values pushed to -1 and 1
        def dssim(img):
            # show some histograms
            # plt.subplot(2, 2, 1)
            # plt.hist(img_test.flatten())
            # plt.subplot(2, 2, 2)
            # plt.hist(img.flatten())
            # plt.subplot(2, 2, 3)
            # plt.hist(logtrafo(img_test).flatten())
            # plt.subplot(2, 2, 4)
            # plt.hist(logtrafo(img).flatten())
            # plt.show()
            ssim_index = ssim(logtrafo(img), logtrafo(img_test))
            result = (1-ssim_index)/2
            return result
        distances = [dssim(image) for image in images]

    elif dist == "SSIM-mm":
        # Like SSIM but image only contains 3 values: 0, 128, 255
        def dssim(img):
            ssim_index = ssim(minmaxtrafo(img), minmaxtrafo(img_test))
            result = (1-ssim_index)/2
            return result
        distances = [dssim(image) for image in images]

    elif dist == "SSIM-blur":
        # Like SSIM but image blurred
        def dssim(img):
            sigma = 3
            # show pictures
            # plt.subplot(2, 2, 1)
            # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # plt.subplot(2, 2, 2)
            # plt.imshow(img_test, cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # plt.subplot(2, 2, 3)
            # plt.imshow(logtrafo(blurtrafo(img, sigma)), cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # plt.subplot(2, 2, 4)
            # plt.imshow(logtrafo(blurtrafo(img_test, sigma)), cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # plt.show()

            ssim_index = ssim(blurtrafo(img, sigma), blurtrafo(img_test, sigma))
            result = (1-ssim_index)/2
            return result
        distances = [dssim(image) for image in images]
    elif dist == "SSIM-threshold":
        # Like SSIM but image blurred
        def dssim(img):
            # show pictures
            # plt.subplot(2, 2, 1)
            # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # plt.subplot(2, 2, 2)
            # plt.imshow(img_test, cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # plt.subplot(2, 2, 3)
            # plt.imshow(thresholdtrafo(img), cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # plt.subplot(2, 2, 4)
            # plt.imshow(thresholdtrafo(img_test), cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # plt.show()

            ssim_index = ssim(thresholdtrafo(img), thresholdtrafo(img_test))
            result = (1-ssim_index)/2
            return result
        distances = [dssim(image) for image in images]
    elif dist == "euclidean":
        distances = [distance.euclidean(image.flatten(), img_test.flatten()) for image in images]
    elif dist == 'cosine':  # careful! 0 means close, 1 means far away
        distances = [distance.cosine(image.flatten(), img_test.flatten()) for image in images]
    elif dist == 'manhattan':
        distances = [distance.cityblock(image.flatten(), img_test.flatten()) for image in images]

    distances = np.array(distances)
    # Top distances
    if top_n < 1:
        idx_ranked = np.argsort(distances)
    else:
        idx_ranked = np.argsort(distances)[:top_n]
    if plot_idx:
        print("Index of top distances: ", idx_ranked)
        print("DISTANCES OVERVIEW")
        print(pd.Series(distances).describe())

    scores = np.array(distances)[idx_ranked]

    if return_data_entry_ranked:
        return scores, [class_data_entry_list[i] for i in idx_ranked]
    else:
        return scores


def get_nearest_hits(test_dataentry, pred_label, data, fe, model=None, top_n: int = 5, distance_measure: str = 'cosine',
                     raw=False, distance_on_image=False):
    """Function to calculates the near hits in respect to a given test inputs sample (DataEntry).

    :param test_dataentry: DataEntry object (test input sample) on which near hits should be selected
    :type test_dataentry: DataEntry
    :param pred_label: Prediction label of the CNN classifier which should be also explained somewhat, which refers to the folder name (class)
    :type pred_label: str
    :param data: Data of DataSet (list of DataEntries)
    :type data: list
    :param fe: FeatureExtractor model that is used to extract the features of the test input sample
    :type fe: FeatureExtractor
    :param model: A trained model
    :param top_n: Set an integer value how many near hits should be selected, defaults to 5
    :type top_n: int, optional
    :param distance_measure: Distance applied in feature embedding, e.g. 'euclidean'/'cosine'/'manhattan', defaults to 'cosine'
    :type distance_measure: str, optional
    :param raw: Defines whether to use the raw image data (True) as comparison or Heatmap/FE
    :type raw: bool, optional
    :param distance_on_image: defines if distance should be measured on the image (True) or on the feature embedding (False)
    :type distance_on_image: bool, optional
    :return: 
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_nearest_hit_data_entry** (`list`) - List of the DataEntries of the near hits
    """

    hit_class_data_entry = list(filter(lambda x: x.ground_truth_label == pred_label, data))

    if distance_measure in ['SSIM', 'CW-SSIM'] or distance_on_image:
        scores_nearest_hit, ranked_nearest_hit_data_entry = calc_distance_score_on_image(hit_class_data_entry,
                                                                                         test_dataentry, model, pred_label,
                                                                                         top_n=top_n, dist=distance_measure,
                                                                                         return_data_entry_ranked=True,
                                                                                         image=not raw)
    else:
        _, x = fe.load_preprocess_img(test_dataentry.img_path)
        feature_vector = fe.extract_features(x)

        scores_nearest_hit, ranked_nearest_hit_data_entry = calc_distances_scores_on_fe(hit_class_data_entry, feature_vector,
                                                                                        top_n=top_n,
                                                                                        dist=distance_measure,
                                                                                        return_data_entry_ranked=True)
        # TODO add parameter for raw FE/image comparison

    return scores_nearest_hit, ranked_nearest_hit_data_entry


def get_nearest_miss(test_dataentry, pred_label, data, fe, model=None, top_n: int = 5, distance_measure: str = 'cosine',
                     raw=False, distance_on_image=False):
    """Function to calculates the near misses in respect to a given test inputs sample (DataEntry).

    :param test_dataentry: DataEntry object (test input sample) on which near misses should be selected
    :type test_dataentry: DataEntry
    :param pred_label: Prediction label of the CNN classifier which should be also explained somewhat, which refers to the folder name (class)
    :type pred_label: str
    :param data: Data of DataSet (list of DataEntries)
    :type data: list
    :param fe: FeatureExtractor model that is used to extract the features of the test input sample
    :type fe: FeatureExtractor
    :param model: A trained model
    :param top_n: Set an integer value how many near misses should be selected, defaults to 5
    :type top_n: int, optional
    :param distance_measure: Distance applied in feature embedding, e.g. 'euclidean'/'cosine'/'manhattan', defaults to 'cosine'
    :type distance_measure: str, optional
    :param raw: Defines whether to use the raw image data (True) as comparison or Heatmap/FE
    :type raw: bool, optional
    :param distance_on_image: defines if distance should be measured on the image (True) or on the feature embedding (False)
    :type distance_on_image: bool, optional
    :return: 
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_nearest_miss__data_entry** (`list`) - List of the DataEntries of the near misses
    """
    miss_class_data_entry = list(filter(lambda x: x.ground_truth_label != pred_label, data))

    if (distance_measure in ['SSIM', 'CW-SSIM']) or distance_on_image:
        scores_nearest_miss, ranked_nearest_miss__data_entry = calc_distance_score_on_image(miss_class_data_entry,
                                                                                            test_dataentry, model, pred_label,
                                                                                            top_n=top_n,
                                                                                            dist=distance_measure,
                                                                                            return_data_entry_ranked=True,
                                                                                            image=not raw)
    else:
        _, x = fe.load_preprocess_img(test_dataentry.img_path)
        feature_vector = fe.extract_features(x)

        scores_nearest_miss, ranked_nearest_miss__data_entry = calc_distances_scores_on_fe(miss_class_data_entry,
                                                                                           feature_vector, top_n=top_n,
                                                                                           dist=distance_measure,
                                                                                           return_data_entry_ranked=True)
        # TODO add parameter for raw FE/image comparison

    return scores_nearest_miss, ranked_nearest_miss__data_entry


def get_nearest_miss_multi(test_dataentry, classes, pred_label, data, fe, model=None, top_n: int = 5,
                           distance_measure: str = 'cosine', raw=False, distance_on_image=False):
    """Function to calculates the near misses for every class in respect to a given test inputs sample (DataEntry).

    :param test_dataentry: DataEntry object (test input sample) on which near misses should be selected
    :type test_dataentry: DataEntry
    :param classes: List of the available classes, can be retrieved from DataSet.available_classes
    :type classes: list
    :param pred_label: Prediction label of the CNN classifier which should be also explained somewhat, which refers to the folder name (class)
    :type pred_label: str
    :param data: Data of DataSet (list of DataEntries)
    :type data: list
    :param fe: FeatureExtractor model that is used to extract the features of the test input sample
    :type fe: FeatureExtractor
    :param model: A trained model
    :param top_n: Set an integer value how many near misses should be selected, defaults to 5
    :type top_n: int, optional
    :param distance_measure: Distance applied in feature embedding, e.g. 'euclidean'/'cosine'/'manhattan', defaults to 'cosine'
    :type distance_measure: str, optional
    :param raw: Defines whether to use the raw image data (True) as comparison or Heatmap/FE
    :type raw: bool, optional
    :param distance_on_image: defines if distance should be measured on the image (True) or on the feature embedding (False)
    :type distance_on_image: bool, optional
    :return:
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_nearest_miss__data_entry** (`list`) - List of the DataEntries of the near misses
    """

    available_classes = [c for c in classes if c != pred_label]

    scores = []
    data_entries = []
    # for distances based on LRP, go through miss classes and find closest heatmaps on pred_label output neuron!
    if distance_measure in ['SSIM', 'CW-SSIM'] or distance_on_image:
        # for miss_class in available_classes:
        #     miss_class_data_entry = list(filter(lambda x: x.ground_truth_label == miss_class, data))
        #     scores_nearest_miss, ranked_nearest_miss_data_entry = get_nearest_miss(test_dataentry, pred_label, miss_class_data_entry, fe,
        #                                                                            model, top_n, distance_measure)
        #     scores.append(scores_nearest_miss)
        #     data_entries.append(ranked_nearest_miss_data_entry)
        scores_nearest_miss, ranked_nearest_miss_data_entry = get_nearest_miss(test_dataentry, pred_label,
                                                                               data, fe, model, -1,
                                                                               distance_measure, raw, distance_on_image)
        for miss_class in available_classes:
            indices = [i for i, entry in enumerate(ranked_nearest_miss_data_entry) if entry.ground_truth_label == miss_class]
            indices = indices[:top_n]
            scores.append([scores_nearest_miss[i] for i in indices])
            data_entries.append([ranked_nearest_miss_data_entry[i] for i in indices])
    else:  # for distances based on FE, go through miss-classes and find the closest FE
        for miss_class in available_classes:
            scores_nearest_miss, ranked_nearest_miss_data_entry = get_nearest_hits(test_dataentry, miss_class, data,
                                                                                   fe, model, top_n,
                                                                                   distance_measure, raw=raw)
            scores.append(scores_nearest_miss)
            data_entries.append(ranked_nearest_miss_data_entry)

    return scores, data_entries


def plot_nmnh(dataentries, distance_scores: float, title: str = "Near Miss/Near Hit Plot"):

    figure = plt.figure(figsize=(20, 8))
    plt.suptitle(title)
    for dataentry, sim, i in zip([x for x in dataentries], distance_scores, range(len(distance_scores))):
        pic = cv2.imread(dataentry.img_path)
        plt.subplot(int(np.ceil(len(distance_scores) / 5)), 5, i + 1)
        plt.title(f"{dataentry.img_name}\n\
        Actual Label : {dataentry.ground_truth_label}\n\
        Distance : {'{:.3f}'.format(sim)}", weight='bold', size=12)

        plt.imshow(pic, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    # plt.show()
    return figure


def plot_nmnh_heatmaps(dataentries, distance_scores: float, outputlabel: str, title: str = "Near Miss/Near Hit Plot"):

    figure = plt.figure(figsize=(20, 8))
    plt.suptitle(title)

    dataset = str.split(dataentries[0].img_path, "/")[-4]
    heatmap_directory = os.path.join(STATIC_DIR, 'heatmaps', dataentries[0].fe.fe_model.name, dataset)

    for dataentry, dist, i in zip([x for x in dataentries], distance_scores, range(len(distance_scores))):
        name = str.split(dataentry.img_name, ".")[0]
        image_path = os.path.join(heatmap_directory, outputlabel, name + "_heatmap.png")
        pic = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(int(np.ceil(len(distance_scores) / 5)), 5, i + 1)
        plt.title(f"{dataentry.img_name}\n\
        Actual Label : {dataentry.ground_truth_label}\n\
        Distance : {'{:.3f}'.format(dist)}", weight='bold', size=12)

        plt.imshow(pic, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

    plt.tight_layout()
    # plt.show()
    return figure


def get_nhnm_overview(dataset, suffix_path="_multicnn", type_of_model="cnn", distance_measure='cosine',
                      top_n=TOP_N_NMNH, use_prediction=True, raw=False, distance_on_image=False):
    # top_n = TOP_N_NMNH   # number of near hits/misses to show
    # use_prediction = True  # if set to true a prediction of the image is used for the near hits/misses
    # suffix_path = "_multicnn"   # if use_prediction=True then you have to specify which model of the dataset to use
    # distance_measure = 'cosine'  # distance measure for near miss/near hit

    from dataset import DataSet
    from modelsetup import ModelSetup, load_model_from_folder

    tic = time.time()

    # Initialize Feature Extractor Instance
    options_cnn = True if type_of_model == "cnn" else False
    tmp_model = load_model_from_folder(dataset, suffix_path=suffix_path)
    fe = FeatureExtractor(loaded_model=tmp_model,
                          model_name=str.upper(type_of_model),
                          options_cnn=options_cnn,
                          feature_model_output_layer=get_output_layer(tmp_model, type_of_model))
    data = DataSet(dataset, fe)

    # if distance measure requires feature embedding check whether all are created and if not -> create them
    if not distance_on_image:
        i = 0   # Counter for newly loaded feature embeddings
        for d in data.data:
            if os.path.exists(d.feature_file):
                np.load(d.feature_file, allow_pickle=True)
                pass
            else:  # if feature embedding doesn't exist yet, it is extracted now and saved
                _, x = d.fe.load_preprocess_img(d.img_file)
                feat = d.fe.extract_features(x)
                np.save(d.feature_file, feat)
                print("SAVE...")
                i += 1

        print("... newly loaded feature embeddings, which were not considered yet : ", i)

    # initialize model
    if type_of_model == "vgg":  # VGG16 will be used -> needs correct input shape # model_for_feature_embedding is None and
        sel_model = ModelSetup(data, sel_size=224)
    else:
        sel_model = ModelSetup(data)
    sel_model._preprocess_img_gen()
    sel_model.set_model(suffix_path=suffix_path)

    if type_of_model == 'cnn':
        sel_model.mode_rgb = False
    else:
        sel_model.mode_rgb = True

    another_image = "y"
    while another_image == "y":  # possibility to get another overview of a random image until user input decides otherwise
        ###### ==== Select a Random Image as an input image ==== ######
        # CAREFUL: As long as randomness is controlled by the seed you will always get the same image here
        rnd_class = random.choice(data.available_classes)
        rnd_img_path = os.path.join(DATA_DIR, dataset, 'test', rnd_class)
        rnd_img_file = random.choice(os.listdir(rnd_img_path))
        # Use for fixed image:
        # rnd_img_path = os.path.join(DATA_DIR, dataset, 'test', '6')
        # rnd_img_file = '5481.jpg'
        print("Random image:", rnd_img_file, " in folder:")
        print(rnd_img_path)

        rnd_img = DataEntry(fe, dataset, os.path.join(rnd_img_path, rnd_img_file))

        img, x = fe.load_preprocess_img(rnd_img.img_path)
        feature_vector = fe.extract_features(x)

        pred_label = rnd_img.ground_truth_label

        if use_prediction:
            pred_label, pred_prob = sel_model.pred_test_img(rnd_img)
            print("Ground Truth: ", rnd_img.ground_truth_label)
            print("Prediction: ", pred_label)
            print("Probability: ", pred_prob)

        hit_class_idx = []
        miss_class_idx = []

        for f in data.data:
            if f.ground_truth_label == rnd_img.ground_truth_label:
                hit_class_idx.append(f)
            else:
                miss_class_idx.append(f)

        print("Number of hits: ", np.size(hit_class_idx))
        print("Number of misses: ", np.size(miss_class_idx))

        # NEW CODE
        print("Calculating Near Hits ...")
        scores_nearest_hit, ranked_nearest_hit_data_entry = get_nearest_hits(rnd_img, pred_label, data.data, fe,
                                                                             sel_model, top_n, distance_measure,
                                                                             raw, distance_on_image)
        if BINARY:
            print("Calculating Near Misses ...")
            scores_nearest_miss, ranked_nearest_miss_data_entry = get_nearest_miss(rnd_img, pred_label, data.data, fe,
                                                                                   sel_model, top_n, distance_measure,
                                                                                   raw, distance_on_image)
        # gets top_n near misses per class instead of over all
        else:
            print("Calculating Near Misses multi ...")
            scores_nearest_miss_multi, ranked_nearest_miss_multi_data_entry = \
                get_nearest_miss_multi(rnd_img, data.available_classes, pred_label, data.data, fe, sel_model, top_n,
                                       distance_measure, raw, distance_on_image)

        toc = time.time()
        print("{}h {}min {}sec ".format(np.floor(((toc - tic) / (60 * 60))), np.floor(((toc - tic) % (60 * 60)) / 60),
                                        ((toc - tic) % 60)))

        # plot near hits and near misses
        plt.ioff()
        if top_n < 1:
            print("SCORES NEAREST HITS")
            print(pd.Series(scores_nearest_hit).describe())
        else:
            # Plot random image + heatmap
            heatmap_directory = os.path.join(STATIC_DIR, 'heatmaps', rnd_img.fe.fe_model.name, dataset)

            fig1 = plt.figure()
            plt.subplot(2, 1, 1)
            # TODO change title formatting
            plt.title(f"{rnd_img.img_name}\n\
                        Actual Label : {rnd_img.ground_truth_label}\n\
                        Predicted Label : {pred_label}", weight='bold', size=12)
            plt.imshow(img, cmap='gray')
            if distance_on_image and not raw:
                # get correct path to heatmap
                test_image_name = str.split(rnd_img.img_name, ".")[0]
                test_image_heatmap_path = os.path.join(heatmap_directory, "test", pred_label[0],
                                                       test_image_name + "_heatmap.png")
                plt.subplot(2, 1, 2)
                plt.title("Heatmap", weight='bold', size=12)
                pic = cv2.imread(test_image_heatmap_path, cv2.IMREAD_GRAYSCALE)
                plt.imshow(pic, cmap='gray', vmin=0, vmax=255)
            plt.tight_layout()
            plt.axis('off')
            fig1.savefig("fig1.png", bbox_inches='tight')
            plt.close(fig1)
            # fig1.show()
            # plt.show()

            fig2 = plot_nmnh(ranked_nearest_hit_data_entry, scores_nearest_hit, title="Near Hits")
            fig2.savefig("fig2.png", bbox_inches='tight')
            plt.close()
            if distance_on_image and not raw:
                fig3 = plot_nmnh_heatmaps(ranked_nearest_hit_data_entry, scores_nearest_hit, pred_label[0], title="Near Hits Heatmaps")
                fig3.savefig("fig3.png", bbox_inches='tight')
                plt.close()

            if BINARY:
                fig4 = plot_nmnh(ranked_nearest_miss_data_entry, scores_nearest_miss, title="Near Misses")
                fig4.savefig("fig4.png", bbox_inches='tight')
                plt.close()
            else:
                fig4 = plot_nmnh(np.concatenate(ranked_nearest_miss_multi_data_entry),
                          np.concatenate(scores_nearest_miss_multi),
                          title="Near Misses per Class")
                fig4.savefig("fig4.png", bbox_inches='tight')
                plt.close()
                if distance_on_image and not raw:
                    fig5 = plot_nmnh_heatmaps(np.concatenate(ranked_nearest_miss_multi_data_entry),
                                       np.concatenate(scores_nearest_miss_multi), pred_label[0],
                                       title="Near Misses Heatmaps per Class")
                    fig5.savefig("fig5.png", bbox_inches='tight')
                    plt.close()

            plt.subplot(2, 3, (1, 4))
            pic1 = cv2.imread("fig1.png", cv2.IMREAD_GRAYSCALE)
            plt.imshow(pic1, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.subplot(2, 3, 2)
            pic2 = cv2.imread("fig2.png", cv2.IMREAD_GRAYSCALE)
            plt.imshow(pic2, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.subplot(2, 3, 3)
            try:
                pic3 = cv2.imread("fig3.png", cv2.IMREAD_GRAYSCALE)
                plt.imshow(pic3, cmap='gray', vmin=0, vmax=255)
            except TypeError:  # FileNotFoundError
                pass
            plt.axis('off')
            plt.subplot(2, 3, 5)
            pic = cv2.imread("fig4.png", cv2.IMREAD_GRAYSCALE)
            plt.imshow(pic, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.subplot(2, 3, 6)
            try:
                pic = cv2.imread("fig5.png", cv2.IMREAD_GRAYSCALE)
                plt.imshow(pic, cmap='gray', vmin=0, vmax=255)
            except TypeError:  # FileNotFoundError
                pass
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.close()

        another_image = input("Do you want to get another overview of a random image? [y/n] ")


def nhnm_calc_for_all_testimages(dataset, suffix_path="_multicnn", type_of_model="cnn", distance_measure='cosine',
                      top_n=TOP_N_NMNH, use_prediction=True, raw=False, distance_on_image=False):

    from dataset import DataSet
    from modelsetup import ModelSetup, load_model_from_folder

    tic = time.time()

    # Load the first model
    # Initialize Feature Extractor Instance
    options_cnn = True if type_of_model == "cnn" else False
    tmp_model = load_model_from_folder(dataset, suffix_path=suffix_path)
    fe = FeatureExtractor(loaded_model=tmp_model,
                          model_name=str.upper(type_of_model),
                          options_cnn=options_cnn,
                          feature_model_output_layer=get_output_layer(tmp_model, type_of_model))
    data = DataSet(dataset, fe)

    # if distance measure requires feature embedding check whether all are created and if not -> create them
    if not distance_on_image:
        i = 0   # Counter for newly loaded feature embeddings
        for d in data.data:
            if os.path.exists(d.feature_file):
                np.load(d.feature_file, allow_pickle=True)
                pass
            else:  # if feature embedding doesn't exist yet, it is extracted now and saved
                _, x = d.fe.load_preprocess_img(d.img_file)
                feat = d.fe.extract_features(x)
                np.save(d.feature_file, feat)
                print("SAVE...")
                i += 1

        print("... newly loaded feature embeddings, which were not considered yet : ", i)

    # initialize model
    if type_of_model == "vgg":  # VGG16 will be used -> needs correct input shape # model_for_feature_embedding is None and
        sel_model = ModelSetup(data, sel_size=224)
    else:
        sel_model = ModelSetup(data)
    sel_model._preprocess_img_gen()
    sel_model.set_model(suffix_path=suffix_path)

    if type_of_model == 'cnn':
        sel_model.mode_rgb = False
    else:
        sel_model.mode_rgb = True

    print("------------- START -------------")
    # for every test image calc NH and NM
    test_names = []
    near_hits = []
    all_scores_nearest_hits = []
    near_misses = []
    all_scores_nearest_miss = []
    counter = 0
    for test_dataentry in random.sample(data.data_t, len(data.data_t)):
        # check if data is already present -> we can jump ahead (only possible with seed set)
        picklename = dataset + suffix_path + "_" + distance_measure + "_usepred" + str(
            use_prediction) + "_raw" + str(raw) + "_distonimg" + str(distance_on_image) + "_" + str(counter+1)
        picklepath = os.path.join(STATIC_DIR, picklename+".pickle")
        if os.path.exists(picklepath):
            test_names.append(test_dataentry.img_path)
            if len(test_names) == 5:
                print(counter)
                counter = counter + 1
                test_names = []
            continue

        # start with new image
        img, x = fe.load_preprocess_img(test_dataentry.img_path)
        feature_vector = fe.extract_features(x)

        pred_label = test_dataentry.ground_truth_label
        if use_prediction:
            pred_label, pred_prob = sel_model.pred_test_img(test_dataentry)

        # print("Calculating Near Hits ...")
        scores_nearest_hit, ranked_nearest_hit_data_entry = get_nearest_hits(test_dataentry, pred_label,
                                                                               data.data, fe,
                                                                               sel_model, top_n,
                                                                               distance_measure,
                                                                               raw, distance_on_image)
        if BINARY:
            # print("Calculating Near Misses ...")
            scores_nearest_miss, ranked_nearest_miss_data_entry = get_nearest_miss(test_dataentry,
                                                                                     pred_label, data.data,
                                                                                     fe, sel_model, top_n,
                                                                                     distance_measure,
                                                                                     raw, distance_on_image)
            near_misses.append([dataentry.img_path for dataentry in ranked_nearest_miss_data_entry])
            all_scores_nearest_miss.append(scores_nearest_miss)
        # gets top_n near misses per class instead of over all
        else:
            # print("Calculating Near Misses multi ...")
            scores_nearest_miss, ranked_nearest_miss_multi_data_entry = \
                get_nearest_miss_multi(test_dataentry, data.available_classes, pred_label, data.data, fe,
                                       sel_model, top_n,
                                       distance_measure, raw, distance_on_image)
            near_misses.append([[dataentry.img_path for dataentry in lst] for lst in ranked_nearest_miss_multi_data_entry])
            all_scores_nearest_miss.append(scores_nearest_miss)

        test_names.append(test_dataentry.img_path)
        near_hits.append([dataentry.img_path for dataentry in ranked_nearest_hit_data_entry])
        all_scores_nearest_hits.append(scores_nearest_hit)
        # if counter == 3:
        #     break
        if len(test_names) == 5:
            counter = counter + 1
            print(counter*5)
            # safe in between
            df = pd.DataFrame({"image_name": test_names,
                               "near_hits": near_hits,
                               "scores_hits": all_scores_nearest_hits,
                               "near_misses": near_misses,
                               "scores_misses": all_scores_nearest_miss})
            toc = time.time()
            print(
                "{}h {}min {}sec ".format(np.floor(((toc - tic) / (60 * 60))), np.floor(((toc - tic) % (60 * 60)) / 60),
                                          ((toc - tic) % 60)))

            # safe dataframe as pickle
            picklename = dataset + suffix_path + "_" + distance_measure + "_usepred" + str(
                use_prediction) + "_raw" + str(raw) + "_distonimg" + str(distance_on_image) + "_" + str(counter)
            picklepath = os.path.join(STATIC_DIR, picklename)
            df.to_pickle(picklepath + ".pickle")
            # clean workspace
            test_names = []
            near_hits = []
            all_scores_nearest_hits = []
            near_misses = []
            all_scores_nearest_miss = []

    df = pd.DataFrame({"image_name": test_names,
                       "near_hits": near_hits,
                       "scores_hits": all_scores_nearest_hits,
                       "near_misses": near_misses,
                       "scores_misses": all_scores_nearest_miss})

    toc = time.time()
    print("{}h {}min {}sec ".format(np.floor(((toc - tic) / (60 * 60))), np.floor(((toc - tic) % (60 * 60)) / 60),
                                    ((toc - tic) % 60)))

    # safe dataframe as pickle
    picklename = dataset+suffix_path+"_"+distance_measure+"_usepred"+str(use_prediction)+"_raw"+str(raw)+"_distonimg"+str(distance_on_image)+"_"+str(counter)
    picklepath = os.path.join(STATIC_DIR, picklename)
    df.to_pickle(picklepath+".pickle")

    print("------------- FINISHED -------------")

    return df


if __name__ == '__main__':
    # dataset_to_use = "mnist_1247"

    # get_nhnm_overview(dataset_to_use, top_n=TOP_N_NMNH, use_prediction=True, suffix_path="_cnn_seed3871",  # TOP_N_NMNH
    #                  type_of_model="cnn", distance_measure='SSIM-pushed', raw=False, distance_on_image=True)

    # dataset_to_use = "oct_cc"  # model_history_oct_cc_cnn_seed3871
    # get_nhnm_overview(dataset_to_use, top_n=TOP_N_NMNH, use_prediction=True, suffix_path="_cnn_seed3871",  # TOP_N_NMNH
    #                   type_of_model="cnn", distance_measure='SSIM-threshold', raw=False, distance_on_image=True)

    # dist == 'CW-SSIM' # Very slow algorithm - up to 50x times slower than SIFT or SSIM. but good results
    # dist == SSIM # Default SSIM implementation of Scikit-Image # quick

    dataset_to_use = "mnist_1247"  # input("Which data set? [mnist_1247/oct_cc] ")
    distance_measure = "SSIM-threshold"  # input("Distance Measure [SSIM/SSIM-pushed/SSIM-mm/SSIM-blur/SSIM-threshold/CW-SSIM/euclidean/cosine/manhatten] ")
    df = nhnm_calc_for_all_testimages(dataset_to_use, top_n=TOP_N_NMNH,
                                      suffix_path="_cnn_seed3871", type_of_model="cnn", distance_measure=distance_measure,
                                      use_prediction=True, raw=False, distance_on_image=True)
    # print(df.describe())
    # print(df.head())


###
# pd.read_pickle("path")

