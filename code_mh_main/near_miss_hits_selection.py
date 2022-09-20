import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
import cv2
import ssim.ssimlib as pyssim
from skimage.metrics import structural_similarity as ssim

from feature_extractor import FeatureExtractor
from dataentry import DataEntry
from utils import *
from LRP_heatmaps import generate_LRP_heatmap, create_special_analyzer

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


def calc_distances_scores(class_data_entry_list, feature_vector, top_n: int = 5, dist: str = 'cosine',
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
    idx_ranked = np.argsort(distances)[:top_n]
    if plot_idx:
        print("Index of top distances: ", idx_ranked)

    scores = distances[idx_ranked]
    
    if return_data_entry_ranked:
        return scores, [class_data_entry_list[i] for i in idx_ranked]
    else:
        return scores


def calc_image_distance(class_data_entry_list, image_path_test, top_n: int = 5, dist: str = 'SSIM', lrp: bool = True,
                        return_data_entry_ranked: bool = False, plot_idx: bool = False):
    """

    :param class_data_entry_list: List of DataEntries, note class reference/filter has to be done in advance if required.
    :type class_data_entry_list: list
    :param image_path_test: Image path of the image for which distances should be computed
    :param top_n: Set an integer value how many nearest samples should be selected, defaults to 5
    :type top_n: int, optional
    :param dist: Distance applied to images, e.g. 'SWIFT'/'SSIM'/'CW-SSIM', defaults to 'SSIM'
    :param lrp: base the distance measurement on the lrp heatmaps (True) or on the raw images (False), defaults to True
    :param return_data_entry_ranked: Set True in order to get a list of the DataEntries of the nearest samples, defaults to False
    :type return_data_entry_ranked: bool, optional
    :param plot_idx: Set to True in order to plot the indices of the nearest data samples, defaults to False
    :type plot_idx: bool, optional

    :return:
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_dataentry** (`list`) - If 'return_data_entry_ranked' set to True, a list of the DataEntries of the nearest samples
    """

    raw_image_paths_list = [img.img_path for img in class_data_entry_list]
    image_paths_list = raw_image_paths_list
    if lrp:
        # TODO: img_path to lrp heatmap, currently distance on raw image
        image_paths_list = raw_image_paths_list
    images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_paths_list]
    img_test = cv2.imread(image_path_test, cv2.IMREAD_GRAYSCALE)

    def sift_similarity(im1, im2, sift_ratio: float = 0.7):
        # Using OpenCV for feature detection and matching
        sift = cv2.SIFT_create()  # cv2.xfeatures2d.SIFT_create()
        k1, d1 = sift.detectAndCompute(im1, None)
        k2, d2 = sift.detectAndCompute(im2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1, d2, k=2)

        good_matches = 0.0
        for m, n in matches:
            if m.distance < sift_ratio * n.distance:
                good_matches += 1.0

        # calculation of similarity:
        if len(matches) == 0:
            similarity = 0.0
        else:
            similarity = good_matches/len(matches)

        # # Custom normalization for better variance in the similarity matrix
        # if good_matches == len(matches):
        #     similarity = 1.0
        # elif good_matches > 1.0:
        #     similarity = 1.0 - 1.0 / good_matches
        # elif good_matches == 1.0:
        #     similarity = 0.1
        # else:
        #     similarity = 0.0

        # similarity is between 0;1 with 0 different and 1 same
        # -> trafo to 0;1 with 0 same and 1 different

        result = 1-similarity

        return result

    if dist == 'SIFT':
        distances = [sift_similarity(image, img_test, sift_ratio=0.7) for image in images]

    elif dist == 'CW-SSIM':
        # FOR EXPERIMENTS ONLY!
        # Very slow algorithm - up to 50x times slower than SIFT or SSIM.
        # Optimization using CUDA or Cython code should be explored in the future.
        # value between 0;1 with 0 different and 1 same
        # -> trafo to 0;1 with 0 same and 1 different
        pil_test_image = Image.open(image_path_test)

        def calc_cw_ssim(image_path):
            pil = Image.open(image_path)
            result = pyssim.SSIM(pil).cw_ssim_value(pil_test_image)
            pil.close()
            result = 1-result
            return result
        distances = [calc_cw_ssim(img_path) for img_path in image_paths_list]
    else:  # dist == SSIM
        # Default SSIM implementation of Scikit-Image
        # value between -1; 1 ; -1 different, 1 the same
        # -> trafo to structural dissimilarity where value between 0;1 with 0 same and 1 different
        def dssim(img):
            ssim_index = ssim(img, img_test)
            result = (1-ssim_index)/2
            return result
        distances = [dssim(image) for image in images]

    # Top distances
    idx_ranked = np.argsort(distances)[:top_n]
    if plot_idx:
        print("Index of top distances: ", idx_ranked)

    scores = np.array(distances)[idx_ranked]

    if return_data_entry_ranked:
        return scores, [class_data_entry_list[i] for i in idx_ranked]
    else:
        return scores


def get_nearest_hits(test_dataentry, pred_label, data, fe, top_n:int = 5, distance_measure: str = 'cosine', rgb: bool = False):
    """Function to calculates the near hits in respect to a given test inputs sample (DataEntry).

    :param test_dataentry: DataEntry object (test input sample) on which near hits should be selected
    :type test_dataentry: DataEntry
    :param pred_label: Prediction label of the CNN classifier which should be also explained somewhat, which refers to the folder name (class)
    :type pred_label: str
    :param data: Data of DataSet (list of DataEntries)
    :type data: list
    :param fe: FeatureExtractor model that is used to extract the features of the test input sample
    :type fe: FeatureExtractor
    :param top_n: Set an integer value how many near hits should be selected, defaults to 5
    :type top_n: int, optional
    :param distance_measure: Distance applied in feature embedding, e.g. 'euclidean'/'cosine'/'manhattan', defaults to 'cosine'
    :type distance_measure: str, optional
    :param rgb: image input for model in rgb (3 channel) or grayscale (1 channel); defaults to False -> grayscale
    :return: 
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_nearest_hit_data_entry** (`list`) - List of the DataEntries of the near hits
    """

    hit_class_data_entry = list(filter(lambda x: x.ground_truth_label == pred_label, data))

    if distance_measure in ['cosine', 'manhatten', 'euclidean']:
        _, x = fe.load_preprocess_img(test_dataentry.img_path)
        feature_vector = fe.extract_features(x)

        scores_nearest_hit, ranked_nearest_hit_data_entry = calc_distances_scores(hit_class_data_entry, feature_vector,
                                                                                  top_n=top_n,
                                                                                  dist=distance_measure,
                                                                                  return_data_entry_ranked=True)
    else:
        # TODO: implement LRP Heatmap
        scores_nearest_hit, ranked_nearest_hit_data_entry = calc_image_distance(hit_class_data_entry,
                                                                                test_dataentry.img_path,
                                                                                top_n=top_n, dist=distance_measure,
                                                                                return_data_entry_ranked=True)
    return scores_nearest_hit, ranked_nearest_hit_data_entry


def get_nearest_miss(test_dataentry, pred_label, data, fe, top_n: int = 5, distance_measure: str = 'cosine', rgb: bool = False):
    """Function to calculates the near misses in respect to a given test inputs sample (DataEntry).

    :param test_dataentry: DataEntry object (test input sample) on which near misses should be selected
    :type test_dataentry: DataEntry
    :param pred_label: Prediction label of the CNN classifier which should be also explained somewhat, which refers to the folder name (class)
    :type pred_label: str
    :param data: Data of DataSet (list of DataEntries)
    :type data: list
    :param fe: FeatureExtractor model that is used to extract the features of the test input sample
    :type fe: FeatureExtractor
    :param top_n: Set an integer value how many near misses should be selected, defaults to 5
    :type top_n: int, optional
    :param distance_measure: Distance applied in feature embedding, e.g. 'euclidean'/'cosine'/'manhattan', defaults to 'cosine'
    :type distance_measure: str, optional
    :param rgb: image input for model in rgb (3 channel) or grayscale (1 channel); defaults to False -> grayscale
    :return: 
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_nearest_miss__data_entry** (`list`) - List of the DataEntries of the near misses
    """
    miss_class_data_entry = list(filter(lambda x: x.ground_truth_label != pred_label, data))

    if distance_measure in ['cosine', 'manhatten', 'euclidean']:
        _, x = fe.load_preprocess_img(test_dataentry.img_path)
        feature_vector = fe.extract_features(x)

        scores_nearest_miss, ranked_nearest_miss__data_entry = calc_distances_scores(miss_class_data_entry,
                                                                                     feature_vector, top_n=top_n,
                                                                                     dist=distance_measure,
                                                                                     return_data_entry_ranked=True)
    else:
        # TODO: implement LRP Heatmap
        scores_nearest_miss, ranked_nearest_miss__data_entry = calc_image_distance(miss_class_data_entry,
                                                                                   test_dataentry.img_path, top_n=top_n,
                                                                                   dist=distance_measure,
                                                                                   return_data_entry_ranked=True)

    return scores_nearest_miss, ranked_nearest_miss__data_entry


def get_nearest_miss_multi(test_dataentry, classes, pred_label, data, fe, top_n:int =5, distance_measure:str ='cosine'):
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
    :param top_n: Set an integer value how many near misses should be selected, defaults to 5
    :type top_n: int, optional
    :param distance_measure: Distance applied in feature embedding, e.g. 'euclidean'/'cosine'/'manhattan', defaults to 'cosine'
    :type distance_measure: str, optional
    :return:
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_nearest_miss__data_entry** (`list`) - List of the DataEntries of the near misses
    """

    available_classes = [c for c in classes if c != pred_label]

    scores = []
    data_entries = []
    for miss_class in available_classes:
        scores_nearest_miss, ranked_nearest_miss_data_entry = get_nearest_hits(test_dataentry, miss_class, data, fe, top_n, distance_measure)
        scores.append(scores_nearest_miss)
        data_entries.append(ranked_nearest_miss_data_entry)

    return scores, data_entries


def plot_nmnh(dataentries, similarity_scores: float, title: str = "Near Miss/Near Hit Plot"):

    plt.figure(figsize=(20, 8))
    plt.suptitle(title)
    for dataentry, sim, i in zip([x for x in dataentries], similarity_scores, range(len(similarity_scores))):
        pic = cv2.imread(dataentry.img_path)
        plt.subplot(int(np.ceil(len(similarity_scores) / 5)), 5, i + 1)
        plt.title(f"{dataentry.img_name}\n\
        Actual Label : {dataentry.ground_truth_label}\n\
        Similarity : {'{:.3f}'.format(sim)}", weight='bold', size=12)

        plt.imshow(pic, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_nhnm_overview(dataset, suffix_path="_multicnn", distance_measure='cosine',
                      top_n=TOP_N_NMNH, use_prediction=True):
    # top_n = TOP_N_NMNH   # number of near hits/misses to show
    # use_prediction = True  # if set to true a prediction of the image is used for the near hits/misses
    # suffix_path = "_multicnn"   # if use_prediction=True then you have to specify which model of the dataset to use
    # distance_measure = 'cosine'  # distance measure for near miss/near hit
    # TODO rgb option

    from dataset import DataSet
    from modelsetup import ModelSetup, load_model_from_folder

    # You don't have to do anything from here on
    tic = time.time()

    fe = FeatureExtractor(loaded_model=load_model_from_folder(dataset, suffix_path=suffix_path)) # Initialize Feature Extractor Instance
    data = DataSet(dataset, fe)

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
            pass

    print("... newly loaded feature embeddings, which were not considered yet : ", i)

    ###### ==== Select a Random Images (-> input image later) ==== ######
    rnd_class = random.choice(data.available_classes)
    rnd_img_path = os.path.join(DATA_DIR, dataset, 'test', rnd_class)
    rnd_img_file = random.choice(os.listdir(rnd_img_path))
    # rnd_img_path = os.path.join(DATA_DIR, dataset, 'test', '6')     # 7 -6576 wrong prediction; 7-1260
    # rnd_img_file = '5481.jpg'
    print("Random image:", rnd_img_file, " in folder:")
    print(rnd_img_path)

    rnd_img = DataEntry(fe, dataset, os.path.join(rnd_img_path, rnd_img_file))

    img, x = fe.load_preprocess_img(rnd_img.img_path)
    feature_vector = fe.extract_features(x)

    pred_label = rnd_img.ground_truth_label
    if use_prediction:
        sel_model = ModelSetup(selected_dataset=data)
        sel_model._preprocess_img_gen(rgb=False)  # TODO atm only cnn model
        sel_model.set_model(suffix_path=suffix_path)
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
    scores_nearest_hit, ranked_nearest_hit_data_entry = get_nearest_hits(rnd_img, pred_label,
                                                                         data.data, fe, top_n, distance_measure)
    scores_nearest_miss, ranked_nearest_miss_data_entry = get_nearest_miss(rnd_img, pred_label,
                                                                           data.data, fe, top_n, distance_measure)

    # gets top_n near misses per class instead of over all
    if not BINARY:
        scores_nearest_miss_multi, ranked_nearest_miss_multi_data_entry = \
            get_nearest_miss_multi(rnd_img, data.available_classes, pred_label, data.data, fe, top_n, distance_measure)

    toc = time.time()
    print("{}h {}min {}sec ".format(np.floor(((toc - tic) / (60 * 60))), np.floor(((toc - tic) % (60 * 60)) / 60),
                                    ((toc - tic) % 60)))

    # Show random image and its near misses and hits
    # pic = cv2.imread(rnd_img.img_path)
    plt.title(f"{rnd_img.img_name}\n\
                Actual Label : {rnd_img.ground_truth_label}", weight='bold', size=12)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plot_nmnh(ranked_nearest_hit_data_entry, scores_nearest_hit, title="Near Hits")
    plot_nmnh(ranked_nearest_miss_data_entry, scores_nearest_miss, title="Near Misses")
    if not BINARY:
        plot_nmnh(np.concatenate(ranked_nearest_miss_multi_data_entry), np.concatenate(scores_nearest_miss_multi),
                  title="Near Misses per Class")


if __name__ == '__main__':
    dataset_to_use = "mnist"

    get_nhnm_overview(dataset_to_use, top_n=TOP_N_NMNH, use_prediction=True, suffix_path="_multicnn2", distance_measure='euclidean')

    # dist == 'SIFT' # quick but results questionable
    # dist == 'CW-SSIM' # Very slow algorithm - up to 50x times slower than SIFT or SSIM. but good results
    # dist == SSIM # Default SSIM implementation of Scikit-Image # quick but negative similarities?


