import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance

from feature_extractor import FeatureExtractor
from dataentry import DataEntry
from utils import *


def calc_distances_scores(class_data_entry_list, feature_vector, top_n:int =5, dist:str ='cosine', return_data_entry_ranked = False, plot_idx:bool =False):
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


def get_nearest_hits(test_dataentry, pred_label, data, fe, top_n:int =5, distance_measure:str ='cosine'):
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
    :return: 
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_nearest_hit_data_entry** (`list`) - List of the DataEntries of the near hits
    """
    _, x = fe.load_preprocess_img(test_dataentry.img_path)
    feature_vector = fe.extract_features(x)


    hit_class_data_entry = list(filter(lambda x: x.ground_truth_label == pred_label, data))
    scores_nearest_hit, ranked_nearest_hit_data_entry = calc_distances_scores(hit_class_data_entry, feature_vector, top_n = top_n, dist = distance_measure, return_data_entry_ranked = True)
    
    return scores_nearest_hit, ranked_nearest_hit_data_entry


def get_nearest_miss(test_dataentry, pred_label, data, fe, top_n:int =5, distance_measure:str ='cosine'):
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
    :return: 
        - **scores** (`list`) - List of scores (based on the selected distance)
        - **ranked_nearest_miss__data_entry** (`list`) - List of the DataEntries of the near misses
    """
    _, x = fe.load_preprocess_img(test_dataentry.img_path)
    feature_vector = fe.extract_features(x)

    miss_class_data_entry = list(filter(lambda x: x.ground_truth_label != pred_label, data))
    scores_nearest_miss, ranked_nearest_miss__data_entry = calc_distances_scores(miss_class_data_entry, feature_vector, top_n = top_n, dist = distance_measure, return_data_entry_ranked = True)
    
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


if __name__ == '__main__':
    from dataset import *

    ###### ==== Select a DATASET ==== ######
    dataset = DATA_DIR_FOLDERS[0]   # TODO: careful: This is hard coded, always takes first data set
    # dataset = "mnist"
    top_n = TOP_N_NMNH   # number of near hits/misses to show
    use_prediction = True  # if set to true a prediction of the image is used for the near hits/misses
    suffix_path = "_multicnn"   # if use_prediction=True then you have to specify which model of the dataset to use
    distance_measure = 'cosine'  # distance measure for near miss/near hit

    # You don't have to do anything from here on
    tic = time.time()

    use_all_datasets = True
    if len(DATA_DIR_FOLDERS) > 0: use_all_datasets = False

    fe = FeatureExtractor()     # Initialize Feature Extractor Instance
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
    # idx_Test = 55

    # grab a random query image
    rnd_class = random.choice(data.available_classes)
    rnd_img_path = os.path.join(DATA_DIR, dataset, 'test', rnd_class)
    rnd_img_file = random.choice(os.listdir(rnd_img_path))
    # rnd_img_path = os.path.join(DATA_DIR, dataset, 'test', '7')     # 7 -6576 wrong prediction
    # rnd_img_file = '6576.jpg'
    print(rnd_img_path)
    print(rnd_img_file)

    rnd_img = DataEntry(fe, dataset, os.path.join(rnd_img_path, rnd_img_file))

    img, x = fe.load_preprocess_img(rnd_img.img_path)
    feature_vector = fe.extract_features(x)

    pred_label = rnd_img.ground_truth_label
    if use_prediction:  # TODO atm only cnn model
        cnn_model = CNNmodel(selected_dataset=data)
        cnn_model._preprocess_img_gen()
        cnn_model.load_model(suffix_path=suffix_path)
        pred_label, pred_prob = cnn_model.pred_test_img(rnd_img)
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
        scores_nearest_miss_multi, ranked_nearest_miss_multi_data_entry =\
            get_nearest_miss_multi(rnd_img, data.available_classes, pred_label, data.data, fe, top_n, distance_measure)

    toc = time.time()
    print("{}h {}min {}sec ".format(round(((toc - tic) / (60 * 60))), np.floor(((toc - tic) % (60 * 60))/60),
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
