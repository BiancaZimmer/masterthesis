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
        distances = np.linalg.norm(feature_embedding-feature_vector, axis=1)  
        distances = np.array([distance.euclidean(feature_vector, feat) for feat in feature_embedding ])
    elif dist == 'cosine':
        distances =  np.array([distance.cosine(feature_vector, feat) for feat in feature_embedding ])
    elif dist == 'manhattan':
        distances =  np.array([distance.cityblock(feature_vector, feat) for feat in feature_embedding ])
        
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


def get_nearest_miss(test_dataentry, pred_label, data, fe,  top_n:int =5, distance_measure:str ='cosine'):
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


# if __name__ == '__main__':
#     ###### ==== Select a DATASET ==== ######
#     # dataset = "quality"
#     dataset = "mnist"

#     # Initialize Feature Extractor Instance 
#     fe = FeatureExtractor()

#     # Image path of the dataset
#     image_path = os.path.join(DATA_DIR,dataset, 'train')

#     # Start Timer
#     tic = time.time()

#     # Allowed image extensions
#     image_extensions = ['.jpg','.jpeg', '.bmp', '.png', '.gif']

#     data = [DataEntry(fe,dataset,os.path.join(path, file)) for path, _, files in os.walk(image_path) for file in files if file.endswith(tuple(image_extensions))]


#     # Counter for loaded Images
#     i = 0

#     for d in data:
#         if os.path.exists(d.feature_file):
#             np.load(d.feature_file, allow_pickle=True)
#             pass
#         else:
#             _, x = d.fe.load_preprocess_img(d.img_file)
#             feat = d.fe.extract_features(x)
#             np.save(d.feature_file, feat)
#             print("SAVE...")
#             i += 1
#             pass

#     print("... reloaded images, which were not considered yet : ", i)

#     ###### ==== Select a Random Images (-> input image later) ==== ######
#     #idx_Test = 55

#     # grab a random query image
#     # = int(len(X_test) * random.random())
#     #print("==> RANDOM INDEX: ", idx_Test, " (Label: ", int(y_test[idx_Test]), ")")
#     #rnd_img = os.path.split(X_test[idx_Test])[1]
#     # rnd_img_path = os.path.join(DATA_DIR,dataset, 'test', 'def_front')
#     rnd_img_path = os.path.join(DATA_DIR,dataset, 'test', 'class_7')
#     rnd_img_file = random.choice(os.listdir(rnd_img_path))
#     print(rnd_img_path)
#     print(rnd_img_file)

#     rnd_img = DataEntry(fe,dataset,os.path.join(rnd_img_path, rnd_img_file))

#     img, x = fe.load_preprocess_img(rnd_img.img_file)
#     feature_vector = fe.extract_features(x)

#     top_n = 5

#     distance_measure = 'cosine'

#     hit_class_idx = []
#     miss_class_idx = []

#     for f in data:
#         if f.ground_truth_label == rnd_img.ground_truth_label:
#             hit_class_idx.append(f)
#         else:
#             miss_class_idx.append(f)

#     print(np.size(hit_class_idx))
#     print(np.size(miss_class_idx))


#     # # hit_class_idx = [int(np.where(img_paths==x)[0]) for x in X_train[np.where(y_train==label_img)[0]]]
#     # # miss_class_idx = [int(np.where(img_paths==x)[0]) for x in X_train[np.where(y_train!=label_img)[0]]]


#     # # scores, idx_ranked = calc_distances_scores(feature_embedding[np.where(y_train[:len(feature_embedding)]==label_img)[0]], feature_vector, top_n = top_n, dist = 'L2', return_idx_ranked = True)
#     scores_nearest_hit, idx_ranked_nearest_hit = calc_distances_scores([x.feature_embedding for x in hit_class_idx], feature_vector, top_n = top_n, dist = distance_measure, return_idx_ranked = True)
#     scores_nearest_miss, idx_ranked_nearest_miss = calc_distances_scores([x.feature_embedding for x in miss_class_idx], feature_vector, top_n = top_n, dist = distance_measure, return_idx_ranked = True)


#     # Plot similar images
#     fig = plt.figure(figsize=(16, 12))

    
#     columns = 5
#     rows = np.ceil(top_n / columns) * 2 + 1

#     #plot random images / query images
#     ax = fig.add_subplot(rows, columns, 1)
#     #fig, ax = plt.subplots(1,1, figsize = (16,12))
#     ax.imshow(Image.open(rnd_img.img_file))
#     ax.title.set_text("Label = " + str(rnd_img.ground_truth_label))
#     ax.axis('off')

#     # print(np.array(hit_class_idx)[idx_ranked_nearest_hit])

#     for i_n, n in enumerate([idx_ranked_nearest_hit, idx_ranked_nearest_miss]):

#         if np.array_equal(n, idx_ranked_nearest_hit):
#             X_n = [x.img_file for x in np.array(hit_class_idx)[idx_ranked_nearest_hit]]
#             scores = scores_nearest_hit
#             y_n = [x.ground_truth_label for x in np.array(hit_class_idx)[idx_ranked_nearest_hit]]
#         elif np.array_equal(n,idx_ranked_nearest_miss):
#             X_n = [x.img_file for x in np.array(miss_class_idx)[idx_ranked_nearest_miss]]
#             scores = scores_nearest_miss
#             y_n = [x.ground_truth_label for x in np.array(miss_class_idx)[idx_ranked_nearest_miss]]


#         for i, idx in enumerate(n):
#             ### img_sim = Image.open(np.array(img_paths)[np.where(y_train==label_img)[0]][idx])
#             #print(y_train[np.where(X_train==X_n[idx])])
#             img_sim = Image.open(X_n[i])
#             # img_sim = Image.open(np.array(img_paths)[np.where(y_train[:len(feature_embedding)]==label_img)[0]][idx])
#             ax = fig.add_subplot(rows, columns, i+1+columns+i_n*columns)
#             ax.imshow(img_sim)
#             ax.title.set_text('Score = ' + str(round(scores[i][0],3)) + ' Label: '+ str(y_n[i]))
#             #ax.title.set_text('Score = ' + str(scores[i][0]) + ' Label: '+ str(os.path.splitext(os.path.split(np.array(img_paths)[np.where(y_train[:len(feature_embedding)]==label_img)[0]][idx])[1])[0].split('_')[1]))
#             ax.axis('off')

#     plt.show()
