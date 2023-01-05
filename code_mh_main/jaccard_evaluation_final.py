# -*- coding: utf-8 -*-
# ## Imports

import math
import os
import time
import random
import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# from feature_extractor import FeatureExtractor
from dataentry import DataEntry
from helpers import jaccard, change_imgpath, vconcat_resize_min, hconcat_resize_max
from utils import *

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


# ## Functions

# +
def humanfriendly_trafo(path):
    """ transforms a path and returns the last element (file name) of the path

    Takes as input:
     str (path)
     list of str
     list of list of str

    :param path: str or list, path to be transformed
    :return: file names of input paths in input format
    """
    if type(path) is str:
        return path.split("/")[-1]
    elif type(path[0]) is str:
        return [entry.split("/")[-1] for entry in path]
    else:
        return [[entry.split("/")[-1] for entry in lst] for lst in path]


def show_humanfriendly(df, columns=["image_name", "near_hits", "near_misses", "top_misses"]):
    """

    :param df: pandas.DataFrame, DataFrame which shall be shown in an easily readable manner
    :param columns: list, list of column names which shall be transformed
    :return: pandas.DataFrame with shortened paths
    """
    dfh = df.copy()
    for c in columns:
        dfh[c] = [humanfriendly_trafo(row) for row in df[c]]
    return dfh


# -

def jaccard_df(df1, df2, method="intersection"):
    """ Calculates Jaccard Index over whole dataframe per row, best put in two column vectors

    :param df1: pandas.DataFrame, first df to be compared
    :param df2: pandas.DataFrame, second df to be compared to df2
    :param method: str, method according to function **jaccard**
    :return: column vector with Jaccard Indeices
    """
    if type(df1[0][0]) is list:
        result = [jaccard(list(chain.from_iterable(l1)), list(chain.from_iterable(l2)), method) for
                  l1, l2 in zip(df1, df2)]
        return result
    else:
        result = [jaccard(l1, l2, method) for l1, l2 in zip(df1, df2)]
        return result


# ## Load pickles

# +
# predefine names
m0000_eucl = m0000_ssim = m0000_cw = []
m0001_eucl = m0001_ssim = m0001_cw = []
m0010_eucl = m0010_ssim = m0010_cw = []
m0011_eucl = m0011_ssim = []  # m0011_cw = []
m0100_eucl = m0101_eucl = []

o1000_eucl = o1000_ssim = []
o1001_eucl = o1001_ssim = []
o1010_eucl = o1010_ssim = []
o1011_eucl = o1011_ssim = []
o1100_eucl = o1101_eucl = []

all_df = {"0000_eucl": m0000_eucl, "0000_ssim": m0000_ssim, "0000_cw": m0000_cw,
          "0001_eucl": m0001_eucl, "0001_ssim": m0001_ssim, "0001_cw": m0001_cw,
          "0010_eucl": m0010_eucl, "0010_ssim": m0010_ssim, "0010_cw": m0010_cw,
          "0011_eucl": m0011_eucl, "0011_ssim": m0011_ssim, # "0011_cw": m0011_cw,
          "0100_eucl": m0100_eucl, "0101_eucl": m0101_eucl, 
          "1000_eucl": o1000_eucl, "1000_ssim": o1000_ssim,
          "1001_eucl": o1001_eucl, "1001_ssim": o1001_ssim,
          "1010_eucl": o1010_eucl, "1010_ssim": o1010_ssim,
          "1011_eucl": o1011_eucl, "1011_ssim": o1011_ssim,
          "1100_eucl": o1100_eucl, "1101_eucl": o1101_eucl
         }

mnist_df_names = ["0000_eucl", "0000_ssim", "0000_cw",
          "0001_eucl", "0001_ssim", "0001_cw",
          "0010_eucl", "0010_ssim", "0010_cw",
          "0011_eucl", "0011_ssim", # "0011_cw",
          "0100_eucl", "0101_eucl"]

oct_df_names = ["1000_eucl", "1000_ssim",
          "1001_eucl", "1001_ssim",
          "1010_eucl", "1010_ssim",
          "1011_eucl", "1011_ssim",
          "1100_eucl", "1101_eucl"]
# -

# load all pickles
for df in all_df:
    picklepath = "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/NHNM/" + df + \
                 "_FINAL50.pickle"
    print(picklepath)
    all_df[df] = pd.read_pickle(picklepath)

mnist_df = {df_name: all_df[df_name] for df_name in all_df if df_name in mnist_df_names}
oct_df = {df_name: all_df[df_name] for df_name in all_df if df_name in oct_df_names}

# ## Overview over distance scores

# ### MNIST

# +
# generate column names
scores_names = []
scores_top_names = []
for m in mnist_df:
    scores_names.append("scores_hit_"+m)
    scores_top_names.append("scores_top_misses_"+m)

# metrics
eucl = [0, 3, 6, 9, 11, 12]
ssim = [1, 4, 7, 10]
cw = [2, 5, 8]

scores_top_names

# +
mnist_scores = pd.DataFrame()

for score_hit, score_top, df in zip(scores_names, scores_top_names, mnist_df.values()):
    mnist_scores[score_hit] = list(chain.from_iterable(df.scores_hits))
    mnist_scores[score_top] = list(chain.from_iterable(df.scores_top_misses))
mnist_scores
# -

mnist_scores.describe().transpose()

# #### Distances for Hits

mnist_scores.boxplot(column=scores_names, rot= 30)

mnist_scores.boxplot(column= [scores_names[i] for i in eucl], rot= 20)

mnist_scores.boxplot(column= [scores_names[i] for i in ssim+cw], rot= 20)

# #### Distances for Top Misses

mnist_scores.boxplot(column=scores_top_names, rot= 20)

mnist_scores.boxplot(column=[scores_top_names[i] for i in eucl], rot= 20)

mnist_scores.boxplot(column=[scores_top_names[i] for i in ssim+cw], rot= 20)

# #### Compare misses + hits

mnist_scores.describe().transpose()[["mean","50%"]]

cols = [scores_names[i] for i in eucl]+[scores_top_names[i] for i in eucl]
cols.sort(key=lambda name: name[-9:])
mnist_scores.boxplot(column= cols, rot= 45)

cols = [scores_names[i] for i in ssim]+[scores_top_names[i] for i in ssim]
cols.sort(key=lambda name: name[-9:])
mnist_scores.boxplot(column= cols, rot= 45)

cols = [scores_names[i] for i in cw]+[scores_top_names[i] for i in cw]
cols.sort(key=lambda name: name[-7:])
mnist_scores.boxplot(column= cols, rot= 45)

# #### Results:
# all distance measures:
# * distances for misses are higher than for hits as expected; exception: 0010_euclidean, here distances for misses are a lot lower
# * distances of near hits behave about the same as distances of near misses.
#
# euclidean distances:
# * euclidean distances for 0011 (lrp + vgg) are a lot higher than all the other euclidean distances -> vgg does not perform well -> as expected
# * euclidean distances for 0101 are also higher than those of 0100 -> vgg does not perform well
# * euclidean distances for FE are a lot smaller than those of LRP -> to be expected since euclidean distanmce might not be the optimal citerion here
#
# SSIM + CW-SSIM:
# * SSIM distances are generally lower than those of the CW-SSIM; this could be due to the fact that rotation + translation is not taken into account and a lot of the images are black; otherwise this is contra-intuitive
# * SSIM on the LRP heatmaps are a lot smaller than on the raw images; this could be due to the fact that a lot more of the LRP heatmaps share the same color (grey) than the original image (black); this observation can not be seen in the CW-LRP heatmaps vs the CW-raw images

# ### OCT

# +
# generate column names
scores_names = []
scores_top_names = []
for m in oct_df:
    scores_names.append("scores_hit_"+m)
    scores_top_names.append("scores_top_misses_"+m)

# metrics
eucl = [0, 2, 4, 6, 8, 9]
ssim = [1, 3, 5, 7]

scores_top_names

# +
oct_scores = pd.DataFrame()

for score_hit, score_top, df in zip(scores_names, scores_top_names, oct_df.values()):
    oct_scores[score_hit] = list(chain.from_iterable(df.scores_hits))
    oct_scores[score_top] = list(chain.from_iterable(df.scores_top_misses))
oct_scores
# -

oct_scores.describe().transpose()

oct_df["1000_eucl"]["image_name"].equals(oct_df["1001_eucl"]["image_name"])

# #### Hits comparison

oct_scores.boxplot(column=scores_names, rot= 30)

oct_scores.boxplot(column= [scores_names[i] for i in eucl], rot= 20)

oct_scores.boxplot(column= [scores_names[i] for i in ssim], rot= 20)

# #### Misses Comparison

oct_scores.boxplot(column=scores_top_names, rot= 30)

oct_scores.boxplot(column= [scores_top_names[i] for i in eucl], rot= 20)

oct_scores.boxplot(column= [scores_top_names[i] for i in ssim], rot= 20)

# #### Compare misses+hits

oct_scores.describe().transpose()[["mean","50%"]]

cols = [scores_names[i] for i in eucl]+[scores_top_names[i] for i in eucl]
cols.sort(key=lambda name: name[-9:])
oct_scores.boxplot(column= cols, rot= 45)

cols = [scores_names[i] for i in ssim]+[scores_top_names[i] for i in ssim]
cols.sort(key=lambda name: name[-9:])
oct_scores.boxplot(column= cols, rot= 45)


# #### Results:
#
# all distance measures:
# * distances for the VGG16-based NHNM are higher than for the corresponding CNN-based NHNM -> as expected; difference not as prominent as in MNIST
# * distances of near hits behave about the same as distances of near misses.
# * distances for misses are higher than for hits as expected; exception: 1000_ssim , 1010_euclidean and 1011_ssim, here distances for misses are lower; however difference it not as prominent as in MNIST -> might need more data
# * weird: 1000 != 1001 (unlike for MNIST)
#
# euclidean distances:
# * euclidean distances for raw images are the highest -> to be expected since euclidean distance might not be the optimal citerion here
# * euclidean distances for FE are a lot smaller than those of LRP -> to be expected since euclidean distance might not be the optimal citerion here
#
#
# SSIM:
# * SSIM on the LRP heatmaps are a smaller than on the raw images; this could be due to the fact that a lot more of the LRP heatmaps share the same color (grey) than the original image (black); -> difference not as prominent as on MNIST data
#

# ## Calculate Jaccard indices

def jaccard_nhnmtm(df1, df2, group=None, df1_name=None, df2_name=None):
    res = pd.DataFrame()
    # hits
    res["jaccard_hits"] = jaccard_df(df1.near_hits, df2.near_hits)
    res["jaccard_hits_abs"] = jaccard_df(df1.near_hits, df2.near_hits, "absolute")
    # misses
    res["jaccard_misses"] = jaccard_df(df1.near_misses, df2.near_misses)
    res["jaccard_misses_abs"] = jaccard_df(df1.near_misses, df2.near_misses, "absolute")
    # top misses
    res["jaccard_top_misses"] = jaccard_df(df1.top_misses, df2.top_misses)
    res["jaccard_top_misses_abs"] = jaccard_df(df1.top_misses, df2.top_misses, "absolute")
    # set group
    if group is not None:
        res["group"] = group
        res["df1_name"] = df1_name
        res["df2_name"] = df2_name
    return res


# +
# jaccards.groupby(["df1_name", "df2_name"]).describe()

def jaccards_heatmap(jaccards_df, column):
    """
    :param columns: str, one of "jaccard_misses", "jaccard_top_misses", "jaccard_hits", "jaccard_misses_abs", "jaccard_top_misses_abs", "jaccard_hits_abs"
    """
    mean_topmisses = jaccards_df[[column,
                                  "df1_name", "df2_name"]].groupby(["df1_name", "df2_name"]).median().unstack()

    fig, ax = plt.subplots(figsize=(14,14))
    sns.set(font_scale=1.3)
    ax = sns.heatmap(mean_topmisses, annot=True, fmt=".3f", cmap='viridis', square = True) # vmin=0.05, vmax=0.2, 
    ax.set(xlabel="df2", ylabel="df1")
    ax.xaxis.tick_top()
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.show()


# -

# ### MNIST

# +
# calculate Jaccard Indeces for every combination of DataFrames
jaccards = pd.DataFrame()
for m1, df1 in mnist_df.items():
    for m2, df2 in mnist_df.items():
        # print(m1+"_"+m2)
        if m1 == m2:  # if compare DataFrame to itself -> continue
            continue
        try:
            if m2+"_"+m1 in np.array(jaccards.group):  # since jaccard(A,B) == jaccard(B,A) we can speed up by skipping
                continue
        except AttributeError:  # if it's the first time we compare these DataFrames it's okay to have an error
            pass
        new = jaccard_nhnmtm(df1, df2, group = m1+"_"+m2, df1_name=m1, df2_name=m2)
        jaccards = pd.concat([jaccards, new], ignore_index = True)
        
jaccards
# -

jaccards_heatmap(jaccards, "jaccard_misses")
jaccards_heatmap(jaccards, "jaccard_top_misses")
jaccards_heatmap(jaccards, "jaccard_hits")

# #### Jaccard Indices

# raw
jaccards[((jaccards.df1_name.str[-9:-6]== "000") |
         (jaccards.df1_name.str[-7:-4] == "000")) &
        ((jaccards.df2_name.str[-9:-6]== "000") |
         (jaccards.df2_name.str[-7:-4] == "000"))].boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], 
                                                      by=["df1_name", "df2_name"], figsize=(30,30), rot = 90, fontsize=20)


# euclidean
# CNN
jaccards[(jaccards.df1_name.str[-6:]== "0_eucl") |
         (jaccards.df2_name.str[-6:] == "0_eucl")].boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], 
                                                      by=["df1_name", "df2_name"], figsize=(30,30), rot = 90, fontsize=20)
# VGG
jaccards[(jaccards.df1_name.str[-6:]== "1_eucl") |
         (jaccards.df2_name.str[-6:] == "1_eucl")].boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], 
                                                      by=["df1_name", "df2_name"], figsize=(30,30), rot = 90, fontsize=20)


# ssim
# CNN
jaccards[(jaccards.df1_name.str[-6:]== "0_ssim") |
         (jaccards.df2_name.str[-6:] == "0_ssim")].boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], 
                                                      by=["df1_name", "df2_name"], figsize=(30,30), rot = 90, fontsize=20)
# VGG
jaccards[(jaccards.df1_name.str[-6:]== "1_ssim") |
         (jaccards.df2_name.str[-6:] == "1_ssim")].boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], 
                                                      by=["df1_name", "df2_name"], figsize=(30,30), rot = 90, fontsize=20)


#cw ssim
# CNN
jaccards[(jaccards.df1_name.str[-4:]== "0_cw") |
         (jaccards.df2_name.str[-4:] == "0_cw")].boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], 
                                                      by=["df1_name", "df2_name"], figsize=(30,30), rot = 90, fontsize=20)
# VGG
jaccards[(jaccards.df1_name.str[-4:]== "1_cw") |
         (jaccards.df2_name.str[-4:] == "1_cw")].boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], 
                                                      by=["df1_name", "df2_name"], figsize=(30,30), rot = 90, fontsize=20)


jaccards.boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], by=["df1_name", "df2_name"],
                figsize=(30,30), rot = 90, fontsize=20)

# #### Absolute intersection
# Reminder: the maximum absolute number is #NHNM for the hits and #NHNM * (#classes - 1) for misses

jaccards.boxplot(column=["jaccard_misses_abs", "jaccard_top_misses_abs", "jaccard_hits_abs"], by=["df1_name", "df2_name"],
                figsize=(30,30), rot = 90, fontsize=20)

# #### Result:
#
# all distances:
# * Since Jaccard index is a measure of similarity 0 means that A nad B are totally different
# * top 15 misses yield higher jaccard indices than top misses -> to be expected since more variability in top15 allowed than in top misses. here the misses have to match per category
# * not surprisingly distances on the raw images produce same NHNM for CNN and VGG -> model is not used here anyway
#
# euclidean:
# * 0010_eucl to 0010_ssim yields the most similar results in the top 15 missed, same goes for top misses
# * 0101_eucl is pretty similar to all the rawData hits/misses -> FE of VGG gives not much different information than raw image data
# * results for VGG are different than those for CNN: 0101_eucl produces similar results to all rawData models (ssim and cw) -> 0101_eucl (FE) does not give much different information than the rawData; not surpising since this was run on VGG and thus not much of an information gain was to be expected
# * no similarity can be seen when looking at 0010_eucl/0010_ssim to 0011_eucl which was to be expected -> CNN gives different information than general VGG; same goes for 010_eucl to 0101_eucl

# ### OCT

# +
# calculate Jaccard Indeces for every combination of DataFrames
jaccards_oct = pd.DataFrame()
for m1, df1 in oct_df.items():
    for m2, df2 in oct_df.items():
        # print(m1+"_"+m2)
        if m1 == m2:  # if compare DataFrame to itself -> continue
            continue
        try:
            if m2 + "_" + m1 in np.array(jaccards_oct.group):  # since jaccard(A,B) == jaccard(B,A) we can speed up by skipping
                continue
        except AttributeError:  # if it's the first time we compare these DataFrames it's okay to have an error
            pass
        new = jaccard_nhnmtm(df1, df2, group=m1 + "_" + m2, df1_name=m1, df2_name=m2)
        jaccards_oct = pd.concat([jaccards_oct, new], ignore_index=True)

jaccards_oct
# -

jaccards_oct.boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], by=["df1_name", "df2_name"],
                 figsize=(30, 30), rot=90, fontsize=20)

jaccards_oct.boxplot(column=["jaccard_misses_abs", "jaccard_top_misses_abs", "jaccard_hits_abs"],
                 by=["df1_name", "df2_name"],
                 figsize=(30, 30), rot=90, fontsize=20)


# #### Results:
# old:
# * Similar to mnist, but more prominent
# * Except for the SSIM-threshold which seems to not give similar results to any of the other metrics. Only for the overall misses there are some overlaps to SSIM
# * This is different to the mnist dataset

plt.close("all")

# ## Plot certain images + maps

def plot_nhnm_overview_from_input(lst,
                                  dataset, suffix_path="_multicnn", type_of_model="cnn", distance_measure='cosine',
                                  top_n=TOP_N_NMNH, use_prediction=True, raw=False, distance_on_image=False
                                  ):
    """
    list should have 
    0 path to test image
    1 list with paths to near hits
    2 list with scores of near hits
    3 list of lists with paths to near misses
    4 list of lists with scores of near misses
    Rest of input params like in near_miss_hits_selection.get_nhnm_overview()
    """

    from dataset import DataSet
    from modelsetup import ModelSetup, load_model_from_folder, get_output_layer
    from feature_extractor import FeatureExtractor
    from near_miss_hits_selection import plot_nmnh, plot_nmnh_heatmaps

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
        i = 0  # Counter for newly loaded feature embeddings
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

    test_img = DataEntry(fe, dataset, change_imgpath(lst[0]))
    img, x = fe.load_preprocess_img(test_img.img_path)

    pred_label = test_img.ground_truth_label

    if use_prediction:
        pred_label, pred_prob = sel_model.pred_test_img(test_img)
        print("Ground Truth: ", test_img.ground_truth_label)
        print("Prediction: ", pred_label)
        print("Probability: ", pred_prob)

    # near hits
    scores_nearest_hit = lst[2]
    ranked_nearest_hit_data_entry = [DataEntry(fe, dataset, change_imgpath(path)) for path in lst[1]]

    # near misses
    scores_nearest_miss_multi = lst[4]
    ranked_nearest_miss_multi_data_entry = [[DataEntry(fe, dataset, change_imgpath(path)) for path in list2] for list2 in lst[3]]

    # plot near hits and near misses
    plt.ioff()
    if top_n < 1:
        print("SCORES NEAREST HITS")
        print(pd.Series(scores_nearest_hit).describe())
    else:
        # Plot random image + heatmap

        heatmap_directory = os.path.join(STATIC_DIR, 'heatmaps', test_img.fe.fe_model.name, dataset)

        fig1 = plt.figure()
        plt.subplot(2, 1, 1)
        plt.title(f"{test_img.img_name}\nActual Label : {test_img.ground_truth_label}\nPredicted Label : {pred_label}",
                  weight='bold', size=10, loc="left")
        plt.imshow(img, cmap='gray')
        if distance_on_image and not raw:
            # get correct path to heatmap
            test_image_name = str.split(test_img.img_name, ".")[0]
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
            fig3 = plot_nmnh_heatmaps(ranked_nearest_hit_data_entry, scores_nearest_hit, pred_label[0],
                                      title="Near Hits Heatmaps")
            fig3.savefig("fig3.png", bbox_inches='tight')
            plt.close()

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

        # concatenate all pictures into one with cv2 and save it
        # load pictures
        pic1 = cv2.imread("fig1.png", cv2.IMREAD_GRAYSCALE)
        pic2 = cv2.imread("fig2.png", cv2.IMREAD_GRAYSCALE)
        try:
            pic3 = cv2.imread("fig3.png", cv2.IMREAD_GRAYSCALE)
            pic23 = cv2.hconcat([pic2, pic3])
        except TypeError:  # FileNotFoundError
            pic23 = pic2
        pic4 = cv2.imread("fig4.png", cv2.IMREAD_GRAYSCALE)
        try:
            pic5 = cv2.imread("fig5.png", cv2.IMREAD_GRAYSCALE)
            pic45 = cv2.hconcat([pic4, pic5])
        except TypeError:  # FileNotFoundError
            pic45 = pic4

        # concatenate them
        pic_all = hconcat_resize_max([pic1, vconcat_resize_min([pic23, pic45])])
        save_path = STATIC_DIR+"/"+test_img.img_name+"_"+distance_measure+"_eval.png"
        cv2.imwrite(save_path, pic_all)


if __name__ == '__main__':
    distance = "SSIM-blur"

    for i in range(0, 21):
        plot_nhnm_overview_from_input(list(oct_df[distance].iloc[i]),
                                          "oct_cc", suffix_path="_cnn_seed3871", type_of_model="cnn", distance_measure=distance,
                                          top_n=TOP_N_NMNH, use_prediction=True, raw=False, distance_on_image=True)




