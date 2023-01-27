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
    if type(path) is str:
        return path.split("/")[-1]
    elif type(path[0]) is str:
        return [entry.split("/")[-1] for entry in path]
    else:
        return [[entry.split("/")[-1] for entry in lst] for lst in path]


def show_humanfriendly(df, columns=["image_name", "near_hits", "near_misses", "top_misses"]):
    dfh = df.copy()
    for c in columns:
        dfh[c] = [humanfriendly_trafo(row) for row in df[c]]
    return dfh


# -

def jaccard_df(df1, df2, method="intersection"):
    if type(df1[0][0]) is list:
        result = [jaccard(list(chain.from_iterable(l1)), list(chain.from_iterable(l2)), method) for
                  l1, l2 in zip(df1, df2)]
        return result
    else:
        result = [jaccard(l1, l2, method) for l1, l2 in zip(df1, df2)]
        return result


# ## Load pickles

# +
path_base = "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/NHNM/"

mnist_eucl = pd.read_pickle(path_base+"mnist_1247_cnn_seed3871_euclidean_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
mnist_SSIM = pd.read_pickle(path_base+"mnist_1247_cnn_seed3871_SSIM_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
mnist_SSIM_mm = pd.read_pickle(path_base+"mnist_1247_cnn_seed3871_SSIM-mm_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
mnist_SSIM_pushed = pd.read_pickle(path_base+"mnist_1247_cnn_seed3871_SSIM-pushed_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
mnist_SSIM_blur = pd.read_pickle(path_base+"mnist_1247_cnn_seed3871_SSIM-blur_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
mnist_CW_SSIM = pd.read_pickle(path_base+"mnist_1247_cnn_seed3871_CW-SSIM_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
mnist_SSIM_threshold = pd.read_pickle(path_base+"mnist_1247_cnn_seed3871_SSIM-threshold_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")


oct_eucl = pd.read_pickle(path_base+"oct_cc_cnn_seed3871_euclidean_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
oct_SSIM = pd.read_pickle(path_base+"oct_cc_cnn_seed3871_SSIM_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
oct_SSIM_mm = pd.read_pickle(path_base+"oct_cc_cnn_seed3871_SSIM-mm_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
oct_SSIM_pushed = pd.read_pickle(path_base+"oct_cc_cnn_seed3871_SSIM-pushed_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
oct_SSIM_blur = pd.read_pickle(path_base+"oct_cc_cnn_seed3871_SSIM-blur_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
oct_SSIM_threshold = pd.read_pickle(path_base+"oct_cc_cnn_seed3871_SSIM-threshold_usepredTrue_rawFalse_distonimgTrue_FINAL100.pickle")
# -

all_df = [mnist_eucl, mnist_SSIM, mnist_SSIM_mm, mnist_SSIM_pushed, mnist_SSIM_blur, mnist_SSIM_threshold, mnist_CW_SSIM,
          oct_eucl, oct_SSIM, oct_SSIM_mm, oct_SSIM_pushed, oct_SSIM_blur, oct_SSIM_threshold]
mnist_df = {"euclidean": mnist_eucl, "SSIM": mnist_SSIM, "SSIM-mm": mnist_SSIM_mm,
            "SSIM-pushed": mnist_SSIM_pushed, "SSIM-blur": mnist_SSIM_blur, "SSIM-threshold": mnist_SSIM_threshold,
            "CW-SSIM": mnist_CW_SSIM}
oct_df = {"euclidean": oct_eucl, "SSIM": oct_SSIM, "SSIM-mm": oct_SSIM_mm, 
          "SSIM-pushed": oct_SSIM_pushed, "SSIM-blur": oct_SSIM_blur, "SSIM-threshold": oct_SSIM_threshold}

# ## Overview over distance scores

# +
# metrics
metrics = ["eucl", "SSIM", "SSIM-mm", "SSIM-pushed", "SSIM-blur", "SSIM-threshold", "CW-SSIM"]

# column names
scores_names = []
scores_top_names = []
for m in metrics:
    scores_names.append("scores_hit_"+m)
    scores_top_names.append("scores_top_misses_"+m)
scores_top_names
# -

# ### MNIST

# +
mnist_scores = pd.DataFrame()


for score_hit, score_top, df in zip(scores_names, scores_top_names, mnist_df.values()):
    mnist_scores[score_hit] = list(chain.from_iterable(df.scores_hits))
    mnist_scores[score_top] = list(chain.from_iterable(df.scores_top_misses))
mnist_scores
# -

mnist_scores.describe().transpose()

sns.boxplot(data=mnist_scores[scores_names], palette="pastel", width=0.5)
plt.xticks(ticks=range(0, len(scores_names)+1),
           labels=[name.split("_")[2] for name in scores_names],
           rotation=20, ha="right")
plt.title("Near Hits - Distance Comparison")
plt.show

sns.boxplot(data=mnist_scores[scores_names[1:]], palette="pastel", width=0.5)
plt.xticks(ticks=range(0, len(scores_names[1:])+1),
           labels=[name.split("_")[2] for name in scores_names[1:]],
           rotation=20, ha="right")
plt.ylim([0,0.6])
plt.title("MNIST Near Hits - Distance Comparison")
plt.show

sns.boxplot(data=mnist_scores[scores_top_names[1:]], palette="pastel", width=0.5)
plt.xticks(ticks=range(0, len(scores_top_names[1:])+1),
           labels=[name.split("_")[3] for name in scores_top_names[1:]],
           rotation=20, ha="right")
plt.ylim([0,0.6])
plt.title("MNIST Near Misses - Distance Comparison")
plt.show

# #### Results:
#
# * euclidean distance seems to be lower in Misses than in Hits, which is contraintuitive
# * In all SSIM distances the Misses have higher distance values than the Hits - which is what we expected
# * SSIM seems to be fairly good compared to the minmax metric, the CW-SSIM and the transformed/pushed metric
# * SSIM and SSIM-threshold give very similar results
# * SSIM on the blurred + transformed pictures seems to give the best results since the distances between the pictures are minimal (close to 0)

# ### OCT

# +
oct_scores = pd.DataFrame()


for score_hit, score_top, df in zip(scores_names[:-1], scores_top_names[:-1], oct_df.values()):
    oct_scores[score_hit] = list(chain.from_iterable(df.scores_hits))
    oct_scores[score_top] = list(chain.from_iterable(df.scores_top_misses))
oct_scores
# -

oct_scores.describe().transpose()

sns.boxplot(data=oct_scores[scores_names[1:-1]], palette="pastel", width=0.5)
plt.xticks(ticks=range(0, len(scores_names[1:-1])+1),
           labels=[name.split("_")[2] for name in scores_names[1:-1]],
           rotation=20, ha="right")
plt.ylim([0,0.6])
plt.title("OCT Near Hits - Distance Comparison")
plt.show

# oct_scores.boxplot(column=scores_names[1:-1], rot= 10)

sns.boxplot(data=oct_scores[scores_top_names[:-1]], palette="pastel", width=0.5)
plt.xticks(ticks=range(0, len(scores_top_names[:-1])+1),
           labels=[name.split("_")[3] for name in scores_top_names[:-1]],
           rotation=20, ha="right")
plt.title("OCT Near Misses - Distance Comparison")
plt.show

sns.boxplot(data=oct_scores[scores_top_names[1:-1]], palette="pastel", width=0.5)
plt.xticks(ticks=range(0, len(scores_top_names[1:-1])+1),
           labels=[name.split("_")[3] for name in scores_top_names[1:-1]],
           rotation=20, ha="right")
plt.ylim([0,0.6])
plt.title("OCT Near Misses - Distance Comparison")
plt.show


# #### Results:
# Similar to those of the MNIST dataset
#
# Probably need more values to varify this but runtime too long

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


def sort_triangular(matrix):
    """ Takes matrix/DataFrame as input and sorts columns and rows into an upper triangular matrix if rest is NAN

    :param matrix: pandas.DataFrame
    :return: pandas.DataFrame with sorted columns and rows
    """
    nan_count_c = matrix.apply(lambda x: x.isna().sum(), axis=0)
    nan_count_r = matrix.apply(lambda x: x.isna().sum(), axis=1)
    triangular = matrix.iloc[np.argsort(nan_count_r), np.argsort(-nan_count_c)]
    return triangular


# +
# jaccards.groupby(["df1_name", "df2_name"]).describe()

def jaccards_heatmap(jaccards_df, column, title = None, vmin=None, vmax=None):
    """
    :param columns: str, one of "jaccard_misses", "jaccard_top_misses", "jaccard_hits", "jaccard_misses_abs", "jaccard_top_misses_abs", "jaccard_hits_abs"
    """
    mean_topmisses = jaccards_df[[column,
                                  "df1_name", "df2_name"]].groupby(["df1_name", "df2_name"]).median().unstack()
    mean_topmisses.columns = mean_topmisses.columns.get_level_values(1)
    mean_topmisses = sort_triangular(mean_topmisses)
    
    fig, ax = plt.subplots(figsize=(10,10))
    sns.set(font_scale=1.3)
    if (vmin is None) and (vmax is None):
        ax = sns.heatmap(mean_topmisses, annot=True, fmt=".3f", cmap='viridis', square = True)
    else:
        ax = sns.heatmap(mean_topmisses, annot=True, fmt=".3f", cmap='viridis', square = True,
                        vmin=vmin, vmax=vmax)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    texts = [t for t in ax.get_xticklabels()]
    plt.xticks(ticks=np.arange(0, len(texts))+0.5,
           labels=texts,
           rotation=90, ha="center")
    if title is None:
        title = column
    plt.title(title)
    plt.show()


# -

# ### MNIST

# +
jaccards = pd.DataFrame()
for m1, df1 in mnist_df.items():
    for m2, df2 in mnist_df.items():
        # print(m1+"_"+m2)
        if m1 == m2:
            continue
        try:
            if m2+"_"+m1 in np.array(jaccards.group):
                continue
        except AttributeError:
            pass
        new = jaccard_nhnmtm(df1, df2, group = m1+"_"+m2, df1_name=m1, df2_name=m2)
        jaccards = pd.concat([jaccards, new], ignore_index = True)
        
jaccards
# -

jaccards_heatmap(jaccards, "jaccard_misses", vmax=1, title = "Jaccard Indices for MNIST Near Misses per Class")
jaccards_heatmap(jaccards, "jaccard_top_misses", vmax=1, title = "Jaccard Indices for MNIST Top 5 Near Misses")
jaccards_heatmap(jaccards, "jaccard_hits", vmax=1, title = "Jaccard Indices for MNIST Near Hits")

sns.reset_defaults()
jaccards.boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], by=["df1_name", "df2_name"],
                figsize=(30,30), rot = 90, fontsize=20)

jaccards.boxplot(column=["jaccard_misses_abs", "jaccard_top_misses_abs", "jaccard_hits_abs"], by=["df1_name", "df2_name"],
                figsize=(30,30), rot = 90, fontsize=20)

# #### Result:
# * Since Jaccard index is a measure of similarity 0 means that A nad B are totally different
# * Euclidean produces very different results from all SSIM metrics
# * Maybe pushed and minmax are closes to each other
# * SSIM threshold and SSIM produce similar results, this was to be expected when looking at the boxplots for the SSIM scores
# * The other metrics produce quite but not totally different results

# ### OCT

# +
jaccards_oct = pd.DataFrame()
for m1, df1 in oct_df.items():
    for m2, df2 in oct_df.items():
        # print(m1+"_"+m2)
        if m1 == m2:
            continue
        try:
            if m2 + "_" + m1 in np.array(jaccards_oct.group):
                continue
        except AttributeError:
            pass
        new = jaccard_nhnmtm(df1, df2, group=m1 + "_" + m2, df1_name=m1, df2_name=m2)
        jaccards_oct = pd.concat([jaccards_oct, new], ignore_index=True)

jaccards_oct
# -

jaccards_heatmap(jaccards_oct, "jaccard_misses", vmax=1, title = "Jaccard Indices for OCT Near Misses per Class")
jaccards_heatmap(jaccards_oct, "jaccard_top_misses", vmax=1, title = "Jaccard Indices for OCT Top 5 Near Misses")
jaccards_heatmap(jaccards_oct, "jaccard_hits", vmax=1, title = "Jaccard Indices for OCT Near Hits")

sns.reset_defaults()
jaccards_oct.boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], by=["df1_name", "df2_name"],
                 figsize=(30, 30), rot=90, fontsize=20)

jaccards_oct.boxplot(column=["jaccard_misses_abs", "jaccard_top_misses_abs", "jaccard_hits_abs"],
                 by=["df1_name", "df2_name"],
                 figsize=(30, 30), rot=90, fontsize=20)


# #### Results:
# * Similar to mnist, but more prominent
# * Except for the SSIM-threshold which seems to not give similar results to any of the other metrics. Only for the overall misses there are some overlaps to SSIM
# * This is different to the mnist dataset

plt.close("all")
