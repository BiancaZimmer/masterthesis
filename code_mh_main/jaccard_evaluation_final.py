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
import scipy.stats as stats

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
m0011_eucl = m0011_ssim = m0011_cw = []
m0100_eucl = m0101_eucl = []

o1000_eucl = o1000_ssim = []
o1001_eucl = o1001_ssim = []
o1010_eucl = o1010_ssim = []
o1011_eucl = o1011_ssim = []
o1100_eucl = o1101_eucl = []

all_df = {"0000_eucl": m0000_eucl, "0000_ssim": m0000_ssim, "0000_cw": m0000_cw,
          "0001_eucl": m0001_eucl, "0001_ssim": m0001_ssim, "0001_cw": m0001_cw,
          "0010_eucl": m0010_eucl, "0010_ssim": m0010_ssim, "0010_cw": m0010_cw,
          "0011_eucl": m0011_eucl, "0011_ssim": m0011_ssim, "0011_cw": m0011_cw,
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
          "0011_eucl", "0011_ssim", "0011_cw",
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
                 "_FINAL100.pickle"
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
eucl = [0, 3, 6, 9, 12, 13]
ssim = [1, 4, 7, 10]
cw = [2, 5, 8, 11]

scores_top_names

# +
mnist_scores = pd.DataFrame()

for score_hit, score_top, df in zip(scores_names, scores_top_names, mnist_df.values()):
    mnist_scores[score_hit] = list(chain.from_iterable(df.scores_hits))
    mnist_scores[score_top] = list(chain.from_iterable(df.scores_top_misses))
mnist_scores
# -

mnist_scores_long = mnist_scores.stack().reset_index()
mnist_scores_long.columns = ["id", "group", "value"]
mnist_scores_long["nhnm"] = [t.split("_")[-3] for t in mnist_scores_long["group"]]
mnist_scores_long["number"] = [t.split("_")[-2] for t in mnist_scores_long["group"]]
mnist_scores_long["distance"] = [t.split("_")[-1] for t in mnist_scores_long["group"]]
mnist_scores_long

mnist_scores.describe().transpose()

# #### Distances for Hits

cols = [scores_names[i] for i in eucl]
sns.boxplot(data=mnist_scores[cols], palette="pastel", width=0.5)
plt.xticks(ticks=range(0, len(cols)+1),
           labels=[name.split("_",2)[2] for name in cols],
           rotation=20, ha="right")
plt.title("Near Hits - Distance Comparison - Euclidean")
plt.show

sns.boxplot(x=mnist_scores_long["number"][mnist_scores_long["distance"] != "eucl"],
            y=mnist_scores_long["value"][(mnist_scores_long["nhnm"] == "hit") &
                                         (mnist_scores_long["distance"] != "eucl")],
            palette="pastel", width=0.5,
            hue=mnist_scores_long["distance"][mnist_scores_long["distance"] != "eucl"])
plt.title("Near Misses - Distance Comparison - SSIM & CW")
plt.show

# #### Distances for Top Misses

cols = [scores_top_names[i] for i in eucl]
sns.boxplot(data=mnist_scores[cols], palette="pastel", width=0.5)
plt.xticks(ticks=range(0, len(cols)+1),
           labels=[name.split("_",3)[3] for name in cols],
           rotation=20, ha="right")
plt.title("Near Misses - Distance Comparison - Euclidean")
plt.show

sns.boxplot(x=mnist_scores_long["number"][mnist_scores_long["distance"] != "eucl"],
            y=mnist_scores_long["value"][(mnist_scores_long["nhnm"] == "misses") &
                                         (mnist_scores_long["distance"] != "eucl")],
            palette="pastel", width=0.5,
            hue=mnist_scores_long["distance"][mnist_scores_long["distance"] != "eucl"])
plt.title("Near Misses - Distance Comparison - SSIM & CW")
plt.show

# #### Compare misses + hits

mnist_scores.describe().transpose()[["mean","50%"]]

sns.boxplot(x=mnist_scores_long["number"], y=mnist_scores_long["value"][mnist_scores_long["distance"] == "eucl"],
            palette="pastel", width=0.5,
            hue=mnist_scores_long["nhnm"])
plt.title("NHNM - Distance Comparison - Euclidean")
plt.show

sns.boxplot(x=mnist_scores_long["number"][mnist_scores_long["distance"] == "ssim"],
            y=mnist_scores_long["value"][mnist_scores_long["distance"] == "ssim"],
            palette="pastel", width=0.5,
            hue=mnist_scores_long["nhnm"])
plt.title("NHNM - Distance Comparison - Euclidean")
plt.show

sns.boxplot(x=mnist_scores_long["number"][mnist_scores_long["distance"] == "cw"],
            y=mnist_scores_long["value"][mnist_scores_long["distance"] == "cw"],
            palette="pastel", width=0.5,
            hue=mnist_scores_long["nhnm"])
plt.title("NHNM - Distance Comparison - CW-SSIM")
plt.show

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

oct_scores_long = oct_scores.stack().reset_index()
oct_scores_long.columns = ["id", "group", "value"]
oct_scores_long["nhnm"] = [t.split("_")[-3] for t in oct_scores_long["group"]]
oct_scores_long["number"] = [t.split("_")[-2] for t in oct_scores_long["group"]]
oct_scores_long["distance"] = [t.split("_")[-1] for t in oct_scores_long["group"]]
oct_scores_long

oct_scores.describe().transpose()

oct_df["1000_eucl"]["image_name"].equals(oct_df["1001_eucl"]["image_name"])

# #### Hits comparison

cols = [scores_names[i] for i in eucl]
sns.boxplot(data=oct_scores[cols], palette="pastel", width=0.5)
plt.xticks(ticks=range(0, len(cols)+1),
           labels=[name.split("_")[2] for name in cols])
plt.title("Near Hits - Comparison - Euclidean")
plt.show

# same plot as above - different layout
sns.boxplot(x=oct_scores_long["number"][oct_scores_long["distance"] == "eucl"],
            y=oct_scores_long["value"][(oct_scores_long["nhnm"] == "hit") &
                                         (oct_scores_long["distance"] == "eucl")],
            palette="pastel", width=0.5,
            hue=oct_scores_long["distance"][oct_scores_long["distance"] == "eucl"])
plt.title("Near Hits - Comparison - Euclidean")
plt.show

sns.boxplot(x=oct_scores_long["number"][oct_scores_long["distance"] != "eucl"],
            y=oct_scores_long["value"][(oct_scores_long["nhnm"] == "hit") &
                                         (oct_scores_long["distance"] != "eucl")],
            palette="pastel", width=0.5,
            hue=oct_scores_long["distance"][oct_scores_long["distance"] != "eucl"])
plt.title("Near Hits - Comparison - SSIM")
plt.show

# #### Misses Comparison

sns.boxplot(x=oct_scores_long["number"][oct_scores_long["distance"] == "eucl"],
            y=oct_scores_long["value"][(oct_scores_long["nhnm"] == "misses") &
                                         (oct_scores_long["distance"] == "eucl")],
            palette="pastel", width=0.5,
            hue=oct_scores_long["distance"][oct_scores_long["distance"] == "eucl"])
plt.title("Near Misses - Comparison - Euclidean")
plt.show

sns.boxplot(x=oct_scores_long["number"][oct_scores_long["distance"] != "eucl"],
            y=oct_scores_long["value"][(oct_scores_long["nhnm"] == "misses") &
                                         (oct_scores_long["distance"] != "eucl")],
            palette="pastel", width=0.5,
            hue=oct_scores_long["distance"][oct_scores_long["distance"] != "eucl"])
plt.title("Near Misses - Comparison - SSIM")
plt.show

# #### Compare misses+hits

oct_scores.describe().transpose()[["mean","50%"]]

sns.boxplot(x=oct_scores_long["number"][oct_scores_long["distance"] == "eucl"],
            y=oct_scores_long["value"][oct_scores_long["distance"] == "eucl"],
            palette="pastel", width=0.5,
            hue=oct_scores_long["nhnm"])
plt.title("NHNM Comparison - Euclidean")
plt.show

sns.boxplot(x=oct_scores_long["number"][oct_scores_long["distance"] == "ssim"],
            y=oct_scores_long["value"][oct_scores_long["distance"] == "ssim"],
            palette="pastel", width=0.5,
            hue=oct_scores_long["nhnm"])
plt.title("NHNM Comparison - SSIM")
plt.show


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

def jaccards_heatmap(jaccards_df, column, title = None, vmin=None, vmax=None, metric="median", q=0.5):
    """
    :param column: str, one of "jaccard_misses", "jaccard_top_misses", "jaccard_hits", "jaccard_misses_abs", "jaccard_top_misses_abs", "jaccard_hits_abs"
    """
    # prepare data + sort into triangulart matrix
    if metric == "mean":
        mean_topmisses = jaccards_df[[column,
                                      "df1_name", "df2_name"]].groupby(["df1_name", "df2_name"]).mean().unstack()
    elif metric == "quantile":
        metric = str(q*100) + "%-quantile"
        mean_topmisses = jaccards_df[[column,
                                      "df1_name", "df2_name"]].groupby(["df1_name", "df2_name"]).quantile(q).unstack()
    else:
        metric = "median"
        mean_topmisses = jaccards_df[[column,
                                      "df1_name", "df2_name"]].groupby(["df1_name", "df2_name"]).quantile(0.5).unstack()
    mean_topmisses.columns = mean_topmisses.columns.get_level_values(1)
    mean_topmisses = sort_triangular(mean_topmisses)
    
    # set figure size + font size
    fig, ax = plt.subplots(figsize=(20,20))
    sns.set(font_scale=1.6)
    # set boundary values for legend
    if (vmin is None) and (vmax is None):
        ax = sns.heatmap(mean_topmisses, annot=True, fmt=".3f", cmap='viridis', square = True)
    else:
        ax = sns.heatmap(mean_topmisses, annot=True, fmt=".3f", cmap='viridis', square = True,
                        vmin=vmin, vmax=vmax)
    # set axes label, put them on top of the figure and rotate them
    ax.set(xlabel="df2", ylabel="df1")
    ax.xaxis.tick_top()
    texts = [t for t in ax.get_xticklabels()]
    plt.xticks(ticks=np.arange(0, len(texts))+0.5,
           labels=texts,
           rotation=90, ha="center")
    # set title
    if title is None:
        title = column + " - " + metric
    plt.title(title)
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

jaccards.groupby(["df1_name", "df2_name"]).describe() #.transpose()

jaccards_heatmap(jaccards, "jaccard_misses")
jaccards_heatmap(jaccards, "jaccard_top_misses")
jaccards_heatmap(jaccards, "jaccard_hits")

# #### Jaccard Indices

sns.reset_defaults()

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

# #### Statistical tests

# t-test
c = 0
for m1 in jaccards.group.value_counts().index:
    _, p = stats.ttest_1samp(a=jaccards[jaccards.group == m1]["jaccard_misses"], popmean=0)
    if p > 0.01:  # p value too big, these are not different from an all 0 distribution
        print(m1)
        print(p)
        c += 1
print(c)

# comparison to all 0 vector
tmp = [0] * 100
c = 0
for m1 in jaccards.group.value_counts().index:
    try:
        _, p = stats.wilcoxon(jaccards[jaccards.group == m1]["jaccard_misses"],tmp)
        if p > 0.005:  # p value too big, these are not different from an all 0 distribution
            print(m1)
            print(p)
            c += 1
    except ValueError:
        print(m1)
        print("all values 0")
        c += 1
print(c)

# #### Result:
#
# all distances:
# * Since Jaccard index is a measure of similarity 0 means that A nad B are totally different
# * top 15 misses yield higher jaccard indices than top misses -> to be expected since more variability in top15 allowed than in top misses. here the misses have to match per category
# * not surprisingly distances on the raw images produce same NHNM for CNN and VGG -> model is not used here anyway
# * all combinations yield Jaccard Indices pretty close to 0 -> all combinations seem to yield different results
# * conducting a Wilcoxon Rank Test to a vector of 100*0, alpha = 0.005 (corrected for multiple 91x testing) we can see that in 15 cases we can not reject the Nullhypothesis that the JaccardIndices are indeed 0.
# * although t-Test normality distribution assumption not met - 13 cases we can not reject the Nullhypothesis that the JaccardIndices are indeed 0
#
# euclidean:
# * 0010_eucl to 0010_ssim yields the most similar results in the top 15 missed, same goes for top misses
# * 0101_eucl is pretty similar to all the rawData hits/misses -> FE of VGG gives not much different information than raw image data
# * results for VGG are different than those for CNN: 0101_eucl produces similar results to all rawData models (ssim and cw) -> 0101_eucl (FE) does not give much different information than the rawData; not surpising since this was run on VGG and thus not much of an information gain was to be expected
# * no similarity can be seen when looking at 0010_eucl/0010_ssim to 0011_eucl which was to be expected -> CNN gives different information than general VGG; same goes for 010_eucl to 0101_eucl
#
# ## TODO!!
#

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

# jaccards_oct.describe()
jaccards_oct.groupby(["df1_name", "df2_name"]).describe()# .transpose()

oct_df["1000_eucl"]["image_name"].equals(oct_df["1000_ssim"]["image_name"])

jaccards_heatmap(jaccards_oct, "jaccard_misses")
jaccards_heatmap(jaccards_oct, "jaccard_top_misses")
jaccards_heatmap(jaccards_oct, "jaccard_hits")

jaccards_heatmap(jaccards_oct, "jaccard_misses", metric = "quantile", q=0.9)
jaccards_heatmap(jaccards_oct, "jaccard_top_misses", metric = "quantile", q=0.9)
jaccards_heatmap(jaccards_oct, "jaccard_hits", metric = "quantile", q=0.9)

sns.reset_defaults()

jaccards_oct.boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"], by=["df1_name", "df2_name"],
                 figsize=(30, 30), rot=90, fontsize=20)

jaccards_oct.boxplot(column=["jaccard_misses_abs", "jaccard_top_misses_abs", "jaccard_hits_abs"],
                 by=["df1_name", "df2_name"],
                 figsize=(30, 30), rot=90, fontsize=20)


# #### Statistical tests

# t-test
c = 0
for m1 in jaccards_oct.group.value_counts().index:
    _, p = stats.ttest_1samp(a=jaccards_oct[jaccards_oct.group == m1]["jaccard_misses"], popmean=0)
    if p > 0.01:  # p value too big, these are not different from an all 0 distribution
        print(m1)
        print(p)
        c += 1
print(c)

# comparison to all 0 vector
tmp = [0] * 100
c = 0
for m1 in jaccards_oct.group.value_counts().index:
    try:
        _, p = stats.wilcoxon(jaccards_oct[jaccards_oct.group == m1]["jaccard_misses"],tmp)
        if p > 0.005:  # p value too big, these are not different from an all 0 distribution
            print(m1)
            print(p)
            c += 1
    except ValueError:
        print(m1)
        print("all values 0")
        c += 1
print(c)

# #### Results:
# all distances:
#
# same as for MNIST:
# * Since Jaccard index is a measure of similarity 0 means that A nad B are totally different
# * not surprisingly distances on the raw images produce same NHNM for CNN and VGG -> model is not used here anyway
#
# different from MNIST:
# * top 15 misses DO NOT yield higher jaccard indices than top misses in general
# * all combinations have Jaccard Indices close to 0. This might have two reasons:
#     * A lot more training samples to choose from (73k for oct vs 22k for mnist) -> more possibilities mean more possibile variety -> solution would be to first cluster the training examples and only take the near miss/hits from here
#     * not enough test samples evaluated. 100 might not be enough
# * if anything then the raw images evaluated with SSIM have some little points in common with the Euclidean distance on FE
#
# -> Qualitative evaluation needed

plt.close("all")


