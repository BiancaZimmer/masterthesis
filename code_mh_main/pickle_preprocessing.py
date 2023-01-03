# -*- coding: utf-8 -*-
# This file was used to preprocess pickles generated with near_miss_hits_selection.py

# ## Imports

import math
import os
import time
import random
import numpy as np
import pandas as pd
from itertools import chain

# from feature_extractor import FeatureExtractor
from helpers import jaccard, change_imgpath_back
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

def combine_pickle(path, number_range):
    """
    Combines pickles
    :param path: path to pickle without number at the end
    :param number_range: number range which will be appended to the path with "_number"
    :return: combined pickle as pandas DataFrame
    """
    df = pd.DataFrame()
    for i in number_range:
        tmp = pd.read_pickle(path + "_" + str(i) + ".pickle")
        df = pd.concat([df, tmp], ignore_index=True)
    return df


# +
def humanfriendly_trafo(path):
    """
    Splits a path and only returns last entry which usually is the name of a file
    Also accepts lists and list of lists as input
    :param path: path to a file or list of paths or list of lists of paths
    :return: same data structure as input with truncated paths
    """
    if type(path) is str:
        return path.split("/")[-1]
    elif type(path[0]) is str:
        return [entry.split("/")[-1] for entry in path]
    else:
        return [[entry.split("/")[-1] for entry in lst] for lst in path]


def show_humanfriendly(df, columns=["image_name", "near_hits", "near_misses", "top_misses"]):
    """
    Returns a whole DataFrame without the longish paths in front. Uses pickle_preprocessing.humanfriendly_trafo() for this
    :param df: pandas DataFrame
    :param columns: name of columns to transform - should be paths
    :return: copy of pandas DataFrame
    """
    dfh = df.copy()
    for c in columns:
        dfh[c] = [humanfriendly_trafo(row) for row in df[c]]
    return dfh


# -

def top_misses(lst, score):
    """
    Calculates the top overall misses from the lists of the multidimensional misses
    :param lst: list of list with paths to the misses
    :param score: list of list with scores of the misses
    :return: tuple with TOP_N_NMNH entries for the misses
    """
    lst = list(chain.from_iterable(lst))
    score = list(chain.from_iterable(score))
    paths = [p for (p, s) in sorted(zip(lst, score))]
    scores = [s for (p, s) in sorted(zip(lst, score))]
    return paths[:TOP_N_NMNH], scores[:TOP_N_NMNH]


def add_top_misses(df):
    """
    Uses top_misses on a whole pandas DataFrame
    :param df: pandas Dataframe with one column named 'near_misses' and one named 'scores_misses' to be transformed
    :return: pandas DataFrame
    """
    temp = df.apply(lambda row: top_misses(row["near_misses"], row["scores_misses"]), axis=1)
    df["top_misses"] = [t[0] for t in temp]
    df["scores_top_misses"] = [t[1] for t in temp]
    return df


def jaccard_df(df1, df2, method="intersection"):
    """
    Calculates the Jaccard Index for two columns of a pandas DataFrame according to helpers.jaccard
    :param df1: first column of pandas DataFrame
    :param df2: second column of pandas DataFrame
    :param method: method according to helpers.jaccard()
    :return: list with jaccard indices
    """
    if type(df1[0][0]) is list:
        result = [jaccard(list(chain.from_iterable(l1)), list(chain.from_iterable(l2)), method) for
                  l1, l2 in zip(df1, df2)]
        return result
    else:
        result = [jaccard(l1, l2, method) for l1, l2 in zip(df1, df2)]
        return result


# ## Datasets for testing

mnist_eucl = pd.read_pickle(
    "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/mnist_1247_cnn_seed3871_euclidean_usepredTrue_rawFalse_distonimgTrue_100notrandom.pickle")
add_top_misses(mnist_eucl)
show_humanfriendly(mnist_eucl)

mnist_SSIM = pd.read_pickle(
    "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/mnist_1247_cnn_seed3871_SSIM_usepredTrue_rawFalse_distonimgTrue_100notrandom.pickle")
add_top_misses(mnist_SSIM)
show_humanfriendly(mnist_SSIM)

# # For metric comparison
# ## Combine pickles

# +
# dataset_to_use = "mnist_1247"

path_base = "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/NHNM/"

r = range(1, 21)

# mnist_1247_cnn_seed3871_euclidean_usepredTrue_rawFalse_distonimgTrue
# mnist_1247_cnn_seed3871_CW-SSIM_usepredTrue_rawFalse_distonimgTrue
# mnist_1247_cnn_seed3871_SSIM_usepredTrue_rawFalse_distonimgTrue
# mnist_1247_cnn_seed3871_SSIM-blur_usepredTrue_rawFalse_distonimgTrue
# mnist_1247_cnn_seed3871_SSIM-mm_usepredTrue_rawFalse_distonimgTrue
# mnist_1247_cnn_seed3871_SSIM-pushed_usepredTrue_rawFalse_distonimgTrue

# oct_cc_cnn_seed3871_euclidean_usepredTrue_rawFalse_distonimgTrue
# oct_cc_cnn_seed3871_SSIM_usepredTrue_rawFalse_distonimgTrue
# oct_cc_cnn_seed3871_SSIM-blur_usepredTrue_rawFalse_distonimgTrue
# oct_cc_cnn_seed3871_SSIM-mm_usepredTrue_rawFalse_distonimgTrue
# oct_cc_cnn_seed3871_SSIM-pushed_usepredTrue_rawFalse_distonimgTrue

mnist_eucl = combine_pickle(path_base + "mnist_1247_cnn_seed3871_euclidean_usepredTrue_rawFalse_distonimgTrue", r)
mnist_SSIM = combine_pickle(path_base + "mnist_1247_cnn_seed3871_SSIM_usepredTrue_rawFalse_distonimgTrue", r)
mnist_SSIM_mm = combine_pickle(path_base + "mnist_1247_cnn_seed3871_SSIM-mm_usepredTrue_rawFalse_distonimgTrue", r)
mnist_SSIM_pushed = combine_pickle(path_base + "mnist_1247_cnn_seed3871_SSIM-pushed_usepredTrue_rawFalse_distonimgTrue",
                                   r)
mnist_SSIM_blur = combine_pickle(path_base + "mnist_1247_cnn_seed3871_SSIM-blur_usepredTrue_rawFalse_distonimgTrue", r)
mnist_CW_SSIM = combine_pickle(path_base + "mnist_1247_cnn_seed3871_CW-SSIM_usepredTrue_rawFalse_distonimgTrue", r)
mnist_SSIM_threshold = combine_pickle(
    path_base + "mnist_1247_cnn_seed3871_SSIM-threshold_usepredTrue_rawFalse_distonimgTrue", r)

oct_eucl = combine_pickle(path_base + "oct_cc_cnn_seed3871_euclidean_usepredTrue_rawFalse_distonimgTrue", r)
oct_SSIM = combine_pickle(path_base + "oct_cc_cnn_seed3871_SSIM_usepredTrue_rawFalse_distonimgTrue", r)
oct_SSIM_mm = combine_pickle(path_base + "oct_cc_cnn_seed3871_SSIM-mm_usepredTrue_rawFalse_distonimgTrue", r)
oct_SSIM_pushed = combine_pickle(path_base + "oct_cc_cnn_seed3871_SSIM-pushed_usepredTrue_rawFalse_distonimgTrue", r)
oct_SSIM_blur = combine_pickle(path_base + "oct_cc_cnn_seed3871_SSIM-blur_usepredTrue_rawFalse_distonimgTrue", r)
oct_SSIM_threshold = combine_pickle(path_base + "oct_cc_cnn_seed3871_SSIM-threshold_usepredTrue_rawFalse_distonimgTrue",
                                    r)

# +
# OCT-threshold was run on a different computer -> different paths
# paths will be transformed

path_columns = ['image_name', 'near_hits', 'near_misses']


oct_SSIM_threshold["image_name"] = [change_imgpath_back(path) for path in oct_SSIM_threshold["image_name"]]
oct_SSIM_threshold["near_hits"] = [[change_imgpath_back(path) for path in lst] for lst in
                                   oct_SSIM_threshold["near_hits"]]
oct_SSIM_threshold["near_misses"] = [[[change_imgpath_back(path) for path in lst2] for lst2 in lst] for lst in
                                     oct_SSIM_threshold["near_misses"]]

oct_SSIM_threshold
# -

all_df = [mnist_eucl, mnist_SSIM, mnist_SSIM_mm, mnist_SSIM_pushed, mnist_SSIM_blur, mnist_SSIM_threshold,
          mnist_CW_SSIM,
          oct_eucl, oct_SSIM, oct_SSIM_mm, oct_SSIM_pushed, oct_SSIM_blur, oct_SSIM_threshold]
mnist_df = {"euclidean": mnist_eucl, "SSIM": mnist_SSIM, "SSIM-mm": mnist_SSIM_mm,
            "SSIM-pushed": mnist_SSIM_pushed, "SSIM-blur": mnist_SSIM_blur, "SSIM-threshold": mnist_SSIM_threshold,
            "CW-SSIM": mnist_CW_SSIM}
oct_df = {"euclidean": oct_eucl, "SSIM": oct_SSIM, "SSIM-mm": oct_SSIM_mm,
          "SSIM-pushed": oct_SSIM_pushed, "SSIM-blur": oct_SSIM_blur, "SSIM-threshold": oct_SSIM_threshold}

# ## Add top Misses for all df

for df in all_df:
    add_top_misses(df)

# ## Save new pickles

# +
for df in mnist_df:
    picklepath = "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/NHNM/mnist_1247_cnn_seed3871_" + \
                 df + "_usepredTrue_rawFalse_distonimgTrue_FINAL100"
    print(picklepath)
    mnist_df[df].to_pickle(picklepath + ".pickle")

for df in oct_df:
    picklepath = "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/NHNM/oct_cc_cnn_seed3871_" + \
                 df + "_usepredTrue_rawFalse_distonimgTrue_FINAL100"
    print(picklepath)
    oct_df[df].to_pickle(picklepath + ".pickle")
# -

# # For final run
# ## Combine Pickles

# +
path_base = "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/NHNM/"

r = range(1, 10)

# 0000_mnist_1247_cnn_seed3871_euclidean
# 0000_mnist_1247_cnn_seed3871_SSIM
# 0000_mnist_1247_cnn_seed3871_CW-SSIM

# 0001_mnist_1247_euclidean
# 0001_mnist_1247_SSIM
# 0001_mnist_1247_CW-SSIM

# 0010_mnist_1247_cnn_seed3871_euclidean
# 0010_mnist_1247_cnn_seed3871_SSIM
# 0010_mnist_1247_cnn_seed3871_CW-SSIM

# 0011_mnist_1247_euclidean
# 0011_mnist_1247_SSIM
# CW-SSIM missing too long runtime

# 0100_mnist_1247_cnn_seed3871_euclidean
# 0101_mnist_1247_euclidean


# 1000_oct_cc_cnn_seed3871_euclidean
# 1000_oct_cc_cnn_seed3871_SSIM
# 1001_oct_cc_euclidean
# # 1001_oct_cc_SSIM



m0000_eucl = combine_pickle(path_base + "0000_mnist_1247_cnn_seed3871_euclidean", r)
m0000_ssim = combine_pickle(path_base + "0000_mnist_1247_cnn_seed3871_SSIM", r)
m0000_cw = combine_pickle(path_base + "0000_mnist_1247_cnn_seed3871_CW-SSIM", r)

m0001_eucl = combine_pickle(path_base + "0001_mnist_1247_euclidean", r)
m0001_ssim = combine_pickle(path_base + "0001_mnist_1247_SSIM", r)
m0001_cw = combine_pickle(path_base + "0001_mnist_1247_CW-SSIM", r)

m0010_eucl = combine_pickle(path_base + "0010_mnist_1247_cnn_seed3871_euclidean", r)
m0010_ssim = combine_pickle(path_base + "0010_mnist_1247_cnn_seed3871_SSIM", r)
m0010_cw = combine_pickle(path_base + "0010_mnist_1247_cnn_seed3871_CW-SSIM", r)

m0011_eucl = combine_pickle(path_base + "0011_mnist_1247_euclidean", r)
m0011_ssim = combine_pickle(path_base + "0011_mnist_1247_SSIM", r)
# m0011_cw = combine_pickle(path_base + "0011_mnist_1247_CW-SSIM", r)

m0100_eucl = combine_pickle(path_base + "0100_mnist_1247_cnn_seed3871_euclidean", r)
m0101_eucl = combine_pickle(path_base + "0101_mnist_1247_euclidean", r)


# o1000_eucl = combine_pickle(path_base + "1000_oct_cc_cnn_seed3871_euclidean", r)
# o1000_ssim = combine_pickle(path_base + "1000_oct_cc_cnn_seed3871_SSIM", r)

# o1001_eucl = combine_pickle(path_base + "1001_oct_cc_euclidean", r)
# o1001_ssim = combine_pickle(path_base + "1001_oct_cc_SSIM", r)

# o1010_eucl = combine_pickle(path_base + "1010_oct_cc_cnn_seed3871_euclidean", r)
# o1010_ssim = combine_pickle(path_base + "1010_oct_cc_cnn_seed3871_SSIM", r)

# o1011_eucl = combine_pickle(path_base + "1011_oct_cc_euclidean", r)
# o1011_ssim = combine_pickle(path_base + "1011_oct_cc_SSIM", r)

# o1100_eucl = combine_pickle(path_base + "1100_oct_cc_cnn_seed3871_euclidean", r)
# o1101_eucl = combine_pickle(path_base + "1101_oct_cc_euclidean", r)
# -

all_df = {"0000_eucl": m0000_eucl, "0000_ssim": m0000_ssim, "0000_cw": m0000_cw,
          "0001_eucl": m0001_eucl, "0001_ssim": m0001_ssim, "0001_cw": m0001_cw,
          "0010_eucl": m0010_eucl, "0010_ssim": m0010_ssim, "0010_cw": m0010_cw,
          "0011_eucl": m0011_eucl, "0011_ssim": m0011_ssim, # "0011_cw": m0011_cw,
          "0100_eucl": m0100_eucl, "0101_eucl": m0101_eucl #, 
          # "1000_eucl": o1000_eucl, "1000_ssim": o1000_ssim,
          # "1001_eucl": o1001_eucl, "1001_ssim": o1001_ssim,
          # "1010_eucl": o1010_eucl, "1010_ssim": o1010_ssim,
          # "1011_eucl": o1011_eucl, "1011_ssim": o1011_ssim,
          # "1100_eucl": o1100_eucl, "1101_eucl": o1101_eucl
         }

# ## Add top Misses for all df

for df in all_df:
    add_top_misses(all_df[df])

# ## Save new pickles

for df in all_df:
    picklepath = "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/NHNM/" + df + \
                 "_FINAL50"
    print(picklepath)
    all_df[df].to_pickle(picklepath + ".pickle")


