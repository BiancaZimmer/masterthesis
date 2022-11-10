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
# import cv2

# from feature_extractor import FeatureExtractor
# from dataentry import DataEntry
from helpers import jaccard
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
        tmp = pd.read_pickle(path+"_"+str(i)+".pickle")
        df = pd.concat([df, tmp])
    return df


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

def top_misses(lst, score):
    """
    Calculates the top overall misses from the lists of the multidimensional misses
    :param lst: list of list with paths to the misses
    :param score: list of list with scores of the misses
    :return: tuple with TOP_N_NMNH entries for the misses
    """
    lst = list(chain.from_iterable(lst))
    score = list(chain.from_iterable(score))
    paths = [p for (p,s) in sorted(zip(lst, score))]
    scores = [s for (p,s) in sorted(zip(lst, score))]
    return paths[:TOP_N_NMNH], scores[:TOP_N_NMNH]


def jaccard_df(df1, df2, method="intersection"):
    if type(df1[0][0]) is list:
        result = [jaccard(list(chain.from_iterable(l1)), list(chain.from_iterable(l2)), method) for
                  l1, l2 in zip(df1, df2)]
        return result
    else:
        result = [jaccard(l1, l2, method) for l1, l2 in zip(df1, df2)]
        return result


# ## Combine pickles

# +
# dataset_to_use = "mnist_1247"

path ="/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/oct_cc_cnn_seed3871_euclidean_usepredTrue_rawFalse_distonimgTrue"
    # "/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/mnist_1247_cnn_seed3871_cosine_usepredTrue_rawFalse_distonimgTrue"
df = combine_pickle(path, range(1, 3))
df.describe()
# -


mnist_eucl = pd.read_pickle("/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/mnist_1247_cnn_seed3871_euclidean_usepredTrue_rawFalse_distonimgTrue_100notrandom.pickle")
temp = mnist_eucl.apply(lambda row : top_misses(row["near_misses"], row["scores_misses"]), axis = 1)
mnist_eucl["top_misses"] = [t[0] for t in temp]
mnist_eucl["scores_top_misses"] = [t[1] for t in temp]
show_humanfriendly(mnist_eucl)

mnist_SSIM = pd.read_pickle("/Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/mnist_1247_cnn_seed3871_SSIM_usepredTrue_rawFalse_distonimgTrue_100notrandom.pickle")
temp = mnist_SSIM.apply(lambda row : top_misses(row["near_misses"], row["scores_misses"]), axis = 1)
mnist_SSIM["top_misses"] = [t[0] for t in temp]
mnist_SSIM["scores_top_misses"] = [t[1] for t in temp]
show_humanfriendly(mnist_SSIM)

# ## Overview over scores

df_scores = pd.DataFrame()
df_scores["scores_hit_eucl"] = list(chain.from_iterable(mnist_eucl.scores_hits))
df_scores["scores_hit_SSIM"] = list(chain.from_iterable(mnist_SSIM.scores_hits))
df_scores["scores_top_misses_eucl"] = list(chain.from_iterable(mnist_eucl.scores_top_misses))
df_scores["scores_top_misses_SSIM"] = list(chain.from_iterable(mnist_SSIM.scores_top_misses))
df_scores

df_scores.describe()

df_scores.boxplot(column=["scores_hit_eucl", "scores_top_misses_eucl"])

df_scores.boxplot(column=["scores_hit_SSIM", "scores_top_misses_SSIM"])

# ## Calculate Jaccard indices

# hits
res = pd.DataFrame()
res["jaccard_hits"] = jaccard_df(mnist_eucl.near_hits, mnist_SSIM.near_hits)
res["jaccard_hits_abs"] = jaccard_df(mnist_eucl.near_hits, mnist_SSIM.near_hits, "absolute")
# misses
res["jaccard_misses"] = jaccard_df(mnist_eucl.near_misses, mnist_SSIM.near_misses)
res["jaccard_misses_abs"] = jaccard_df(mnist_eucl.near_misses, mnist_SSIM.near_misses, "absolute")
# top misses
res["jaccard_top_misses"] = jaccard_df(mnist_eucl.top_misses, mnist_SSIM.top_misses)
res["jaccard_top_misses_abs"] = jaccard_df(mnist_eucl.top_misses, mnist_SSIM.top_misses, "absolute")
res

res.describe()

res.boxplot(column=["jaccard_misses", "jaccard_top_misses", "jaccard_hits"])

res.boxplot(column=["jaccard_misses_abs", "jaccard_top_misses_abs", "jaccard_hits_abs"])

# ##
# pd.read_pickle("path")

