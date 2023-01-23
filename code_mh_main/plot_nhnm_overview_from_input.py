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


# ## Load pickles
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

# ### MNIST
# generate column names
scores_names = []
scores_top_names = []
for m in mnist_df:
    scores_names.append("scores_hit_"+m)
    scores_top_names.append("scores_top_misses_"+m)

# +
mnist_scores = pd.DataFrame()

for score_hit, score_top, df in zip(scores_names, scores_top_names, mnist_df.values()):
    mnist_scores[score_hit] = list(chain.from_iterable(df.scores_hits))
    mnist_scores[score_top] = list(chain.from_iterable(df.scores_top_misses))


# ### OCT
# generate column names
scores_names = []
scores_top_names = []
for m in oct_df:
    scores_names.append("scores_hit_"+m)
    scores_top_names.append("scores_top_misses_"+m)

oct_scores = pd.DataFrame()

for score_hit, score_top, df in zip(scores_names, scores_top_names, oct_df.values()):
    oct_scores[score_hit] = list(chain.from_iterable(df.scores_hits))
    oct_scores[score_top] = list(chain.from_iterable(df.scores_top_misses))

plt.close("all")

# ## Plot certain images + maps

def plot_nhnm_overview_from_input(lst,
                                  dataset, suffix_path="_multicnn", type_of_model="cnn", save_name='cosine',
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
    if suffix_path == "":
        fe = FeatureExtractor(loaded_model=None,
                              model_name=str.upper(type_of_model),
                              options_cnn=options_cnn,
                              feature_model_output_layer=None)
    else:
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

    test_img = DataEntry(fe, dataset, change_imgpath(lst[0]))
    img, x = fe.load_preprocess_img(test_img.img_path)

    pred_label = test_img.ground_truth_label

    if use_prediction:
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

        pred_label, pred_prob = sel_model.pred_test_img(test_img)
        print("Ground Truth: ", test_img.ground_truth_label)
        print("Prediction: ", pred_label)
        print("Probability: ", pred_prob)

    # pred_label needs to be a string
    if type(pred_label) != str:
        pred_label = pred_label[0]
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
        # Plot image + heatmap
        heatmap_directory = os.path.join(STATIC_DIR, 'heatmaps', test_img.fe.fe_model.name, dataset)

        fig1 = plt.figure()
        plt.subplot(2, 1, 1)
        plt.title(f"{test_img.img_name}\nActual Label : {test_img.ground_truth_label}\nPredicted Label : {pred_label}",
                  weight='bold', size=10, loc="left")
        plt.imshow(img, cmap='gray')
        if distance_on_image and not raw:
            # get correct path to heatmap
            test_image_name = str.split(test_img.img_name, ".")[0]
            test_image_heatmap_path = os.path.join(heatmap_directory, "test", pred_label,
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
            fig3 = plot_nmnh_heatmaps(ranked_nearest_hit_data_entry, scores_nearest_hit, pred_label,
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
                                      np.concatenate(scores_nearest_miss_multi), pred_label,
                                      title="Near Misses Heatmaps per Class")
            fig5.savefig("fig5.png", bbox_inches='tight')
            plt.close()

        # concatenate all pictures into one with cv2 and save it
        # load pictures
        pic1 = cv2.imread("fig1.png", cv2.IMREAD_GRAYSCALE)
        pic2 = cv2.imread("fig2.png", cv2.IMREAD_GRAYSCALE)
        if distance_on_image and not raw:
            pic3 = cv2.imread("fig3.png", cv2.IMREAD_GRAYSCALE)
            pic23 = cv2.hconcat([pic2, pic3])
        else:  # distance on FE
            pic23 = pic2
        pic4 = cv2.imread("fig4.png", cv2.IMREAD_GRAYSCALE)
        if distance_on_image and not raw:
            pic5 = cv2.imread("fig5.png", cv2.IMREAD_GRAYSCALE)
            pic45 = cv2.hconcat([pic4, pic5])
        else:  # distance on FE
            pic45 = pic4

        # concatenate them
        pic_all = hconcat_resize_max([pic1, vconcat_resize_min([pic23, pic45])])
        save_path = STATIC_DIR+"/"+test_img.img_name+"_"+save_name+"_eval.png"
        cv2.imwrite(save_path, pic_all)
        plt.close("all")


if __name__ == '__main__':
    # _cnn_seed3871
    combination = "011_eucl"
    use_prediction = False
    raw = False
    distance_on_image = True
    # #000_eucl  #001_eucl  #000_ssim #001_ssim  #100_eucl  #101_eucl  #010_eucl  #010_ssim  #011_eucl #011_ssim
    # #000_cw #001_cw #010_cw #011_cw

    # for i in range(0, 20):
    #     plot_nhnm_overview_from_input(list(mnist_df["0" + combination].iloc[i]),
    #                                   "mnist_1247", suffix_path="_cnn_seed3871", type_of_model="cnn",
    #                                   save_name="0" + combination,  top_n=TOP_N_NMNH,
    #                                   use_prediction=use_prediction, raw=raw, distance_on_image=distance_on_image)
    for i in range(0, 20):
        plot_nhnm_overview_from_input(list(oct_df["1" + combination].iloc[i]),
                                      "oct_cc", suffix_path="", type_of_model="vgg",
                                      save_name="1" + combination, top_n=TOP_N_NMNH,
                                      use_prediction=use_prediction, raw=raw, distance_on_image=distance_on_image)
