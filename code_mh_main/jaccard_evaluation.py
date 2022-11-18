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
import cv2

# from feature_extractor import FeatureExtractor
from dataentry import DataEntry
from helpers import jaccard, change_imgpath
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

mnist_scores.describe()

mnist_scores.boxplot(column=scores_names, rot= 10)

mnist_scores.boxplot(column=scores_names[1:], rot= 10)

# mnist_scores.boxplot(column=scores_top_names, rot= 10)

mnist_scores.boxplot(column=scores_top_names[1:], rot= 10)

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

oct_scores.describe()

oct_scores.boxplot(column=scores_names[:-1], rot= 10)

# oct_scores.boxplot(column=scores_names[1:-1], rot= 10)

oct_scores.boxplot(column=scores_top_names[:-1], rot= 10)

# oct_scores.boxplot(column=scores_top_names[1:-1], rot= 10)


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

    hit_class_idx = []
    miss_class_idx = []

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
        def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
            # https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
            w_min = min(im.shape[1] for im in im_list)
            im_list_resize = [
                cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                for im in im_list]
            return cv2.vconcat(im_list_resize)

        def hconcat_resize_max(im_list, interpolation=cv2.INTER_CUBIC):
            # https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
            h_min = max(im.shape[0] for im in im_list)
            im_list_resize = [
                cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                for im in im_list]
            return cv2.hconcat(im_list_resize)

        pic_all = hconcat_resize_max( [pic1, vconcat_resize_min([pic23, pic45])] )
        cv2.imwrite('fig12345.png', pic_all)


if __name__ == '__main__':
    distance = "SSIM-pushed"
    plot_nhnm_overview_from_input(list(oct_df[distance].iloc[0]),
                                      "oct_cc", suffix_path="_cnn_seed3871", type_of_model="cnn", distance_measure=distance,
                                      top_n=TOP_N_NMNH, use_prediction=True, raw=False, distance_on_image=True)







