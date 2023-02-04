"""**Offline Phase** - This includes solely a main function where the feature embedding is calculated from the given dataset.
"""
    
import numpy as np
import os
import time

from feature_extractor import FeatureExtractor
from dataentry import DataEntry

MAIN_DIR = os.getcwd()
DATA_DIR = os.path.join(MAIN_DIR,'static', 'data')
DATA_OUTPUT_DIR = os.path.join(MAIN_DIR, 'data_output')



if __name__ == '__main__':

    ###### ==== Select a DATASET ==== ######
    # dataset = 'mnist'
    dataset = 'quality'


    # Initialize Feature Extractor Instance 
    fe = FeatureExtractor()

    # Image path of the dataset
    image_path = os.path.join(DATA_DIR,dataset, 'train')

    # Start Timer
    tic = time.time()

    # Allowed image extensions
    image_extensions = ['.jpg','.jpeg', '.bmp', '.png', '.gif']

    data = [DataEntry(fe,dataset,os.path.join(path, file)) for path, _, files in os.walk(image_path) for file in files if file.endswith(tuple(image_extensions))]


    # Counter for loaded Images
    i = 0

    for d in data:
        if os.path.exists(d.feature_file):
            # print(d.feature_file)
            # print(d.img_file)
            # print('pass')
            pass
        else:
            _, x = d.fe.load_preprocess_img(d.img_file)
            feat = d.fe.extract_features(x)
            print(d.feature_file)

            np.save(d.feature_file, feat)
            print("SAVE...")
            i += 1
            pass


    print("... Loaded images: ", np.size(data))
    print("... Re-loade & saved images: ", i)

    toc = time.time()
    elap = toc-tic
    print("Time: %4.4f seconds." % (elap))

