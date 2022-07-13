"""**Utils** - This includes the paths used in the implementations.
"""

import os 

# Allowed image extensions
image_extensions = ['.jpg', '.jpeg', '.bmp', '.png', '.gif']

MAIN_DIR = os.getcwd()
STATIC_DIR = os.path.join(MAIN_DIR, 'static')
DATA_DIR = os.path.join(MAIN_DIR, 'static', 'data')
DATA_OUTPUT_DIR = os.path.join(MAIN_DIR, 'data_output')
DIR_FEATURES = os.path.join(MAIN_DIR, 'static', 'feature_embedding')
DATA_DIR_FOLDERS = ['mnist', 'oct_small_cc', 'oct_cc']    # set to [] if you want to use all folders in the DATA_DIR, else give name of folders as list that should be used
BINARY = False  # is your data set binary? then set to True, otherwise, set to false; algorithms are then chosen automatically for multiclass/binary
TOP_N_NMNH = 5  # Number of Near Misses/Near Hits to calculate
