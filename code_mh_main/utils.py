"""**Utils** - Set paths and global variables for the implementation

MAIN_DIR:           base directory - takes working directory
STATIC_DIR:         directory for static data
DATA_DIR:           directory of folders containing your original data, should be in the static folder
DATA_OUTPUT_DIR:    output folder - can be anywhere you have writing access to
DIR_FEATURES:       directory to put the feature embeddings to, should be in the static folder
DATA_DIR_FOLDERS:   set to [] if you want to use all folders in the DATA_DIR, else give name of subfolders as list that
                    should be used; subfolders have to contain "test", "train" and "val" folder, these in turn contain
                    folders with the labels:
                    dataset name
                    ├── train
                        ├── Label1
                        ├── Label2
                        ├── Label3
                    ├── test
                        ├── Label1
                        ├── Label2
                        ├── Label3
                    ├── val
                        ├── Label1
                        ├── Label2
                        ├── Label3
BINARY:             boolean value; True for binary data, else False; algorithms are then chosen automatically for multiclass/binary
TOP_N_NMNH:         Number of Near Misses/Near Hits to calculate
RANDOMSEED:         random seed for reproducability; master thesis: 3871

"""

import os 

# Allowed image extensions
image_extensions = ['.jpg', '.jpeg', '.bmp', '.png', '.gif']

MAIN_DIR = os.getcwd()
STATIC_DIR = os.path.join(MAIN_DIR, 'static')
DATA_DIR = os.path.join(MAIN_DIR, 'static', 'data')
DATA_OUTPUT_DIR = os.path.join(MAIN_DIR, 'data_output')
DIR_FEATURES = os.path.join(MAIN_DIR, 'static', 'feature_embedding')
DATA_DIR_FOLDERS = ['mnist_147', 'mnist_1247', 'mnist_12567'] # 'mnist_147', 'mnist_1247', 'mnist_12567', 'oct_small_cc', 'oct_cc'
BINARY = False  # True for binary data, else False;
TOP_N_NMNH = 5  # Number of Near Misses/Near Hits to calculate
RANDOMSEED = 3871  # 3871
