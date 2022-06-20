"""**Utils** - This includes the paths used in the implementations.
"""

import os 

# Allowed image extensions
image_extensions = ['.jpg','.jpeg', '.bmp', '.png', '.gif']

MAIN_DIR = os.getcwd()
DATA_DIR = os.path.join(MAIN_DIR,'static', 'data')
DATA_OUTPUT_DIR = os.path.join(MAIN_DIR, 'data_output')     # os.path.join(DATA_DIR, 'data_output')
STATIC_DIR = os.path.join(MAIN_DIR,'static')
DIR_DATASETS = DATA_DIR # os.path.join("./static", 'data')
DIR_FEATURES = os.path.join(MAIN_DIR, 'static', 'feature_embedding')
DATA_DIR_FOLDERS = ['mnist']    # set to [] if you want to use all folders in the DATA_DIR, else give name of folders as list that should be used
# DIR_PROTOTYPES_DATASET = os.path.join(MAIN_DIR,'static/prototypes', self.dict_datasets[self.sel_dataset].fe.fe_model.name ,self.sel_dataset)
BINARY = False  # is your data set binary? then set to True, otherwise, set to false
