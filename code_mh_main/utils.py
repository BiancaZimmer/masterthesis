"""**Utils** - This includes the paths used in the implementations.
"""

import os 

# Allowed image extensions
image_extensions = ['.jpg','.jpeg', '.bmp', '.png', '.gif']

MAIN_DIR = os.getcwd()
DATA_DIR = os.path.join(MAIN_DIR,'static', 'data')
DATA_OUTPUT_DIR = os.path.join(MAIN_DIR, 'data_output')
STATIC_DIR = os.path.join(MAIN_DIR,'static')
DIR_DATASETS = os.path.join("./static", 'data')
DIR_FEATURES = os.path.join("./static", 'feature_embedding')