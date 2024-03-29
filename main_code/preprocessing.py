import sys
# from cnn_model import *
import time
from feature_extractor import *
from dataentry import *
from prototype_selection import *
from helpers import crop_to_square, walk_directory_return_img_path
from modelsetup import *
from LRP_heatmaps import *

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


def code_from_prototype_selection(dataset_name):
    # CODE FROM PROTOTYPE_SELECTION.PY
    print("----- PROTOTYPESELECTION -----")
    print("If you haven't run an evaluation for your prototypes the following code might not yield reliable results. "
          "However you can try it out to get an overview.")

    create_prototypes = True
    feature_embeddings_to_use = "current"
    while create_prototypes:
        # Select Feature Extractor
        if feature_embeddings_to_use == "current":
            # gets FE from a loaded CNN with the dataset name and a suffix
            fe = FeatureExtractor(loaded_model=load_model_from_folder(dataset_name, suffix_path=suffix_path))
        else:
            # Standard FE for general model:
            fe = FeatureExtractor()  # loaded_model=VGG16(weights='imagenet', include_top=True)
        print("Initiating ", fe.fe_model)

        # Load Dataset
        dataset = DataSet(name=dataset_name, fe=fe)

        # Initialize Prototype Selector
        num_prototypes = input("How many prototypes do you want to calculate? Type an integer. ")

        a = input("Do you want to show the selected prototypes? [y/n] ")
        make_plots_protos = True
        if a == "n":
            make_plots_protos = False

        a = input("Should the prototypes be calculated on the basis of the embeddings "
                  "or on the raw data? [e/r] ")
        use_image_embeddings_for_protoypes = False
        if a == "e":
            use_image_embeddings_for_protoypes = True

        gamma_value = input("What gamma value do you want to use for the prototype selection?"
                            "Type a positive floating number or 'help'. ")
        while gamma_value == "help" or gamma_value < 0:
            print("Your gamma value is either smaller than 0 or you typed help. \n"
                  "Here are some sample gamma values that we deem to be a good fit: \n"
                  "-------------"
                  "For a binary mnist: \n"
                  "gamma_vgg16_mnist = 4e-05\n"
                  "gamma_simpleCNN_mnist = 1 \n"
                  "gamma_rawData_mnist = 0.0001 \n"
                  "-------------"
                  "For 5 class mnist: \n"
                  "-------------"
                  "For 4 class Kermany et al. OCT data set: \n"
                  "-------------")
            gamma_value = input("What gamma value do you want to use for the prototype selection?"
                                "Type a positive floating number or 'help'. ")

        tester = PrototypesSelector_MMD(dataset, num_prototypes=int(num_prototypes),
                                        use_image_embeddings=use_image_embeddings_for_protoypes,
                                        gamma=float(gamma_value),
                                        verbose=1, make_plots=make_plots_protos)
        tester.fit()
        tester.score()

        # TODO ask if also kmedoids + implement

        if tester.use_image_embeddings:
            DIR_PROTOTYPES_DATASET = os.path.join(MAIN_DIR, 'static/prototypes', fe.fe_model.name, dataset.name)
        else:
            DIR_PROTOTYPES_DATASET = os.path.join(MAIN_DIR, 'static/prototypes', "rawData", dataset.name)

        if not os.path.exists(DIR_PROTOTYPES_DATASET):
            os.makedirs(DIR_PROTOTYPES_DATASET)

        protos_file = os.path.join(DIR_PROTOTYPES_DATASET, str(tester.num_prototypes) + '.json')

        protos_img_files = tester.get_prototypes_img_files()

        if os.path.exists(protos_file):
            overwrite = input("[!!!] A file with these settings already exists! Do you want to overwrite it? [y/n] ")
            if overwrite == "y":
                overwrite = True
            else:
                overwrite = False
        else:
            overwrite = True
        if overwrite:
            print(protos_file)
            # np.save(protos_file, protos_img_files)
            print('SAVE ...')
            with open(protos_file, 'w') as fp:
                json.dump(protos_img_files, fp)


def questions_cropping():
    b = None
    a = input("All your images need to be in a squared format. Do you want them to be cropped? [y/n] ")
    if a == "y":
        b = input("Do you want them to be center (c) cropped or randomly cropped (r)? [c/r] ")
    return [a, b]


def crop_train_test_val(dataset_to_use, crop='y', centre='c'):
    if crop == "y":
        if centre == "c":
            centre = True
            new_folder = str(dataset_to_use + "_cc")
        else:
            centre = False
            new_folder = str(dataset_to_use + "_rc")

        print("Cropping training data ...")
        for img in walk_directory_return_img_path(os.path.join(DATA_DIR, dataset_to_use, 'train')):
            new_path = os.path.join(DATA_DIR, new_folder, 'train', img.split("/")[-2], img.split("/")[-1])
            crop_to_square(img, centre=centre, save=True, new_path=new_path)
        print("Cropping test data ...")
        for img in walk_directory_return_img_path(os.path.join(DATA_DIR, dataset_to_use, 'test')):
            new_path = os.path.join(DATA_DIR, new_folder, 'test', img.split("/")[-2], img.split("/")[-1])
            crop_to_square(img, centre=centre, save=True, new_path=new_path)
        print("Cropping validation data ...")
        for img in walk_directory_return_img_path(os.path.join(DATA_DIR, dataset_to_use, 'val')):
            new_path = os.path.join(DATA_DIR, new_folder, 'val', img.split("/")[-2], img.split("/")[-1])
            crop_to_square(img, centre=centre, save=True, new_path=new_path)


def questions_training(dataset_to_use):
    from tensorflow.keras.models import load_model

    imbalanced = False
    fit = input("Do you want to fit (f) a model or load (l) an exciting one? [f/l] ")
    if fit == "f":
        modeltype = input("What kind of model do you want to train? [cnn/vgg/inception] ")
        suffix_path = input("What should the suffix of your model be? Type a string. e.g. _testcnn ")
        model_for_feature_embedding = None
        fit = True
        if input("Do you want to correct the training for imbalanced data set? [y/n] ") == 'y':
            imbalanced = True
    else:
        suffix_path = input("What is the suffix of your model? Type a string. e.g. _testcnn ")
        modeltype = input("What kind of model is your loaded model? [cnn/vgg/inception] ")
        model_for_feature_embedding = load_model(
            os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'))
        fit = False

    eval_ = False
    loss = False
    misclassified = False
    if input("Do you want to run the evaluation of your model? [y/n] ") == "y":
        eval_ = True
        if input("Do you want to plot the loss and accuracy of your model? [y/n] ") == "y":
            loss = True
        if input("Do you want to plot the evaluation of the miss-classified data of your model? [y/n] ") == "y":
            misclassified = True

    return [fit, modeltype, suffix_path, model_for_feature_embedding, eval_, loss, misclassified, imbalanced]


if __name__ == '__main__':
    print("\n---------------------- START ----------------------\n")
    # a = input("Did you change the folders in the utils.py file as describe in the README? [y/n] ")
    # if a == "n":
    #     sys.exit("Go and do that or else the file won't run!")
    #
    dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information. ")
    if not os.path.exists(os.path.join(DATA_DIR, dataset_to_use)):
        print("WARNING: Folder does not exist. Please check for spelling mistakes and if it is in the static/data folder.")
        dataset_to_use = "help"
    while dataset_to_use == "help":
        print("We need the folder name of a data set that is saved in your DATA_DIR. Usually that would be "
              "one of the names you specified in the DATA_DIR_FOLDERS list. e.g. 'mnist'")
        dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information. ")
        if not os.path.exists(os.path.join(DATA_DIR, dataset_to_use)):
            print("WARNING: Folder does not exist. Please check for spelling mistakes and if it is in the static/data folder.")
            dataset_to_use = "help"

    # dataset_to_use = "mnist_1247"

    # centre crop images?
    # crop, centre = questions_cropping()
    # crop_train_test_val(dataset_to_use, crop=crop, centre=centre)

    # Train or load and evaluate CNN Model
    training = questions_training(dataset_to_use)
    # training = [True, 'cnn', '_cnnori_balanced', None, True, True, False, True]
    # training = [False, 'cnn', '_cnn5c2d6bn_balanced', None, False, False, False]
    setup_model = train_eval_model(dataset_to_use, fit=training[0], type_of_model=training[1], suffix_path=training[2],
                                   model_for_feature_embedding=training[3],
                                   eval=training[4], loss=training[5], misclassified=training[6],
                                   correct_for_imbalanced_data=training[7])
    suffix_path = training[2]

    # Create feature embeddings
    a = input("Do you want to create the feature embeddings for this model? [y/n/help] ")
    while a == "help":
        print("You only need to create the feature embeddings if you haven't created them before. e.g. when you "
              "trained a brand new model. ")
        a = input("Do you want to create the feature embeddings for this model? [y/n/help] ")
    if a == "n":
        if input("Do you want to create the feature embeddings for the general VGG16? [y/n] ") == "y":
            code_from_dataentry(dataset_to_use, suffix_path, feature_embeddings_to_initiate="VGG16", type_of_model=training[1])
        else:
            print("No feature embeddings created. ")
    else:
        code_from_dataentry(dataset_to_use, suffix_path, type_of_model=training[1])

    # Create LRP Heatmaps
    a = input("Do you want to create LRP heatmaps for your current data set and trained model now? [y/n] ")
    if a == "y":
        method = input("Which method would you ike to use? We propose: \n"
                       "lrp.sequential_preset_a for the mnist data\n"
                       "lrp.sequential_preset_a_flat for the oct data\n")
        if method == "lrp.sequential_preset_a":
            epsilon = input("Which epsilon value would you like to use? We propose 0.1 ")
            parameters = {"epsilon": float(epsilon)}
        else:
            parameters = {}
        generate_LRP_heatmaps_for_dataset(dataset_to_use=dataset_to_use, suffix_path=suffix_path,
                                          type_of_model=training[1],
                                          method=method, parameters=parameters)
    if input("Do you want to create LRP heatmaps for a general (untrained) VGG16? [y/n] ") == "y":
        method = input("Which method would you ike to use? We propose: \n"
                       "lrp.sequential_preset_a for the mnist data\n"
                       "lrp.sequential_preset_a_flat for the oct data\n")
        if method == "lrp.sequential_preset_a":
            epsilon = input("Which epsilon value would you like to use? We propose 0.1 ")
            parameters = {"epsilon": float(epsilon)}
        else:
            parameters = {}
        generate_LRP_heatmaps_for_dataset(dataset_to_use=dataset_to_use, suffix_path=suffix_path,
                                          type_of_model='vgg',
                                          method=method, parameters=parameters, base_vgg=True)

    # Create prototypes
    print("For the creation of the prototypes you first need to select the parameters."
          "Please do this via 'prototype_selection.py'."
          "Then you can run the code in 'prototype_selection.py' to select your prototypes.")

    # Create NHNMs
    print("For the creation of the NHNMs please refer to 'near_miss_hits_selection.py'")

