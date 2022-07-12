import sys
from cnn_model import *
import time
from feature_extractor import *
from dataentry import *
from prototype_selection import *


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
            fe = FeatureExtractor(loaded_model=get_CNNmodel(dataset, suffix_path=suffix_path))
        else:
            # Standard FE for general model:
            fe = FeatureExtractor()  # loaded_model=VGG16(weights='imagenet', include_top=True)
        print("Initiating ", fe.fe_model)

        # Load Dataset
        dataset = DataSet(name=dataset_name, fe=fe)

        # Initialize Prototype Selector
        num_prototypes = input("How many prototypes do you want to calculate? Type an integer.")

        a = input("Do you want to show the selected prototypes? [y/n]")
        make_plots_protos = True
        if a == "n":
            make_plots_protos = False

        a = input("Should the prototypes be calculated on the basis of the embeddings "
                                                   "or on the raw data? [e/r]")
        use_image_embeddings_for_protoypes = False
        if a == "e":
            use_image_embeddings_for_protoypes = True

        gamma_value = input("What gamma value do you want to use for the prototype selection?"
                            "Type a positive floating number or 'help'.")
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
                                "Type a positive floating number or 'help'.")

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
            # TODO ask if we want to overwrite the file
            print('[!!!] A file already exists! Please delete this file to save again prototypes of these settings.')
        else:
            print(protos_file)
            # np.save(protos_file, protos_img_files)
            print('SAVE ...')
            with open(protos_file, 'w') as fp:
                json.dump(protos_img_files, fp)


if __name__ == '__main__':
    a = input("Did you change the folders in the utils.py file as describe in the README? [y/n]")
    if a == "n":
        sys.exit("Go and do that or else the file won't run!")

    dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information.")
    while dataset_to_use == "help":
        print("We need the folder name of a data set that is saved in your DATA_DIR. Usually that would be"
              "one of the names you specified in the DATA_DIR_FOLDERS list. e.g. 'mnist'")
        dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information.")

    # Train or load and evaluate CNN Model
    # If you want to skip this point please specify the suffix_path below
    # suffix_path = ''
    suffix_path = code_from_ccn_model(dataset_to_use)

    # Create feature embeddings
    a = input("Do you want to create the feature embeddings for this model? [y/n/help]")
    while a == "help":
        print("You only need to create the feature embeddings if you haven't created them before. e.g. when you "
              "trained a brand new model.")
        a = input("Do you want to create the feature embeddings for this model? [y/n/help]")
    if a == "n":
        new_embedding = False
        print("No feature embeddings created.")
    else:
        code_from_dataentry(dataset_to_use, suffix_path)

    # Create prototypes
    a = input("Do you want to create the prototypes for your current data set now? [y/n]")
    if a == "y":
        code_from_prototype_selection(dataset_to_use)





