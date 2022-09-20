# Functions to sort images into train, test and validation folders

# In order to set up the  training we need a certain file structure. For this wee need all our images divided into
# training, validation and test set (each a unique folder) and each again divided into the label folders.
# This code takes all the folders in the image directory, from this it infers how many and which labels we have,
# creates the folder structure as mentioned above and sorts the images accordingly.
# The validation folder is optional, use -s <float> to give a validation split.
# If you do not want a test folder simply set the test split to 0.0

# sample usage:
# python3 sort_ttv.py <from imagedir> <to basedir> <testsplit> -s <validationsplit>
# python3 sort_ttv.py /Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small2/train /Users/biancazimmer/Documents/Masterthesis_data 0.2
# python3 sort_ttv.py /Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small2/train /Users/biancazimmer/Documents/Masterthesis_data 0.2 -s 0.1
# python3 sort_ttv.py /Users/biancazimmer/Documents/Masterthesis_data/MNIST_data_jpg/train /Users/biancazimmer/Documents/Masterthesis_data 0.0 -s 0.1
# python3 sort_ttv.py /Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/data/oct_cc/test /Users/biancazimmer/Documents/PycharmProjects/masterthesis/code_mh_main/static/data/oct_small_cc2 0.05


import argparse
import os
import random
import shutil
import numpy as np

# initialize parser for command line args
parser = argparse.ArgumentParser()
parser.add_argument("imagedir",
                    help="Image directory in which images can be found. Images need to be sorted into folders by labels",
                    type=str)
parser.add_argument("basedir",
                    help="Base directory into which images shall be sorted/folders should be created",
                    type=str)
parser.add_argument("testsplit",
                    help="determins the test split. All_images * testsplit = size of test set",
                    type=float)
parser.add_argument("-s", "--validationsplit",
                    help="adds a third folder for the validation set, else only the test set is split off. Value "
                         "gives the validation split. Traindata * validationsplit = size of validation set",
                    type=float)
args = parser.parse_args()


# help functions


def get_labels():
    """Infers all the labels from the base directory"""
    return [i for i in os.listdir(args.imagedir) if i != ".DS_Store"]  # TODO: there might be a nicer way to check this
    # return [i for i in os.listdir(args.imagedir) if os.path.isdir(i)]


def makedirs():
    """Makes directories for train, test and optional validation set"""
    all_labels = get_labels()

    train_dir = os.path.join(args.basedir, 'train')
    testing_dir = os.path.join(args.basedir, 'test')

    if args.validationsplit:
        validation_dir = os.path.join(args.basedir, 'val')

    for l in all_labels:
        os.makedirs(os.path.join(train_dir, l), exist_ok=True)
        if args.validationsplit:
            os.makedirs(os.path.join(validation_dir, l), exist_ok=True)
        os.makedirs(os.path.join(testing_dir, l), exist_ok=True)


def dim(a):
    lengthdim = []
    for elem in a:
        lengthdim.append(len(elem))
    return lengthdim


def split_by_patient(fnames, perc_split=0.2):
    """ In the case of the Kermany et al data set we have multiple pictures per patient. Pictures of a patient should all be put into one split and not randomly assigned into both.

    :param fnames: file names of the structure "XX-XX-XX"
    :param perc_split: default=0.2
    :return: Two list with the paths of the images in them
    """
    # split files at "-" and take the patient number
    patientnumbers = [f.split("-")[1] for f in fnames]
    set_of_patientnumbers = set(patientnumbers)
    # sample from patient numbers
    testpatients = random.sample(set_of_patientnumbers, round(perc_split * len(set_of_patientnumbers)))
    # get files with sampled patient ID
    idx_test = np.where(np.isin(patientnumbers, testpatients))[0]
    idx_rest = np.where(np.isin(patientnumbers, testpatients, invert=True))[0]
    return idx_test, idx_rest

# python sort_ttv.py /Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small3/train /Users/biancazimmer/Documents/Masterthesis_data 0.2


def makeimagesplits(imagedir, perc_split=0.2, splitbypatient=False, maxiter=10, deviation=0.01):
    """Splits the images from the imagedir into two splits
    :param imagedir: directory where original images are
    :type imagedir: str
    :param perc_split: default=0.2
    :type perc_split: float
    :param splitbypatient: In the case of the Kermany et al data set we have multiple pictures per patient. Pictures of a patient should all be put into one split and not randomly assigned into both.
    :type splitbypatient: boolean
    :param maxiter: maximum iteration when splitbypatient=True for resampling
    :type maxiter: integer
    :param deviation: maximum deviation from the perc_split when splitbypatient=True for resampling
    :type deviation: float

    :return: Two list with the paths of the images in them
    """

    listoffnames = []
    test_fnames = []
    train_fnames = []

    for l in get_labels():
        # get file names per label
        fnames = [f for f in os.listdir(os.path.join(imagedir, l)) if (f.endswith('.jpg') or f.endswith('.jpeg'))]
        listoffnames.append(fnames)
        # first put away split off data
        test = random.sample(fnames, round(perc_split * len(fnames)))
        rest = list(set(fnames) - set(test))
        if splitbypatient:
            # while proportion is not right -> resample
            proportion = 0
            c = 0
            while (perc_split*(1+deviation) < proportion or proportion < perc_split*(1-deviation)) and c < maxiter:
                print("Cycle ", c, " failed. Proportion of ", proportion, " Trying again ...")
                idx_test, idx_rest = split_by_patient(fnames, perc_split=perc_split)
                proportion = len(idx_test)/len(fnames)
                c = c+1
            print("Winning proportion for ", l, ": ", proportion)
            test = [np.array(fnames)[i] for i in idx_test]
            rest = [np.array(fnames)[i] for i in idx_rest]
        test_fnames.append(test)
        train_fnames.append(rest)

    print("Total number of all images: " + str(sum(dim(listoffnames))))
    print(get_labels())
    print(dim(listoffnames))
    print(['Split1', 'Split2'])
    print(str(sum(dim(train_fnames))) + ',' + str(sum(dim(test_fnames))))
    return [train_fnames, test_fnames]


def copyfiles(train_fnames, imagefolder, destinationfolder):
    """copy files into their respective directories
    :param train_fnames:
    :type train_fnames: list
    :param imagefolder: original folder where images are
    :type imagefolder: str
    :param destinationfolder: destination folder of where to copy the files to
    :type destinationfolder: str

    :return: number of copied images
    """
    numcopied = []
    for files, d in zip(train_fnames, get_labels()):
        pathfrom = os.path.join(imagefolder, d)
        pathto = os.path.join(destinationfolder, d)
        for f in files:
            if os.path.isfile(os.path.join(pathfrom, f)):
                shutil.copy(os.path.join(pathfrom, f), os.path.join(pathto, f))
        numcopied.append(len(os.listdir(pathto)))
    return numcopied


def movefiles(train_fnames, imagefolder, destinationfolder):
    """moves files into their respective directories
    :param train_fnames:
    :type train_fnames: list
    :param imagefolder: original folder where images are
    :type imagefolder: str
    :param destinationfolder: destination folder of where to copy the files to
    :type destinationfolder: str

    :return: number of moved images
    """
    nummoved = []
    for files, d in zip(train_fnames, get_labels()):
        pathfrom = os.path.join(imagefolder, d)
        pathto = os.path.join(destinationfolder, d)
        for f in files:
            if os.path.isfile(os.path.join(pathfrom, f)):
                shutil.move(os.path.join(pathfrom, f), os.path.join(pathto, f))
        nummoved.append(len(os.listdir(pathto)))
    return nummoved


def checkmove():
    """Checks if the copy/move of the files was successful. Prints number of pictures in folders onto the console"""
    numimg = []
    for l in get_labels():
        numimg.append(len(os.listdir(os.path.join(args.imagedir, l))))
    print("Pictures to move: ", sum(numimg))
    numimg = []
    for l in get_labels():
        numimg.append(len(os.listdir(os.path.join(args.basedir, "train", l))))
    print("Train pictures: ", sum(numimg))
    numimg = []
    for l in get_labels():
        numimg.append(len(os.listdir(os.path.join(args.basedir, "test", l))))
    print("Test pictures: ", sum(numimg))
    if os.path.exists(os.path.join(args.basedir, "val")):
        numimg = []
        for l in get_labels():
            numimg.append(len(os.listdir(os.path.join(args.basedir, "val", l))))
        print("Validation pictures: ", sum(numimg))


# MAIN starts here
if __name__ == "__main__":
    import time
    tic = time.time()
    makedirs()
    train_fnames, test_fnames = makeimagesplits(args.imagedir, args.testsplit, splitbypatient=False, maxiter=10,
                                                deviation=0.05)
    print("Copying train files ...")
    numtrain = copyfiles(train_fnames, args.imagedir, os.path.join(args.basedir, "train"))
    print("Copying test files ...")
    numtest = copyfiles(test_fnames, args.imagedir, os.path.join(args.basedir, "test"))

    if args.validationsplit:
        train_fnames, val_fnames = makeimagesplits(os.path.join(args.basedir, "train"), args.validationsplit,
                                                   splitbypatient=False, maxiter=10, deviation=0.05)
        print("Moving validation files ...")
        numval = movefiles(val_fnames, os.path.join(args.basedir, "train"), os.path.join(args.basedir, "val"))

    checkmove()

    toc = time.time()
    print("{}h {}min {}sec ".format(np.floor(((toc - tic) / (60 * 60))), np.floor(((toc - tic) % (60 * 60)) / 60),
                                    ((toc - tic) % 60)))
