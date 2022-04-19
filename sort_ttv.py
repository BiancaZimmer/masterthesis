# functions to sort images into train, test and validation folders

import argparse
import os
import random
import shutil

# sample usage:
# python sort_ttv.py /Users/biancazimmer/Documents/Masterthesis_data/data_keramy_small2/train Users/biancazimmer/Documents/Masterthesis_data 0.2

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
parser.add_argument("-v", "--validationset",
                    help="adds a third folder for the validation set, else only the test set is split off",
                    action="store_true")
parser.add_argument("-s", "--validationsplit",
                    help="determins the validation split. Traindata * validationsplit = size of validation set",
                    action="store_true")
args = parser.parse_args()


# help functions


def get_labels():
    """Infers all the labels from the base directory"""
    return [i for i in os.listdir(args.imagedir) if i != ".DS_Store"]  # TODO: there might be a nicer way to check this

def makedirs():
    """Makes directories for train, test and optional validation set"""
    all_labels = get_labels()

    train_dir = os.path.join(args.basedir, 'train')
    testing_dir = os.path.join(args.basedir, 'test')

    if args.validationset:
        validation_dir = os.path.join(args.base_dir, 'val')

    for l in all_labels:
        os.makedirs(os.path.join(train_dir, l), exist_ok=True)
        if args.validationset:
            os.makedirs(os.path.join(validation_dir, l), exist_ok=True)
        os.makedirs(os.path.join(testing_dir, l), exist_ok=True)


def dim(a):
    lengthdim = []
    for elem in a:
        lengthdim.append(len(elem))
    return lengthdim


def makeimagesplits(imagedir, perc_split=0.2):
    """Splits the images from the imagedir into two splits"""
    """Returns the current paths of the images as a list"""

    listoffnames = []
    test_fnames = []
    train_fnames = []

    for l in get_labels():
        # get file names per label
        fnames = [f for f in os.listdir(os.path.join(imagedir, l)) if f.endswith('.jpeg')]
        listoffnames.append(fnames)
        #print(fnames) # TODO
        # first put away split off data
        test = random.sample(fnames, round(perc_split * len(fnames)))
        test_fnames.append(test)
        rest_fnames = list(set(fnames) - set(test))
        train_fnames.append(rest_fnames)

    print("Total number of all images: " + str(sum(dim(listoffnames))))
    print(get_labels())
    print(dim(listoffnames))
    print(['Split1', 'Split2'])
    print(str(sum(dim(train_fnames))) + ',' + str(sum(dim(test_fnames))))
    return [train_fnames, test_fnames]


def copyfiles(train_fnames, imagefolder, destinationfolder):
    """copy files into their respective directories"""
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
    """copy files into their respective directories"""
    nummoved = []
    for files, d in zip(train_fnames, get_labels()):
        pathfrom = os.path.join(imagefolder, d)
        pathto = os.path.join(destinationfolder, d)
        for f in files:
            if os.path.isfile(os.path.join(pathfrom, f)):
                shutil.move(os.path.join(pathfrom, f), os.path.join(pathto, f))
        nummoved.append(len(os.listdir(pathto)))
    return nummoved


def checkmove(ntrain, ntest, nval=[0]):
    numimg = []
    for l in get_labels():
        numimg.append(len(os.listdir(os.path.join(args.imagedir, l))))
    print("pictures to move: " + str(sum(numimg)))
    print("Train pictures: " + str(sum(ntrain)))
    print("Test pictures: " + str(sum(ntest)))
    print("Validation pictures: " + str(sum(nval)))


# MAIN starts here
makedirs()
train_fnames, test_fnames = makeimagesplits(args.imagedir, args.testsplit)
numtrain = copyfiles(train_fnames, args.imagedir, os.path.join(args.basedir, "train"))
numtest = copyfiles(test_fnames, args.imagedir, os.path.join(args.basedir, "test"))

if args.validationset:
    train_fnames, val_fnames = makeimagesplits(os.path.join(args.basedir, "train"), args.validationsplit)
    numval = movefiles(val_fnames, os.path.join(args.basedir, "train"), os.path.join(args.basedir, "val"))
    checkmove(numtrain, numtest, numval)
else:
    checkmove(numtrain, numtest)

# ----------------------------------


# Get File Names and Make Directories
# In order to set up the  training we need a certain file structure. For this wee need all out images devided into
# training, validation and test set (each a unique folder) and each again devided into the category folders.
# This code takes all the folders in the base directory, from this it infers how many and which classes we have,
# creates the folder structure as mentioned above and sorts the images accordingly. For this to be possible there
# mustn't be any other folders in the base directory except the generated folders, a folder called "Models"
# and one folder per class.
# **NOTE** If the directories already exist they will be updated but not over-written. So just put your new data in
# the base_dir/class_folder and run this code chunk again.
