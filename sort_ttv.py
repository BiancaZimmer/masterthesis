# Functions to sort images into train, test and validation folders

# In order to set up the  training we need a certain file structure. For this wee need all our images divided into
# training, validation and test set (each a unique folder) and each again divided into the label folders.
# This code takes all the folders in the image directory, from this it infers how many and which labels we have,
# creates the folder structure as mentioned above and sorts the images accordingly.
# The validation folder is optional, use -s <float> to give a validation split.
# If you do not want a test folder simply set the test split to 0.0

# sample usage:
# python sort_ttv.py /Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small2/train /Users/biancazimmer/Documents/Masterthesis_data 0.2
# python sort_ttv.py /Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small2/train /Users/biancazimmer/Documents/Masterthesis_data 0.2 -s 0.1
# python sort_ttv.py /Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small2/train /Users/biancazimmer/Documents/Masterthesis_data 0.0 -s 0.1


import argparse
import os
import random
import shutil

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


def makeimagesplits(imagedir, perc_split=0.2):
    """Splits the images from the imagedir into two splits
    :param imagedir: directory where original images are
    :type imagedir: str
    :param perc_split: default=0.2
    :type perc_split: float

    :return: Current paths of the images as a list
    """

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


def checkmove(ntrain, ntest, nval=[0]):
    """Checks if the copy/move of the files was successful. Prints number of pictures in folders onto the console"""
    numimg = []
    for l in get_labels():
        numimg.append(len(os.listdir(os.path.join(args.imagedir, l))))
    print("Pictures to move: " + str(sum(numimg)))
    print("Train pictures: " + str(sum(ntrain)))
    print("Test pictures: " + str(sum(ntest)))
    print("Validation pictures: " + str(sum(nval)))


# MAIN starts here
if __name__ == "__main__":
    makedirs()
    train_fnames, test_fnames = makeimagesplits(args.imagedir, args.testsplit)
    numtrain = copyfiles(train_fnames, args.imagedir, os.path.join(args.basedir, "train"))
    numtest = copyfiles(test_fnames, args.imagedir, os.path.join(args.basedir, "test"))

    if args.validationsplit:
        train_fnames, val_fnames = makeimagesplits(os.path.join(args.basedir, "train"), args.validationsplit)
        numval = movefiles(val_fnames, os.path.join(args.basedir, "train"), os.path.join(args.basedir, "val"))
        checkmove(numtrain, numtest, numval)
    else:
        checkmove(numtrain, numtest)

