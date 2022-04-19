# functions to sort images into train, test and validation folders

import os
import random
import shutil

base_dir = '/Users/biancazimmer/Documents/Masterthesis_data/data_keramy_small2'

## functions
def moveFileRename(file, pathfrom,  pathto):
  if os.path.isfile(os.path.join(pathfrom,file)):
    fileo = file
    while os.path.isfile(os.path.join(pathto,file)):
      file = str(random.sample([1,2,3,4,5,6,7,8,9,0], 1)).strip('[]') + file
      print(fileo + " renamed to " + file)
    os.rename(os.path.join(pathfrom,fileo), os.path.join(pathfrom,file))
    shutil.move(os.path.join(pathfrom,file), os.path.join(pathto,file))
  elif os.path.isfile(os.path.join(pathto,file)):
    pass #file already moved
  else:
    print('File could not be moved: ' + os.path.join(pathfrom,file))

def dim(a):
  lengthdim = []
  for elem in a:
    lengthdim.append(len(elem))
  return lengthdim

def makedirs(perc_test = 0.2, perc_validation = 0.3, base_dir=base_dir):
  listoffnames = []
  test_fnames = []
  val_fnames = []
  train_fnames = []
  all_features = [i for i in os.listdir(base_dir) if i not in ['Training', 'Testing', 'Validation', 'Models']]

  train_dir = os.path.join(base_dir, 'Training')
  validation_dir = os.path.join(base_dir, 'Validation')
  testing_dir = os.path.join(base_dir, 'Testing')
  for f in all_features:
    os.makedirs(os.path.join(train_dir, f), exist_ok= True)
    os.makedirs(os.path.join(validation_dir, f), exist_ok= True)
    os.makedirs(os.path.join(testing_dir, f), exist_ok= True)
    # get file names
    fnames = [f for f in os.listdir(os.path.join(base_dir,f)) if f.endswith('.jpg')]
    listoffnames.append(fnames)
    #first put away testing data
    test = random.sample(fnames, round(perc_test*len(fnames)))
    test_fnames.append(test)
    rest_fnames = list(set(fnames) - set(test))
    # then define validation and training data
    val = random.sample(rest_fnames, round(perc_validation*len(rest_fnames)))
    val_fnames.append(val)
    train = list(set(rest_fnames) - set(val))
    train_fnames.append(train)

  print("Total number of all images: " + str(sum(dim(listoffnames))))
  print(all_features)
  print(dim(listoffnames))
  print(['Training', 'Validation', 'Testing',])
  print(str(sum(dim(train_fnames))) + ','  + str(sum(dim(val_fnames))) + ',' + str(sum(dim(test_fnames))))
  return [train_fnames, val_fnames, test_fnames]

def movefiles(train_fnames, val_fnames, test_fnames , base_dir=base_dir):
  # move files into their respective directories
  # training
  for files, dir in zip(train_fnames, all_features):
    for f in files:
      moveFileRename(f, os.path.join(base_dir,dir),  os.path.join(base_dir,train_dir,dir))

  # validation
  for files, dir in zip(val_fnames, all_features):
    for f in files:
      moveFileRename(f, os.path.join(base_dir,dir),  os.path.join(base_dir,validation_dir,dir))

  # testing
  for files, dir in zip(test_fnames, all_features):
    for f in files:
      moveFileRename(f, os.path.join(base_dir,dir),  os.path.join(base_dir,testing_dir,dir))


# Get File Names and Make Directories
# In order to set up the the training we need a certain file structure. For this wee need all out images devided into
# training, validation and test set (each a unique folder) and each again devided into the category folders.
# This code takes all the folders in the base directory, from this it infers how many and which classes we have,
# creates the folder structure as mentioned above and sorts the images accordingly. For this to be possible there
# mustn't be any other folders in the base directory except the generated folders, a folder called "Models"
# and one folder per class.
# **NOTE** If the directories already exist they will be updated but not over-written. So just put your new data in
# the base_dir/class_folder and run this code chunk again.

all_features = [i for i in os.listdir(base_dir) if i not in ['Training', 'Testing', 'Validation', 'Models']]
train_fnames, val_fnames, test_fnames = makedirs()
test_dir = os.path.join(base_dir, 'Testing')
validation_dir = os.path.join(base_dir, 'Validation')
train_dir = os.path.join(base_dir, 'Training')

movefiles(train_fnames, val_fnames, test_fnames)

# check if moving worked
num_train_images = 0
num_val_images = 0
# overview stats
print(all_features)
for dir in ['Training', 'Validation', 'Testing']:
    per_dir = []
    for f in all_features:
        # get file names
        fnames = [f for f in os.listdir(os.path.join(base_dir, dir, f)) if f.endswith('.jpg')]
        per_dir.append(len(fnames))
    print(str(per_dir) + "   " + str(sum(per_dir)) + " - " + dir)
    if dir == 'Training':
        num_train_images = sum(per_dir)
    if dir == 'Validation':
        num_val_images = sum(per_dir)
percentages = [round(i / sum(per_dir), 2) * 100 for i in per_dir]
print(str(percentages) + "  - percentages  ")

# TODO:
# add an option to not make a validation directory -> only 2
# add sys arguments to execute this from the console (for this: research sys arguments)
