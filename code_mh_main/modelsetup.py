import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2

import json
import random
import time
import math

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


import keras
from keras import models
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler


#from dataentry import *
from dataset import *
from utils import *


class ModelSetup():
    """ TODO
    """
    def __init__(self, selected_dataset, sel_size:int =128, batch_size:int =32):
        """Initiliaze the data-specific attributes for training the model.

        :param selected_dataset: Dataset on which the classifier is to be trained
        :type selected_dataset: DataSet
        :param sel_size: Image size onto which the images from the quality dataset are reduced, defaults to 128
        :type sel_size: int, optional
        :param batch_size: Batch size used to train the CNN, defaults to 32
        :type batch_size: int, optional
        """
        self.selected_dataset = selected_dataset.name
        self.dataset = selected_dataset
        self.img_size = sel_size
        self.image_shape = None

        if self.selected_dataset == 'mnist':
            self.img_size = 28
            # self.image_shape = (28,28,1)
        elif self.selected_dataset in ['oct_small_cc', 'oct_small_rc', 'oct_cc', 'oct_rc']:
            self.img_size = 299
        # else:
            # self.image_shape = (sel_size, sel_size, 1)

        self.batch_size = batch_size

        # Image Generator
        self.train_set = None
        self.test_set = None
        self.mode_rgb = None
        # Model specific
        self.model = None
        self.model_history = None
        self.predictions = None
        # for multiclass
        self.labelencoder = None

    def _preprocess_img_gen(self, rgb=False):
        """Based on the given dataset, the ImageDataGenerator for the CNN are created.
        """

        # here x is of the class DataEntry, y would be index of folder, ground_truth_label is label
        if rgb:  # convert image to rgb, no array expansion needed
            self.mode_rgb = True
            train_data = [(x.image_numpy(img_size=self.img_size, mode='RGB'), x.ground_truth_label)
                          for x in self.dataset.data]
        else:  # convert image to grayscale, need to expand array by 1 dimension
            self.mode_rgb = False
            train_data = [(np.expand_dims(x.image_numpy(img_size=self.img_size), -1), x.ground_truth_label)
                          for x in self.dataset.data]
        X_train = np.array(list(zip(*train_data))[0])
        y_train = np.array(list(zip(*train_data))[1])
        print('X_train shape: ', X_train.shape)
        # print('Sample: ', X_train[0], ' label: ', y_train[0])
        # print('y_train shape', y_train.shape)
        # print(y_train)

        if rgb:
            test_data = [(x.image_numpy(img_size=self.img_size, mode='RGB'), x.ground_truth_label)
                         for x in self.dataset.data_t]
        else:
            test_data = [(np.expand_dims(x.image_numpy(img_size=self.img_size), -1), x.ground_truth_label)
                         for x in self.dataset.data_t]
        X_test = np.array(list(zip(*test_data))[0])
        y_test = np.array(list(zip(*test_data))[1])
        # print(y_test)

        if not BINARY:
            # encode class values as integers
            self.labelencoder = LabelEncoder()
            self.labelencoder.fit(y_train)
            y_train = self.labelencoder.transform(y_train)
            y_test = self.labelencoder.transform(y_test)
            # print("LE classes from preprocess img gen: ", self.labelencoder.classes_)
            # convert integers to one hot encoded variables
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            # print('y_train shape: ', y_train.shape)
            # print(y_train)

        print('Initializing Image Generator ...')
        
        if self.selected_dataset == 'quality':
            image_gen = ImageDataGenerator(
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True            
            )
        elif self.selected_dataset == 'oct':
            image_gen = ImageDataGenerator(
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=False
            )
        else:
            image_gen = ImageDataGenerator()

        image_gen_test = ImageDataGenerator()

        self.train_set = image_gen.flow(X_train, y_train, 
                            batch_size=self.batch_size, 
                            shuffle=True)

        self.test_set = image_gen_test.flow(X_test, y_test, 
                                    batch_size=self.batch_size, 
                                    shuffle=False)

    def _binary_cnn_model(self):
        """Set up SimpleCNN for binary image classification.
        """
        image_shape = (self.img_size, self.img_size, 1)

        backend.clear_session()
        print('Clear Session & Setup Model ...')

        self.model = Sequential()
        self.model.add(Conv2D(filters=8, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(224))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        print(self.model.summary())

    def _multiclass_cnn_model(self):
        """Set up MultiCNN for multi class image classification.
        """
        image_shape = (self.img_size, self.img_size, 1)

        backend.clear_session()
        print('Clear Session & Setup Model ...')

        self.model = Sequential()
        self.model.add(Conv2D(filters=8, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(224))
        self.model.add(Activation('relu'))
        self.model.add(Dense(len(self.labelencoder.classes_)))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        print(self.model.summary())

    def _pretrained_network(self, pretrainedmodel, optimizer):
        numclasses = len(self.labelencoder.classes_)
        base_model = pretrainedmodel  # Topless
        # Add top layer
        x = base_model.output
        x = Flatten()(x)
        predictions = Dense(numclasses, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        # Train top layer
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # Fit model
        # callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
        # history = model.fit(xtrain, ytrain, epochs=numepochs, class_weight=classweight, validation_data=(xtest, ytest),
        #                     verbose=1, callbacks=[MetricsCheckpoint('logs')])
        return model

    def _multiclass_transferlearning_inception_model(self):
        image_shape = (self.img_size, self.img_size, 3)

        backend.clear_session()
        print('Clear Session & Setup Model ...')

        weight_path = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pretrained_model = InceptionV3(weights='imagenet', include_top=False, input_shape=image_shape)
        optimizer = Adam()
        # optimizer = RMSprop(lr=0.0001)

        self.model = self._pretrained_network(pretrained_model, optimizer)
        print(self.model.summary())

    def _multiclass_transferlearning_vgg16_model(self):
        image_shape = (self.img_size, self.img_size, 3)

        backend.clear_session()
        print('Clear Session & Setup Model ...')

        weight_path = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)
        optimizer = Adam()
        # optimizer = RMSprop(lr=0.0001)

        self.model = self._pretrained_network(pretrained_model, optimizer)
        print(self.model.summary())

    def load_model(self, suffix_path: str = ''):
        """Load the already trained SimpleCNN and its history about training.

        :param suffix_path: Optional suffix to give a model a special name, defaults to ''
        :type suffix_path: str, optional
        """

        print(os.path.join(STATIC_DIR, 'models', 'model_history_' + str(self.selected_dataset) + str(suffix_path) + '.hdf5'), " loading ...")
        self.model = load_model(os.path.join(STATIC_DIR, 'models', 'model_history_' + str(self.selected_dataset) + str(suffix_path) + '.hdf5'))
        self.model_history = json.load(open(os.path.join(STATIC_DIR, 'models', 'model_history_' + str(self.selected_dataset) + str(suffix_path) + '.json'), 'r'))

    def fit(self, n_epochs: int = 50, patience: int = 10, suffix_path: str = '',  save_model: bool = True):
        """Fit the SimpleCNN on the given dataset

        :param n_epochs: Number of epochs, defaults to 50
        :type n_epochs: int, optional
        :param patience: Early stopping patience, defaults to 10
        :type patience: int, optional
        :param suffix_path: optional suffix to give a model a special name, defaults to ''
        :type suffix_path: str, optional
        :param save_model: Set to `False` if the model should not be saved, defaults to True
        :type save_model: bool, optional
        """
        # TODO: add over/undersampling for biased datasets
        model_save_path = os.path.join(STATIC_DIR, 'models', 'model_history_' + str(self.selected_dataset) + str(suffix_path) + '.hdf5')

        early_stop = EarlyStopping(monitor='val_loss', patience=patience)
        checkpoint = ModelCheckpoint(filepath=model_save_path, verbose=1, save_best_only=save_model, monitor='val_loss')

        tic = time.time()
        results = self.model.fit_generator(self.train_set, 
                                    epochs=n_epochs, 
                                    validation_data=self.test_set,
                                    callbacks=[early_stop, checkpoint])
        toc = time.time()
        print("Training needed: ",
              "{}h {}min {}sec ".format(round(((toc - tic) / (60 * 60))), math.floor(((toc - tic) % (60 * 60)) / 60),
                                        ((toc - tic) % 60)))
        
        if save_model:
            self.model_history = results.history
            json.dump(self.model_history, open(os.path.join(STATIC_DIR, 'models', 'model_history_' + str(self.selected_dataset) + str(suffix_path) + '.json'), 'w'))

    def eval(self, plot_losses: bool = True):
        """Return classification performance of the SimpleCNN and plot Loss curve

        :param plot_losses: Set to `False` if no loss curve should be plotted, defaults to True
        :type plot_losses: bool, optional
        """
        if plot_losses:
            losses = pd.DataFrame(self.model_history)
            losses.index = map(lambda x: x+1, losses.index)
            print(losses)
            plt.figure(figsize=(10, 6))
            plt.title("Loss/Accuracy Plot", size=20, weight='bold')
            plt.ylabel('Epochs', size=14, weight='bold')
            plt.xlabel('Loss/Accuracy', size=14, weight='bold')
            train_loss, = plt.plot(losses['loss'])
            train_acc, = plt.plot(losses['accuracy'])
            val_loss, = plt.plot(losses['val_loss'])
            val_acc, = plt.plot(losses['val_accuracy'])
            plt.legend(handles=[train_loss, val_loss, train_acc, val_acc],
                       labels=['Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy'])
            plt.show()

        self._set_selfprediction()

        if BINARY:
            groundtruth = self.test_set.y
        else:
            groundtruth = self.labelencoder.inverse_transform(np.argmax(self.test_set.y, axis=1))
            # print("le classes: ", self.labelencoder.classes_)

        custom_map = sns.light_palette("#13233D", as_cmap=True)

        plt.figure(figsize=(10, 6))
        plt.title("Confusion Matrix - " + str(self.dataset.name) + " dataset", size=20, weight='bold')
        sns.heatmap(
            confusion_matrix(groundtruth, self.predictions),
            annot=True,
            annot_kws={'size': 14, 'weight': 'bold'},
            fmt='d',
            #cmap=plt.cm.Blues,
            cmap=custom_map,
            xticklabels=self.dataset.available_classes,
            yticklabels=self.dataset.available_classes)
        plt.tick_params(axis='both', labelsize=14)
        plt.ylabel('Actual', size=14, weight='semibold')
        plt.xlabel('Predicted', size=14, weight='semibold')
        plt.show()

        print(classification_report(groundtruth, self.predictions, digits=3))

    def pred_test_img(self, test_dataentry, plot:bool =False):
        """Classify the given test image.

        :param test_dataentry: DataEntry object which should be taken from the test dataset
        :type test_dataentry: DataEntry
        :param plot: Set to `True`if the images and the classification results should be shown, defaults to False
        :type plot: bool, optional
        :return: 
            - **predicted_label** (`str`) - Predicted label by using Simple CNN
            - **prob** (`float`) - Probability score for the predicted label
        """

        if self.mode_rgb:
            img_pred = np.expand_dims(test_dataentry.image_numpy(img_size=self.img_size, mode='RGB'), 0)
            pass
        else:
            img_pred = np.expand_dims(np.expand_dims(test_dataentry.image_numpy(img_size=self.img_size), 0), -1)
        
        prediction = self.model.predict(img_pred)
        #print(prediction)
        
        img = cv2.imread(test_dataentry.img_path)
        label = test_dataentry.ground_truth_label

        if BINARY:
            if prediction > 0.5:
                predicted_label = self.dataset.available_classes[0]
                prob = prediction.sum() * 100
            else:
                predicted_label = self.dataset.available_classes[1]
                prob = (1-prediction.sum()) * 100
        else:
            predicted_label = self.labelencoder.inverse_transform(np.argmax(prediction, axis=1))
            prob = np.max(prediction) * 100
        
        if plot:
            plt.figure(figsize=(20,8))

            plt.title(f"{test_dataentry.img_name}\n\
            Actual Label : {label}\n\
            Predicted Label : {predicted_label}\n\
            Probability : {'{:.3f}'.format(prob)}%", weight='bold', size=12)    

            plt.imshow(img,cmap='gray')
            plt.axis('off')
                
            plt.tight_layout()
            plt.show()
        
        return predicted_label, prob

    def plot_rand10_pred(self):
        """Plot 10 random prediction by using images from the test dataset.
        """
        rand_idx = [random.randint(0, len(self.dataset.data_t)) for p in range(0, 10)]

        plt.figure(figsize=(20, 8))
        for i, idx in enumerate(rand_idx):

            predicted_label, prob = self.pred_test_img(self.dataset.data_t[idx], plot = False)
            
            img = cv2.imread(self.dataset.data_t[idx].img_path)
            label = self.dataset.data_t[idx].ground_truth_label

            plt.subplot(2, 5, i+1)
            
            plt.title(f"{self.dataset.data_t[idx].img_name}\n\
            Actual Label : {label}\n\
            Predicted Label : {predicted_label}\n\
            Probability : {'{:.3f}'.format(prob)}%", weight='bold', size=12)    

            plt.imshow(img,cmap='gray')
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

    def _set_selfprediction(self):
        """ Sets the attribute prediction according to the model and the test_set
        """

        if BINARY:
            pred_probability = self.model.predict(self.test_set)
            self.predictions = pred_probability > 0.5
            self.predictions = self.predictions.reshape(-1)
        else:
            pred_probability = self.model.predict(self.test_set)
            predictions = np.argmax(pred_probability, axis=1)
            self.predictions = self.labelencoder.inverse_transform(predictions)
            # print("le classes: ", self.labelencoder.classes_)

    def transform_prediction(self, prediction, threshold = 0.5):
        """ Transforms the output of a model.predict() into a correctly predicted label and the according probability

        :param prediction: prediction output from model.predict()
        :param threshold: if model was binary you can set a threshold here. Default: 0.5
        :return: list of the predicted label and the according probability
        """
        if BINARY:
            if prediction < threshold:
                predicted_label = self.dataset.available_classes[0]
                prob = (1 - prediction.sum()) * 100
            else:
                predicted_label = self.dataset.available_classes[1]
                prob = prediction.sum() * 100
        else:
            predicted_label = self.labelencoder.inverse_transform(np.argmax(prediction, axis=1))
            prob = np.max(prediction) * 100

        return [predicted_label, prob]

    def get_misclassified(self, plot:bool =False):
        """Identifiy misclassifcation by the Simple CNN

        :param plot: Set to `True` to plot the misclassified images and the classification results, defaults to False
        :type plot: bool, optional
        :return: **misclassified** (`list`) - List of `DataEntry` instances that were not correctly classified by the SimpleCNN
        """

        if self.predictions is None:
            self._set_selfprediction()

        # misclassifications done on the test data where y_pred is the predicted values
        if BINARY:
            groundtruth = self.test_set.y
        else:
            groundtruth = self.labelencoder.inverse_transform(np.argmax(self.test_set.y, axis=1))

        idx_misclassified = np.where(self.predictions != groundtruth)[0]
        idx_misclassified = list(idx_misclassified)
        misclassified_names = [self.dataset.data_t[idx].img_name for idx in idx_misclassified]

        misclassified = []
        
        print(f'[=>] {len(idx_misclassified)} missclassfied Images with names: {misclassified_names}')

        plt.figure(figsize=(20,8))
        for i, idx in enumerate(idx_misclassified):

            misclassified.append(self.dataset.data_t[idx])

            img = cv2.imread(self.dataset.data_t[idx].img_path)
            label = self.dataset.data_t[idx].ground_truth_label

            predicted_label, prob = self.pred_test_img(self.dataset.data_t[idx], plot=False)
            
            if plot:
                plt.subplot(int(np.ceil(len(idx_misclassified)/5)), 5, i+1)
                
                plt.title(f"{self.dataset.data_t[idx].img_name}\n\
                Actual Label : {label}\n\
                Predicted Label : {predicted_label}\n\
                Probability : {'{:.3f}'.format(prob)}%", weight='bold', size=12)    

                plt.imshow(img,cmap='gray')
                plt.axis('off')
                
        if plot:
            plt.tight_layout()
            plt.show()
        
        return misclassified

    def plot_activation_map(self, test_dataentry):
        """Plot activation maps of the SimpleCNN by predicting a test image

        :param test_dataentry: DataEntry object which should be taken from the test dataset
        :type test_dataentry: DataEntry
        """
        #custom_map = sns.light_palette("#13233D", as_cmap=True, reverse=True)
        img_pred = np.expand_dims(np.expand_dims(test_dataentry.image_numpy(img_size=self.img_size), 0), -1)

        layer_outputs = [layer.output for layer in self.model.layers] # Extracts the outputs of the top 12 layers
        activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

        activations = activation_model.predict(img_pred) # Returns a list of five Numpy arrays: one array per layer activation

        layer_names = []
        for layer in self.model.layers[:12]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
            
        images_per_row = activations[0].shape[-1]

        l = 0
        plt.subplots(figsize=(8, 8))

        num_a_layer = sum([len(a.shape) >= 4 for a in activations])

        print(num_a_layer, '<========')
        for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
            print(layer_name, ': ', layer_activation.shape)
            if len(layer_activation.shape) < 4: 
                continue
            else:
                n_features = layer_activation.shape[-1] # Number of features in the feature map
                size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
                n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
                display_grid = np.zeros((size * n_cols, images_per_row * size))
                for col in range(n_cols): # Tiles each filter into a big horizontal grid
                    for row in range(images_per_row):
                        channel_image = layer_activation[0,
                                                        :, :,
                                                        col * images_per_row + row]
                        channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[col * size : (col + 1) * size, # Displays the grid
                                    row * size : (row + 1) * size] = channel_image

                plt.subplot(num_a_layer, 1, l+1)
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                #plt.imshow(display_grid, aspect='auto', cmap=custom_map)
                plt.axis('off')

                l += 1
        plt.tight_layout()
        plt.show()


def get_CNNmodel(sel_dataset:str, suffix_path:str =''):
    """Load solely the  trained SimpleCNN.

    :param sel_dataset: Name of the `Dataset` for which the trained classifier is loaded
    :type sel_dataset: str
    :param suffix_path: Optional suffix to give a model a special name, defaults to ''
    :type suffix_path: str, optional
    :return: **model** (`tf.keras.Sequential()`) - Trained SimpleCNN on the selected `DataSet`

    """
    model = load_model(os.path.join(STATIC_DIR, 'models', 'model_history_' + str(sel_dataset) + str(suffix_path) + '.hdf5'))
    return model


def train_eval_model(dataset_to_use, fit = True, type = 'vgg', suffix_path = '_testincep',
                     eval = False, loss = False, missclassified = False):
    """ Essentially this is the code to run for fitting/loading/evaluating a cnn model
    :param dataset_to_use: A string of the folder name of the data set you want to use

    :return: suffix_path which was given as user input
    """
    # fit = input("Do you want to fit (f) a model or load (l) an exciting one? [f/l]")
    # options[1] = input("What should the suffix of your cnn_model be? Type a string. e.g. '_testcnn'")
    # options[2] = input("Do you want to run the evaluation of your CNN model? [y/n]")
    # options[3] = input("Do you want to plot the loss and accuracy of your CNN model? [y/n]")
    # options[4] = input("Do you want to plot the evaluation of the miss-classified data of your CNN model? [y/n]")

    # CODE FROM CNN_MODEL.PY
    print("----- TRAINING OF MODEL -----")
    from dataset import get_dict_datasets

    plot_losses = False
    plotmisclassified = False
    suffix_path = suffix_path

    # dictionary of data sets to use
    use_CNN_feature_embedding = False  # Set to True in order to save the CNN-based feature embedding; else VGG16 embedding is used
    if use_CNN_feature_embedding:
        fe_CNNmodel = FeatureExtractor(loaded_model=get_CNNmodel(sel_dataset=dataset_to_use))
        dataset_used = DataSet(name=dataset_to_use, fe=fe_CNNmodel)
    else:
        dataset_used = DataSet(name=dataset_to_use, fe=FeatureExtractor(loaded_model=None))

    sel_model = ModelSetup(dataset_used)

    # initialize img generator
    if type == 'cnn':
        sel_model._preprocess_img_gen(rgb=False)
    else:
        sel_model._preprocess_img_gen(rgb=True)

    if fit:
        if BINARY:
            sel_model._binary_cnn_model()
        else:
            if type == 'cnn':
                sel_model._multiclass_cnn_model()
            elif type == 'vgg':
                sel_model._multiclass_transferlearning_vgg16_model()
            else:
                sel_model._multiclass_transferlearning_inception_model()
        print("Fitting model ...")
        sel_model.fit(save_model=True, suffix_path=suffix_path)
    else:
        sel_model.load_model(suffix_path=suffix_path)

    if eval:
        if loss:
            plot_losses = True
        sel_model.eval(plot_losses=plot_losses)
        sel_model.plot_rand10_pred()
        if missclassified:
            plotmisclassified = True
        quality_misclassified = sel_model.get_misclassified(plot=plotmisclassified)

    return suffix_path


if __name__ == "__main__":
    dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information. ")
    while dataset_to_use == "help":
        print("We need the folder name of a data set that is saved in your DATA_DIR. Usually that would be"
              "one of the names you specified in the DATA_DIR_FOLDERS list. e.g. 'mnist'")
        dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information.")

    train_eval_model(dataset_to_use, fit = False, type = 'vgg', suffix_path = '_smallvgg',
                     eval = True, loss = False, missclassified = True)

    # predicts one test image
    # label, prob = sel_model.pred_test_img(dict_datasets[sel_model.selected_dataset].data_t[0], plot=True)

    # plots activation maps of one image
    # sel_model.plot_activation_map(dict_datasets[sel_model.selected_dataset].data_t[42])
    # sel_model.plot_activation_map(dict_datasets[sel_model.selected_dataset].data_t[55])


    ### OLD  CODE ####
    # from dataset import get_dict_datasets, get_available_dataset
    #
    # # boolean switches, please adjust to your liking
    # use_CNN_feature_embedding = True    # Set to True in order to save the CNN-based feature embedding; else VGG16 embedding is used
    # fit_model = False    # if True a model is fitted, if false a model is loaded
    # suffix_path = "_multicnn"   # suffix_path = "_multicnn"
    # evaluate = True     # if True evaluation plus plot is run
    # plot_losses = False  # if True the evaluation losses are plotted
    # plotmisclassified = True   # if True misclassified jpgs are shown with classified and true label
    #
    # # from here on you don't have to change anything anymore
    # use_all_datasets = True
    # if len(DATA_DIR_FOLDERS) > 0: use_all_datasets = False
    #
    # # dictionary of data sets to use
    # dict_datasets = get_dict_datasets(use_CNN_feature_embedding, use_all_datasets)
    #
    # # careful: This is hard coded, always takes first data set
    # sel_model = CNNmodel(dict_datasets[DATA_DIR_FOLDERS[0]])
    #
    # # initialize img generator
    # sel_model._preprocess_img_gen()
    #
    # if fit_model:
    #     if BINARY:
    #         sel_model._binary_model()
    #     else:
    #         sel_model._multiclass_model()
    #     sel_model.fit(save_model=True, suffix_path=suffix_path)
    # else:
    #     sel_model.load_model(suffix_path=suffix_path)
    #
    # if evaluate:
    #     sel_model.eval(plot_losses=plot_losses)
    #     sel_model.plot_rand10_pred()
    #     quality_misclassified = sel_model.get_misclassified(plot=plotmisclassified)
    #     # print("Misclassified: ", len(quality_misclassified)) not needed since code above gives number
