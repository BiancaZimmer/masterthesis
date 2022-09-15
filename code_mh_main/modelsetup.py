import sys
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
import gc
import psutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight


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
    """ Contains a model with its respective data set and other important attributes
    self.selected_dataset
    self.dataset
    self.img_size
    self.image_shape
    # Image Generator
    self.train_set = None
    self.val_set = None
    self.test_set = None
    self.mode_rgb = None
    # Model specific
    self.model = None
    self.model_history = None
    self.predictions = None
    # for multiclass
    self.labelencoder = None
    # for imbalanced data
    self.correct_for_imbalanced_data = True
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
        self.val_set = None
        self.test_set = None
        self.mode_rgb = None
        # Model specific
        self.model = None
        self.model_history = None
        self.predictions = None
        # for multiclass
        self.labelencoder = None
        # for imbalanced data
        self.correct_for_imbalanced_data = True

    def _preprocess_img_gen(self):
        """Based on the given dataset, the ImageDataGenerator for the models are created.
        """
        if not BINARY:
            label_train = [x.ground_truth_label for x in self.dataset.data_t]
            self.labelencoder = LabelEncoder()
            self.labelencoder.fit(label_train)  # uses alphabetical order

        print('Initializing Image Generator ...')

        if self.selected_dataset == 'quality':
            image_gen = ImageDataGenerator(
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True
            )
        elif self.selected_dataset in ['oct', 'oct_small_cc', 'oct_small_rc', 'oct_cc', 'oct_rc']:
            image_gen = ImageDataGenerator(
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=False,
                rotation_range=30,
                shear_range=20
            )
        else:
            image_gen = ImageDataGenerator()

        image_gen_val = ImageDataGenerator()
        image_gen_test = ImageDataGenerator()

        if self.mode_rgb:
            self.train_set = image_gen.flow_from_directory(self.dataset.DIR_TRAIN_DATA,
                                                           target_size=(self.img_size, self.img_size),
                                                           color_mode='rgb', class_mode='categorical',
                                                           batch_size=self.batch_size, shuffle=True)
            self.val_set = image_gen_val.flow_from_directory(self.dataset.DIR_VAL_DATA,
                                                          target_size=(self.img_size, self.img_size),
                                                          color_mode='rgb', class_mode='categorical',
                                                          batch_size=self.batch_size, shuffle=False)
            self.test_set = image_gen_test.flow_from_directory(self.dataset.DIR_TEST_DATA,
                                                          target_size=(self.img_size, self.img_size),
                                                          color_mode='rgb', class_mode='categorical',
                                                          batch_size=self.batch_size, shuffle=False)

        else:
            self.train_set = image_gen.flow_from_directory(self.dataset.DIR_TRAIN_DATA,
                                                           target_size=(self.img_size, self.img_size),
                                                           color_mode='grayscale', class_mode='categorical',
                                                           batch_size=self.batch_size, shuffle=True)
            self.val_set = image_gen_val.flow_from_directory(self.dataset.DIR_VAL_DATA,
                                                          target_size=(self.img_size, self.img_size),
                                                          color_mode='grayscale', class_mode='categorical',
                                                          batch_size=self.batch_size, shuffle=False)
            self.test_set = image_gen_test.flow_from_directory(self.dataset.DIR_TEST_DATA,
                                                          target_size=(self.img_size, self.img_size),
                                                          color_mode='grayscale', class_mode='categorical',
                                                          batch_size=self.batch_size, shuffle=False)

        # print(self.train_set.class_indices)
        # print(self.train_set.class_indices.keys())
        # self.labelencoder = LabelEncoder().set_params(**self.train_set.class_indices)

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

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

    def _multiclass_cnn_model(self, use_original_cnn: bool = False):
        """Set up MultiCNN for multi class image classification.

        :param use_original_cnn: If set to True the smaller, original CNN of Marvin Herchenbach et al. is used. Else a bigger one with Batch Normalization is used
        :type use_original_cnn: boolean, optional, default = False
        """
        image_shape = (self.img_size, self.img_size, 1)
        backend.clear_session()
        print('Clear Session & Setup Model ...')

        if use_original_cnn:
            self.model = Sequential()
            self.model.add(Conv2D(filters=8, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(224))
        else:
            self.model = Sequential()
            self.model.add(Conv2D(filters=8, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(filters=128, kernel_size=(3,3),input_shape=image_shape, activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(256))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128))
            self.model.add(Dropout(0.5))
            self.model.add(BatchNormalization())

        self.model.add(Activation('relu'))
        self.model.add(Dense(len(self.labelencoder.classes_)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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

    def set_model(self, suffix_path: str = ''):
        """Load the already trained SimpleCNN and its history about training.

        :param suffix_path: Optional suffix to give a model a special name, defaults to ''
        :type suffix_path: str, optional
        """

        print(os.path.join(STATIC_DIR, 'models', 'model_history_' + str(self.selected_dataset) + str(suffix_path) + '.hdf5'), " loading ...")
        self.model = load_model(os.path.join(STATIC_DIR, 'models', 'model_history_' + str(self.selected_dataset) + str(suffix_path) + '.hdf5'))
        self.model_history = json.load(open(os.path.join(STATIC_DIR, 'models', 'model_history_' + str(self.selected_dataset) + str(suffix_path) + '.json'), 'r'))
        print(self.model.summary())
        print("Model input shape: ", self.model.input_shape)

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

        model_save_path = os.path.join(STATIC_DIR, 'models', 'model_history_' + str(self.selected_dataset) + str(suffix_path) + '.hdf5')

        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        checkpoint = ModelCheckpoint(filepath=model_save_path, verbose=1, save_best_only=save_model, monitor='val_loss')
        class_weights = None

        tic = time.time()

        # if data set is imbalanced calculate class weights for the loss function; else this is None
        if self.correct_for_imbalanced_data:
            y_train = [x.y for x in self.dataset.data]
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = {i: class_weights[i] for i in np.unique(y_train)}
            print("Class weights for imbalanced data: ", class_weights)

        results = self.model.fit(self.train_set,
                                 epochs=n_epochs,
                                 validation_data=self.val_set,
                                 callbacks=[early_stop, checkpoint],
                                 class_weight=class_weights)
        toc = time.time()
        print("Training needed: ",
              "{}h {}min {}sec ".format(math.floor(((toc - tic) / (60 * 60))), math.floor(((toc - tic) % (60 * 60)) / 60),
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
            groundtruth = self.test_set.labels
        else:
            groundtruth = self.labelencoder.inverse_transform(self.test_set.labels)
            # print("le classes: ", self.labelencoder.classes_)
            # print("datagen classes: ", self.test_set.class_indices)

        custom_map = sns.light_palette("#13233D", as_cmap=True)

        plt.figure(figsize=(10, 6))
        plt.title("Confusion Matrix - " + str(self.dataset.name) + " dataset", size=20, weight='bold')
        sns.heatmap(
            confusion_matrix(groundtruth, self.predictions),
            annot=True,
            annot_kws={'size': 14, 'weight': 'bold'},
            fmt='d',
            # cmap=plt.cm.Blues,
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
        
        prediction = self.model.predict(img_pred, verbose=0)
        # print(prediction)
        
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
            groundtruth = self.test_set.labels
        else:
            groundtruth = self.labelencoder.inverse_transform(self.test_set.labels)

        idx_misclassified = np.where(self.predictions != groundtruth)[0]
        idx_misclassified = list(idx_misclassified)
        misclassified_names = [self.dataset.data_t[idx].img_name for idx in idx_misclassified]

        misclassified = []
        
        print(f'[=>] {len(idx_misclassified)} misclassified images with names: {misclassified_names}')

        plt.figure(figsize=(20, 8))
        for i, idx in enumerate(idx_misclassified):

            misclassified.append(self.dataset.data_t[idx])
            
            if plot:
                img = cv2.imread(self.dataset.data_t[idx].img_path)
                label = self.dataset.data_t[idx].ground_truth_label
                predicted_label, prob = self.pred_test_img(self.dataset.data_t[idx], plot=False)

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
        # custom_map = sns.light_palette("#13233D", as_cmap=True, reverse=True)
        img_pred = np.expand_dims(np.expand_dims(test_dataentry.image_numpy(img_size=self.img_size), 0), -1)
        # Extracts the outputs of the top 12 layers
        layer_outputs = [layer.output for layer in self.model.layers]
        # Creates a model that will return these outputs, given the model input
        activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
        # Returns a list of five Numpy arrays: one array per layer activation
        activations = activation_model.predict(img_pred)

        layer_names = []
        for layer in self.model.layers[:12]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
            
        images_per_row = activations[0].shape[-1]

        l = 0
        plt.subplots(figsize=(8, 8))

        num_a_layer = sum([len(a.shape) >= 4 for a in activations])

        print(num_a_layer, '<========')
        for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
            print(layer_name, ': ', layer_activation.shape)
            if len(layer_activation.shape) < 4: 
                continue
            else:
                n_features = layer_activation.shape[-1]  # Number of features in the feature map
                size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
                n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
                display_grid = np.zeros((size * n_cols, images_per_row * size))
                for col in range(n_cols):  # Tiles each filter into a big horizontal grid
                    for row in range(images_per_row):
                        channel_image = layer_activation[0,
                                                        :, :,
                                                        col * images_per_row + row]
                        channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[col * size : (col + 1) * size,  # Displays the grid
                                    row * size : (row + 1) * size] = channel_image

                plt.subplot(num_a_layer, 1, l+1)
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                # plt.imshow(display_grid, aspect='auto', cmap=custom_map)
                plt.axis('off')

                l += 1
        plt.tight_layout()
        plt.show()


def load_model_from_folder(sel_dataset: str, suffix_path: str = ''):
    """Load a trained model from the static/models folder.

    :param sel_dataset: Name of the `Dataset` for which the trained classifier is loaded
    :type sel_dataset: str
    :param suffix_path: Optional suffix to give a model a special name, defaults to ''
    :type suffix_path: str, optional
    :return: **model** (`tf.keras.Sequential()`) - Trained SimpleCNN on the selected `DataSet`

    """
    model = load_model(os.path.join(STATIC_DIR, 'models', 'model_history_' + str(sel_dataset) + str(suffix_path) + '.hdf5'))
    return model


def get_output_layer(model, type_of_model):
    """ Returns output layer used for the FeatureExtractor depending on the type of the model (vgg or cnn)

    :param model: model
    :param type_of_model: str, vgg or cnn
    :return: output layer which then can be used for the FeatureExtractor
    """
    if type_of_model == "vgg":
        return model.get_layer('flatten').output
    elif type_of_model == "cnn":
        return model.layers[-3].output
    else:
        print("Output layer not defined for this type of model")
        warnings.warn("Output layer not defined for this type of model")
        return None


def train_eval_model(dataset_to_use, fit = True, type_of_model ='vgg', suffix_path ='_testincep',
                     model_for_feature_embedding = None,
                     eval = False, loss = False, misclassified = False,
                     feature_model_output_layer = None, correct_for_imbalanced_data = True):
    """ Essentially this is the code to run for fitting/loading/evaluating a  model
    :param dataset_to_use: A string of the folder name of the data set you want to use
    :param fit: bool, if set to True a model is fitted otherwise an existing one is loaded
    :param type_of_model: str, one of cnn or vgg results in taking speical options for these two models. if anything else an InceptionNet will be fitted
    :param suffix_path: str, suffix which will be appended to the saved model name
    :param model_for_feature_embedding: model which shall be used for the FeatureExtractor and thus the embeddings
    :param eval: bool, should the evaluation of the model be run or not
    :param loss: bool, should the loss and accuracy be plotted
    :param misclassified: bool, should the misclassified images be plotted
    :param feature_model_output_layer: output layer which shall be used for the FeatureExtractor and thus the embeddings
    :param correct_for_imbalanced_data: bool, should the loss function of the model be weighted and thus correct for imbalanced data
    :return: obj of class ModelSetup which contains the model
    """
    from dataset import DataSet, get_dict_datasets
    from feature_extractor import FeatureExtractor

    # CODE FROM CNN_MODEL.PY
    print("----- TRAINING OF MODEL -----")
    plot_losses = False
    suffix_path = suffix_path
    options_cnn = False
    if type_of_model == "cnn":
        options_cnn = True
    if model_for_feature_embedding is not None:
        feature_model_output_layer = get_output_layer(model_for_feature_embedding, type_of_model)

    # dictionary of data sets to use
    dataset_used = DataSet(name=dataset_to_use,
                           fe=FeatureExtractor(loaded_model=model_for_feature_embedding,
                                               model_name=str.upper(type_of_model),
                                               options_cnn=options_cnn,
                                               feature_model_output_layer=feature_model_output_layer))
    # initialize model
    sel_model = ModelSetup(dataset_used)
    sel_model.correct_for_imbalanced_data = correct_for_imbalanced_data
    if type_of_model == 'cnn':
        sel_model.mode_rgb = False
    else:
        sel_model.mode_rgb = True

    # initialize img generator
    print("Preprocessing images ...")
    sel_model._preprocess_img_gen()

    if fit:
        if BINARY:
            sel_model._binary_cnn_model()
        else:
            if type_of_model == 'cnn':
                sel_model._multiclass_cnn_model()
            elif type_of_model == 'vgg':
                sel_model._multiclass_transferlearning_vgg16_model()
            else:
                sel_model._multiclass_transferlearning_inception_model()
        print("Fitting model ...")
        sel_model.fit(save_model=True, suffix_path=suffix_path)
    else:
        sel_model.set_model(suffix_path=suffix_path)

    if eval:
        print("Evaluating model ...")
        if loss:
            plot_losses = True
        sel_model.eval(plot_losses=plot_losses)
        sel_model.plot_rand10_pred()
        quality_misclassified = sel_model.get_misclassified(plot=misclassified)

    return sel_model


if __name__ == "__main__":
    # dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information. ")
    # while dataset_to_use == "help":
    #     print("We need the folder name of a data set that is saved in your DATA_DIR. Usually that would be"
    #           "one of the names you specified in the DATA_DIR_FOLDERS list. e.g. 'mnist'")
    #     dataset_to_use = input("Which data set would you like to choose? Type 'help' if you need more information.")

    dataset_to_use = "oct_small_cc"
    train_eval_model(dataset_to_use, fit = True, type_of_model='cnn', suffix_path='_test',
                     model_for_feature_embedding = None,
                     eval = True, loss = True, misclassified= True)

    # predicts one test image
    # label, prob = sel_model.pred_test_img(dict_datasets[sel_model.selected_dataset].data_t[0], plot=True)

    # plots activation maps of one image
    # sel_model.plot_activation_map(dict_datasets[sel_model.selected_dataset].data_t[42])
    # sel_model.plot_activation_map(dict_datasets[sel_model.selected_dataset].data_t[55])

    ## Test model on different data set
    # type_of_model = 'vgg'
    # from dataset import DataSet
    # from feature_extractor import FeatureExtractor
    #
    # # CODE FROM CNN_MODEL.PY
    # print("----- TRAINING OF MODEL -----")
    # plot_losses = False
    # suffix_path = "_vgg2balanced"
    # options_cnn = False
    #
    # # dictionary of data sets to use
    # dataset_used = DataSet(name=dataset_to_use,
    #                        fe=FeatureExtractor(loaded_model=None,
    #                                            model_name=str.upper(type_of_model),
    #                                            options_cnn=options_cnn))
    # # initialize model
    # sel_model = ModelSetup(dataset_used)
    # sel_model.mode_rgb = True
    #
    # # initialize img generator
    # print("Preprocessing images ...")
    # sel_model._preprocess_img_gen()
    # # sel_model.set_model(suffix_path=suffix_path)
    # sel_model.model = load_model(
    #     os.path.join(STATIC_DIR, 'models', 'model_history_' + "oct_small_cc" + str(suffix_path) + '.hdf5'))
    # sel_model.model_history = json.load(open(
    #     os.path.join(STATIC_DIR, 'models', 'model_history_' + "oct_small_cc" + str(suffix_path) + '.json'),
    #     'r'))
    # print(sel_model.model.summary())
    # print("Model input shape: ", sel_model.model.input_shape)
    #
    # print("Evaluating model ...")
    # sel_model.eval(plot_losses=True)
    # sel_model.plot_rand10_pred()
    # quality_misclassified = sel_model.get_misclassified(plot=True)
