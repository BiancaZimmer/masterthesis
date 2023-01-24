import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from prototype_selection import *
from sklearn.metrics import recall_score, f1_score

# simple class to build 1NN classifier and classify using it
class Classifier:
    """1-NN Classifier which is used for evaluating the selected prototypes, as well as for 
    hyperparameter optimization (e.g. kernel parameter of MMD-based approach).

    """
    model=None
    """Global variable for storing the model (e.g. 1-NN)
    """

    def __init__(self):
        pass

    def build_model(self, trainX, trainy, verbose:int =0):
        """The 1-NN classification model is set up and trained.

        :param trainX: N-dimensional numpy array containing all feature vectors / raw data  of train data (e.g. prototypes)
        :type trainX: numpy.ndarray
        :param trainy: N-dimensional numpy array containing the labels of train data (e.g. prototypes)
        :type trainy: numpy.ndarray
        :param verbose: Set to 1 in order to plot the number of used data points for training, defaults to 0
        :type verbose: int, optional
        """
        if verbose > 1: print("building model using %d points " %len(trainy)) 
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(trainX, trainy)

    def classify(self, testX, testy, verbose:int =0):
        """The trained 1-NN classifier is applied on the test data and various metrics are calculated.

        :param testX: N-dimensional numpy array containing all feature vectors / raw data  of test data 
        :type testX: numpy.ndarray
        :param testy: N-dimensional numpy array containing the labels of test data
        :type testy: numpy.ndarray
        :param verbose: Set to 1 in order to plot the results, defaults to 0
        :type verbose: int, optional

        :return: 
            - **acc** (`str`) - Accuracy score
            - **err** (`float`) - Error rate        
            - **recall** (`float`) - Recall score        

        """
        predy = self.model.predict(testX)
        if verbose > 1: 
            print("classifying %d points " %len(testy))
            print('Prediction:\n', predy)
            print('True Label:\n', testy)

        ncorrect = np.sum(predy == testy)
        acc = ncorrect/(len(predy) + 0.0)
        err = 1 - acc

        recall = recall_score(testy, predy, average=None)
        f1 = f1_score(testy, predy, average="weighted")

        print("F1scores: " + f1_score(testy, predy, average=None))

        return acc, err, recall, f1


