import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as rbf


def default_gamma(X:np.ndarray):
    """Function that calculates a default gamma value if no gamma defined, by using the formula: 1.0/ X.shape[1]

    :param X: Data of DataSet which should transformed to a numpy array first
    :type X: np.ndarray

    :return: *gamma* (`float`) - Default gamma value
    """
    gamma = 1.0 / X.shape[1]
    return gamma

def rbf_kernel(X:np.ndarray, gamma:float=None):
    """Function that calculated the RBF Kernel.

    :param X: Data of DataSet which should transformed to a numpy array first
    :type X: np.ndarray
    :param gamma: Pass a gamma value, else the default gamma value will be calculated , defaults to None
    :type gamma: float, optional

    :return: *K* (`np.ndarray`) - Calculated RBF Kernel
    """
    assert len(X.shape) == 2

    if gamma is None:
        gamma = default_gamma(X)

    K = rbf(X, gamma = gamma)
    return K


def local_rbf_kernel(X:np.ndarray, y:np.ndarray, gamma:float=None):
    """ToDo: In original setting of Been Kim, a local kernel in respect to the classes is also applied, which is not feasible here since near hits and misses alread shows class relation.
    However, this could be used for further exploration, maybe sub-classes.

    """
    print('kernel',gamma)
    print(len(X.shape))
    # todo make final representation sparse (optional)
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert np.all(y == np.sort(y)), 'This function assumes the dataset is sorted by y'

    if gamma is None:
        gamma = default_gamma(X)
    K = np.zeros((X.shape[0], X.shape[0]))
    y_unique = np.unique(y)
    print(y_unique) 
    print('gamma', gamma)
    print('range', int(y_unique[-1] + 1))
    for i in range(int(y_unique[-1] + 1)): # compute kernel blockwise for each class
        print('ind', np.where(y == y_unique[i])[0])
        ind = np.where(y == y_unique[i])[0]
        start = ind.min()
        print('start', start)
        end = ind.max() + 1
        print('end', end)
        K[start:end, start:end] = rbf_kernel(X[start:end, :], gamma=gamma)
    return K



if __name__ == "__main__":
    from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn
    test_X = np.random.rand(100, 128)
    print('Testing default gamma')
    assert np.allclose(rbf_kernel(test_X),rbf_kernel_sklearn(test_X.numpy()))
    print('Testing gamma=0.026')
    assert np.allclose(rbf_kernel(test_X, gamma=0.026), rbf_kernel_sklearn(test_X.numpy(), gamma=0.026))
