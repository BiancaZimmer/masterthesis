"""**MMD Critic** - This includes function which are original created by Been Kim for selecting prototypes.
"""

from pathlib import Path
import numpy as np
import scipy
from kernels import rbf_kernel, local_rbf_kernel
from feature_extractor import FeatureExtractor
from dataentry import DataEntry

from sklearn import preprocessing



def compute_rbf_kernel(X: np.ndarray, gamma:float=None):
    """Original function that use function self-create function to calculate the RBF Kernel.

    :param X: Data of DataSet which should transformed to a numpy array first
    :type X: np.ndarray
    :param gamma: Pass a gamma value, else the default gamma value will be calculated , defaults to None
    :type gamma: float, optional

    :return: *K* (`np.ndarray`) - Calculated RBF Kernel
    """
    K = rbf_kernel(X, gamma)
    return K

    


def select_prototypes(K: np.ndarray, num_prototypes:int):
    """MMD-bases prototype selection implemented by Been Kim.

    :param K: RBF Kernel of the data
    :type K: np.ndarray
    :param num_prototypes: Number of prototypes that should be selected
    :type num_prototypes: int

    :return: *selected_in_order* (`np.ndarray`) - Array/List of indices of prototypes used to select the corresponding DataEntry objects
    """
    sample_indices = np.arange(0, K.shape[0])
    num_samples = sample_indices.shape[0]
    
    
    colsum = 2*np.sum(K, axis=0) / num_samples
    is_selected = np.zeros_like(sample_indices)
    selected = sample_indices[is_selected > 0]

    for i in range(num_prototypes):
        candidate_indices = sample_indices[is_selected == 0]
        s1 = colsum[candidate_indices]
        
        if selected.shape[0] == 0:
            s1 -= np.abs(np.diagonal(K)[candidate_indices])
            #print("K diag: ", np.shape(np.abs(np.diagonal(K)[candidate_indices])))
        else:
            temp = K[selected, :][:, candidate_indices]
            s2 = np.sum(temp, axis=0) *2 + np.diagonal(K)[candidate_indices] 
            s2 /= (len(selected) + 1)
            s1 -= s2
            
    
        best_sample_index = candidate_indices[np.argmax(s1)]
        #print("max %f" %np.max(s1))
        #print(best_sample_index)
        is_selected[best_sample_index] = i + 1
        selected = sample_indices[is_selected > 0]
      
    selected_in_order = selected[is_selected[is_selected > 0].argsort()]
    return selected_in_order


def select_criticisms(K:np.ndarray, prototype_indices:np.ndarray, num_criticisms:int, regularizer=None):
    """ToDo: The implementation of criticism can be found here.  However, this is not used and might need to be changed slightly to make it work.

    """
    # if sparse matrix
    if scipy.sparse.issparse(K):
      K = K.todense()
      print("[!] sparse K matrix was converted to dense matrix")
    
    prototype_indices = prototype_indices.copy()
    available_regularizers = {None, 'logdet', 'iterative'}
    assert regularizer in available_regularizers, f'Unknown regularizer: "{regularizer}". Available regularizers: {available_regularizers}'
    
    sample_indices = np.arange(0, K.shape[0])
    num_samples = sample_indices.shape[0]
    
    is_selected = np.zeros_like(sample_indices)
    is_selected[prototype_indices] = num_criticisms + 1 # is_selected > 0 indicates either selected (1 to num_criticisms) or prototype (if num_criticisms +1)
    selected = sample_indices[is_selected > 0]
    
    colsum = np.sum(K, axis = 0)/num_samples
    inverse_of_prev_selected = None
    for i in range(num_criticisms):
        candidate_indices = sample_indices[is_selected == 0]
        s1 = colsum[candidate_indices]
    
        temp = K[prototype_indices, :][:, candidate_indices]
        s2 = temp.sum(0)
        s2 /= prototype_indices.shape[0]
        s1 = np.abs(s1 - s2)
    
        if regularizer == 'logdet':
            if inverse_of_prev_selected is not None: # first call has been made already
                temp = K[selected, :][:, candidate_indices]
                temp2 = np.array(np.dot(inverse_of_prev_selected, temp))
                reg = temp2 * temp
                regcolsum =  np.sum(reg, axis=0)
                reg = np.log(np.abs(np.diagonal(K)[candidate_indices] - regcolsum))
                s1 += reg
            else:
                s1 -= np.log(np.abs(np.diagonal(K)[candidate_indices])) 
    
        best_sample_index = candidate_indices[np.argmax(s1)]
        is_selected[best_sample_index] = i + 1
    
        selected = sample_indices[(is_selected > 0) & (is_selected != (num_criticisms + 1))]
        
        if regularizer == 'iterative':
            prototype_indices = np.append(prototype_indices, best_sample_index)
    
        if regularizer == 'logdet':
            KK = K[selected,:][:,selected]
            inverse_of_prev_selected = np.linalg.inv(KK) # shortcut
    
    selected_in_order = selected[is_selected[(is_selected > 0) & (is_selected != (num_criticisms + 1))].argsort()]
    return selected_in_order
