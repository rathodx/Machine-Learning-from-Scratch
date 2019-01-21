import numpy as np
import h5py
import json
from numpy import linalg as LA


def read_data(path):

    f=h5py.File(path)
    key = (f.keys())    
    X=f[key[0]]
    Y=f[key[1]]
    X = np.array(X, dtype='float64')/np.max(X)
    Y = np.array(Y)
    if Y.ndim>1:
         
        new_labels=np.zeros((Y.shape[0]))

        for k in range(0,Y.shape[0]):
            new_labels[k]=Y[k].argmax(axis=0)
            
        return X, new_labels
            
            
    else:
        return X,Y
    
    
def make_folds(features, labels, n):

    pivot=labels.shape[0]//n
    
    features_train=np.zeros((features.shape[0]-pivot,features.shape[1]))
    labels_train=np.zeros((labels.shape[0]-pivot))
    features_test=np.zeros((pivot,features.shape[1]))
    labels_test=np.zeros((pivot))
    
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    
    for k in range(0,n):

        features_train=np.delete(features, np.s_[pivot*k:pivot*(k+1)],0)
        x_train.append(features_train)
        labels_train=np.delete(labels, np.s_[pivot*k:pivot*(k+1)],0)
        y_train.append(labels_train)

        features_test=features[pivot*k:pivot*(k+1)]
        x_test.append(features_test)
        labels_test=labels[pivot*k:pivot*(k+1)]
        y_test.append(labels_test)
        
    return x_train, y_train, x_test, y_test
    

    
def shuffle_data(x,y):
    
        x1=x.T
        y1=y.T
        perm = np.random.permutation(len(x1))
        x1 = x1[perm]
        y1 = y1[perm]
        
        return x1.T,y1.T
    
    