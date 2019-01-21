import numpy as np
import h5py

def read_data(path):

    f=h5py.File(path)
    X=f['X']
    Y=f['Y']
    X = np.array(X, dtype='float64')/np.max(X)
    
   
    Y = np.array(Y)
    return X,Y

    