''' Implementation of Logistic Regression from scratch using numpy'''


import numpy as np
import math
import scipy.io
from numpy.linalg import inv

''' LogRegScr class: has three objects, fit and predict'''
class LogRegScr(object):

    def __init__(self, features=None, labels=None):
        self.features=features
        self.labels=labels


    def fit(self,features, labels, decay, epochs):
        features=np.insert(features,0,1,axis=1)

        data_row=features.shape[0]
        data_col=features.shape[1]
        self.weights=np.zeros(data_col)
        lam=1.0*1/decay
        hes_reg=(lam/data_row)*np.diag(np.ones(data_col))

        for k in range(0,epochs):
            grad_reg=(lam/data_row)*self.weights
            ini_pred=np.dot(features,self.weights)
            pred=1/(1+np.exp(-ini_pred))
            gradient=(1/data_row)*(np.dot(np.transpose(features),(pred-labels)))+grad_reg
            hessian=(1/data_row)*(np.dot(np.dot(np.transpose(features), np.diag(pred)),np.dot(np.diag(1-pred), features))) +\
            hes_reg
            self.weights=self.weights-np.dot(inv(hessian),gradient)


    def predict(self,features):

        features=np.insert(features,0,1,axis=1)
        data_row=features.shape[0]

        ini_pred=np.dot(features,self.weights)
        pred=1/(1+np.exp(-ini_pred))


        predictions=np.zeros((data_row))

        for k in range(0,data_row):
            if pred[k]>=.5:
                predictions[k]=1

        return predictions
