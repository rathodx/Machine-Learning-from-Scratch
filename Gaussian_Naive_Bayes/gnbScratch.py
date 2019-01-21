''' Implementation of Gaussian Naive Bayes from scratch using numpy'''

import numpy as np
import math
import scipy.io


def cal_mean(features):
    variable_mean=np.zeros((1,features.shape[1]))
    variable_mean=np.mean(features, axis=0,dtype=np.float64)
    return variable_mean

def cal_var(features):
    variable_var=np.zeros((1,features.shape[1]))
    variable_var=np.var(features, axis=0,dtype=np.float64)
    return variable_var

def cal_prob(features, num_classes, features_mean, features_var, prob_classes):
    prob=np.zeros((num_classes, features.shape[0]))
    prob[0:num_classes]=(1 / (np.sqrt(2*math.pi*features_var[0:num_classes]) ))   * (np.exp(-(np.power(features-features_mean[0:num_classes],2)/(2*np.power(features_var[0:num_classes],2)))))

    prob=np.multiply(prob,prob_classes)
    prob=np.log10(prob)
    return  np.sum(prob, axis=1)


''' GNB class: has three objects, fit and predict'''

class gnbScratch(object):

    def __init__(self, features=None, labels=None):
        self.features=features

    '''fit() finds the GNB parameters using the training data'''

    def fit(self,features, labels):


        self.num_classes=np.max(labels).astype('uint8')
        self.features_mean=np.zeros((self.num_classes+1,features.shape[1]))
        self.features_var=np.zeros((self.num_classes+1,features.shape[1]))
        self.prob_classes=np.zeros((self.num_classes+1,1), dtype='float64')


        for i in range(0,self.num_classes+1):
            indexes=np.where(labels==i)[0]
            self.features_mean[i]=cal_mean(features[indexes])+.00001
            self.features_var[i]=cal_var(features[indexes])+.00001
            self.prob_classes[i]=1.0*len(indexes)/features.shape[0]


    ''' predict() makes predictions using the trained GNB'''        

    def predict(self, features):



        predictions=np.zeros((features.shape[0]))


        for k in range(0,features.shape[0] ):
            predictions[k]=np.argmax(cal_prob(features[k],self.num_classes+1,self.features_mean,self.features_var, self.prob_classes ))


        return predictions
