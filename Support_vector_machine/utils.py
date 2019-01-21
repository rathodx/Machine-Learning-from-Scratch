import numpy as np
import h5py
import itertools
from sklearn.svm import SVC 
from numpy.linalg import norm






def max_frequency(array, classes):
    
    c=np.array(array).astype('uint8')
    m=np.zeros((1,classes))

    for k in range(classes):
        m[0,k]=len(list(np.where(c==k))[0])
       
    return (np.argmax(m, axis=1))





class svm_ovr(object):
    
    def __init__(self, C=1.0, degree=3, gamma=.5, kernel = 'linear', random_state=None): 
       
        self.c=C
        self.de=degree
        self.g=gamma
        self.ker=kernel
        self.ran=random_state
        
        
        
    
       
    def fit(self,features,labels):
        
        X,y=features,labels
 
        self.classes=np.max(y)
    
        self.clf=SVC(kernel = self.ker,random_state = self.ran,C=self.c, gamma=self.g, degree=self.de).fit(X,y)
        self.support_vectors=self.clf.support_vectors_ 
        self.coef=self.clf.dual_coef_
        
    
        

        if (self.classes>1):
       
            self.classifier = []

            for k in range(0, self.classes+1):
                indices=np.where(y==k)
                y_train=np.zeros(y.shape).astype('uint8')
                y_train[indices]=1 
                self.classifier.append(SVC(kernel = self.ker,random_state = self.ran,C=self.c, gamma=self.g, degree=self.de).fit(X,y_train))
                #print((self.classifier))
                
        
                
        else:
            self.classifier=SVC(kernel = self.ker,random_state = self.ran,C=self.c, gamma=self.g, degree=self.de).fit(X,y)
            
    
            

          

        #self.accuracy=1.0*np.sum(predictions==self.labels)/self.features.shape[0]
        #return self.accuracy
    
    def predict(self, features):
        
        X=features
        
        
        
        if(self.classes>1):
            
            prediction=np.zeros((X.shape[0], self.classes+1))
        
            for k in range(0, self.classes+1):

                vector=self.classifier[k].support_vectors_
                #print(vector)
                dual_coeff=self.classifier[k].dual_coef_
                #print(dual_coeff)
                if self.ker=='rbf':
                    for j in range(X.shape[0]):
                        for i in range(0,len(vector)):
#             weights=weights+vector[i,:]*dual_coeff[0,i]
                            prediction[j,k]=prediction[j,k]+np.exp(-self.g*np.power(norm(X[j]-vector[i]),2))*dual_coeff[0,i]
                prediction[:,k]+=self.classifier[k].intercept_
    
                if self.ker=='linear':

                    weights=np.zeros((1,X.shape[1]))

                    for i in range(0,len(vector)):
                        weights=weights+vector[i,:]*dual_coeff[0,i]



                    prediction[:,k]=(np.dot(X, np.transpose(weights))+ self.classifier[k].intercept_ ).reshape(X.shape[0])
                #print(prediction)

            prediction = (np.argmax(prediction, axis=1))
            return prediction
        
        
        else:
            
            prediction=np.zeros((X.shape[0]))
            
            vector=self.classifier.support_vectors_
            #print(vector)
            dual_coeff=self.classifier.dual_coef_
            
            #print(dual_coeff)
            
            if self.ker=='rbf':
                
                for j in range(X.shape[0]):
    
                    for i in range(0,len(vector)):
#             weights=weights+vector[i,:]*dual_coeff[0,i]
                        prediction[j]=prediction[j]+np.exp(-self.g*np.power(norm(X[j]-vector[i]),2))*dual_coeff[0,i]
    
                prediction+=self.classifier.intercept_

            
            
            if self.ker=='linear':
                
                weights=np.zeros((1,X.shape[1]))

                for i in range(0,len(vector)):
                    weights=weights+vector[i,:]*dual_coeff[0,i]

                prediction=(np.dot(X, np.transpose(weights))+ self.classifier.intercept_ ).reshape(X.shape[0])

            for k in range(0, prediction.shape[0]):
                    if prediction[k]>0:
                        prediction[k]=1
                    else:
                        prediction[k]=0

            return prediction.astype('uint8')
        
    def support_vectors(self):
        return self.support_vectors
    
    def coef(self):
        return self.coef
    
    def dec_func(self, X):
        dec_func=self.clf.decision_function(X)
        return dec_func
    
    
    
    
    
class svm_ovo(object):
    
    def __init__(self, C=1.0, degree=3, gamma=.5,kernel = 'linear', random_state=None): 
       
        self.c=C
        self.de=degree
        self.g=gamma
        self.ker=kernel
        self.ran=random_state
        
        
    
       
    def fit(self,features,labels):
        
        X,y=features,labels
        clf=SVC(kernel = self.ker,random_state = self.ran,C=self.c, gamma=self.g, degree=self.de).fit(X,y)
        self.support_vectors=clf.support_vectors_ 
        self.coef=clf.dual_coef_
 
        self.classes=np.max(y)+1
    
        

        if (self.classes>2):
       
            self.classifier = []

            comb=list(itertools.combinations(range(self.classes), 2))

            self.total_comb=len(comb)
            
            self.classifier = []

            for k in range(0, self.total_comb):
                indices=range(self.classes)


                for i in range(2):

                    indices.remove(comb[k][i])

                X_new=np.delete(X, np.where(y==indices[0]),0)
                y_new=np.delete(y, np.where(y==indices[0]),0)

                self.classifier.append(SVC(kernel = self.ker,random_state = self.ran,C=self.c, gamma=self.g, degree=self.de).fit(X_new,y_new))


        else:
            self.classifier=svm_ovr(self.c, self.de, self.g,self.ker, self.ran)
            self.classifier.fit(X,y)
            
        
            

          

        #self.accuracy=1.0*np.sum(predictions==self.labels)/self.features.shape[0]
        #return self.accuracy
    
    def predict(self, features):
        
        X=features
        
        
        
        if(self.classes>2):
            
            predictions=np.zeros((X.shape[0],self.total_comb )).astype('uint8')
            new_predictions=np.zeros((X.shape[0])).astype('uint8')

            for k in range(self.total_comb):
                predictions[:,k]=self.classifier[k].predict(X)


            for k in range(X.shape[0]):
                new_predictions[k]=max_frequency(predictions[k,:], self.classes)
                
             

        
        else:
            
            
                
            new_predictions=np.zeros((X.shape[0])).astype('uint8')
            new_predictions=self.classifier.predict(X)
            
            
        return new_predictions
    
    def support_vectors(self):
        return self.support_vectors
    
    def coef(self):
        return self.coef

