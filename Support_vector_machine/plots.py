from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(X,y,z=None):
    
    fig=plt.figure()
    
    for i, j in enumerate(np.unique(y)):
   
  
        plt.scatter(X[y == j, 0], X[y == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)
    plt.title(z)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    return fig, plt


def create_and_plot_cm(y_train, prediction):
    
    cl=np.max(y_train)
    classes=[]
    mat=np.zeros((cl+1,cl+1)).astype('int64')
    for k in range(cl+1):
        classes.append(str(k))
        ind=np.where(y_train==k)
        p1=prediction[ind]
        for i in range(cl+1):
            mat[k,i]=len(list(np.where(p1==i)[0]))

           
    
    
 
    
    return mat, classes

