
# coding: utf-8

# In[1]:


import numpy as np
import h5py
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.svm import SVC

from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from data_utils import*
from utils import*
from plots import*


# In[2]:


for k in range(1,6):

    X,y=read_data('../data/data_{}.h5'.format(k))

    scat,scat1=scatter_plot(X,y, 'Dataset {}'.format(k))


    scat.savefig('data_{}.png'.format(k))
    scat1.show()


# In[3]:



for k in range(1,6):

    X,y=read_data('../data/data_{}.h5'.format(k))
    X_row=X.shape[0]
    X_new=np.zeros((X_row,2))



    if(k==1):

        for k in range(0,X_row):
            X_new[k,0]=X[k,0]+X[k,1]
            X_new[k,1]=X[k,0]*X[k,0]+X[k,1]*X[k,1]

        fig=plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_new[y == j, 0], X_new[y == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)
        plt.title('linearly separated dataset 1')
        plt.xlabel('X1')
        plt.ylabel('X2')

        plt.legend()

        X_mesh, y_mesh = X_new, y

        classifier = SVC(kernel = 'linear',  random_state = 0).fit(X_mesh, y_mesh)

        X_1, X_2 = np.meshgrid(np.arange(start = X_mesh[:, 0].min() - 1, stop = X_mesh[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_mesh[:, 1].min() - 1, stop = X_mesh[:, 1].max() + 1, step = 0.01))
        plt.subplot(1,2,2)
        plt.contourf(X_1, X_2, classifier.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape),
                     alpha = 0.75, cmap = ListedColormap(('gray', 'yellow')))
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_mesh[y_mesh == j, 0], X_mesh[y_mesh == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.axis('tight')
        plt.title('Dataset 1 with decision boundary')

        fig.savefig( 'data_1_sepr.png')
        plt.show()



    if k==2:

        for k in range(0,X_row):
            X_new[k,0]=X[k,0]
            X_new[k,1]=np.power(X[k,0]-1,2)+np.power(X[k,1]-.5,2)
            X_new[k,1]=np.sqrt(X_new[k,1])


        fig=plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_new[y == j, 0], X_new[y == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)
        plt.title('linearly separated dataset 2')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()

        X_mesh, y_mesh = X_new, y

        classifier = SVC(kernel = 'linear',  random_state = 0).fit(X_mesh, y_mesh)

        X_1, X_2 = np.meshgrid(np.arange(start = X_mesh[:, 0].min() - 1, stop = X_mesh[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_mesh[:, 1].min() - 1, stop = X_mesh[:, 1].max() + 1, step = 0.01))
        plt.subplot(1,2,2)
        plt.contourf(X_1, X_2, classifier.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape),
                     alpha = 0.75, cmap = ListedColormap(('gray', 'yellow')))
#         plt.xlim(X_1.min(), X_1.max())
#         plt.ylim(X_2.min(), X_2.max())
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_mesh[y_mesh == j, 0], X_mesh[y_mesh == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.title('Dataset 2 with decision boundary')
        fig.savefig( 'data_2_sepr.png')
        plt.show()

    if k==4:

        for k in range(0,X_row):
            X_new[k,0]=X[k,0]
            X_new[k,1]=np.power(X[k,0],2)+np.power(X[k,1],2)
            X_new[k,1]=np.sqrt(X_new[k,1])

        fig=plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_new[y == j, 0], X_new[y == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)
        plt.title('linearly separated dataset 4')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()

        X_mesh, y_mesh = X_new, y

        classifier = SVC(kernel = 'linear',  random_state = 0).fit(X_mesh, y_mesh)

        X_1, X_2 = np.meshgrid(np.arange(start = X_mesh[:, 0].min() - 1, stop = X_mesh[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_mesh[:, 1].min() - 1, stop = X_mesh[:, 1].max() + 1, step = 0.01))
        plt.subplot(1,2,2)
        plt.contourf(X_1, X_2, classifier.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape),
                     alpha = 0.75, cmap = ListedColormap(('gray', 'yellow')))
#         plt.xlim(X_1.min(), X_1.max())
#         plt.ylim(X_2.min(), X_2.max())
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_mesh[y_mesh == j, 0], X_mesh[y_mesh == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.title('Dataset 4 with decision boundary')
        fig.savefig('data_4_sepr.png')
        plt.show()




    if k==5:

        for k in range(0,X_row):
            X_new[k,0]=X[k,0]
            X_new[k,1]=np.power(X[k,0],2)+np.power(X[k,1],2)
            X_new[k,1]=np.sqrt(X_new[k,1])


        fig=plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_new[y == j, 0], X_new[y == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)
        plt.title('linearly separated dataset 5')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()

        X_mesh, y_mesh = X_new, y

        classifier = SVC(kernel = 'linear',  random_state = 0).fit(X_mesh, y_mesh)

        X_1, X_2 = np.meshgrid(np.arange(start = X_mesh[:, 0].min() - 1, stop = X_mesh[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_mesh[:, 1].min() - 1, stop = X_mesh[:, 1].max() + 1, step = 0.01))
        plt.subplot(1,2,2)
        plt.contourf(X_1, X_2, classifier.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape),
                     alpha = 0.75, cmap = ListedColormap(('gray', 'yellow')))
        plt.xlim(X_1.min(), X_1.max())
        plt.ylim(X_2.min(), X_2.max())
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_mesh[y_mesh == j, 0], X_mesh[y_mesh == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.title('Dataset 5 with decision boundary')
        fig.savefig('data_5_sepr.png')
        plt.show()

    if k==3:
        X_new=X


        fig=plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_new[y == j, 0], X_new[y == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)
        plt.title('linearly separated dataset 3')
        plt.xlabel('X1')
        plt.ylabel('X2')

        plt.legend()

        X_mesh, y_mesh = X_new, y

        classifier = SVC(kernel = 'linear',  random_state = 0).fit(X_mesh, y_mesh)

        X_1, X_2 = np.meshgrid(np.arange(start = X_mesh[:, 0].min() - 1, stop = X_mesh[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_mesh[:, 1].min() - 1, stop = X_mesh[:, 1].max() + 1, step = 0.01))
        plt.subplot(1,2,2)
        plt.contourf(X_1, X_2, classifier.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape),
                     alpha = 0.75, cmap = ListedColormap(('gray', 'yellow', 'orange')))
#         plt.xlim(X_1.min(), X_1.max())
#         plt.ylim(X_2.min(), X_2.max())
        for i, j in enumerate(np.unique(y)):


            plt.scatter(X_mesh[y_mesh == j, 0], X_mesh[y_mesh == j, 1],c = ListedColormap(('black', 'red'))(i), label = j)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.axis('tight')
        plt.title('Dataset 3 with decision boundary')
        fig.savefig('data_3_sepr.png')
        plt.show()
