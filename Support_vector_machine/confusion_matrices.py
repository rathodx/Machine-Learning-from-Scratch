
# coding: utf-8

# In[1]:


import h5py
from plots import*
from utils import*
from data_utils import*
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# In[2]:


ker=[ 'linear','rbf']
data=['data_1.h5','data_2.h5','data_3.h5','data_4.h5','data_5.h5','data_6.h5','data_7.h5','data_8.h5']
C_range=[.01,1,100]
gamma_range=[.1,1,10]


# In[3]:


for k in ker:
    for d in data:

        print('Processing {} with {} kernel'.format(d, k))


        features, labels=read_data('../data/'+d)
        pivot=features.shape[0]//3

        if features.shape[1]>2:
            tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=5000)
            features_new = tsne.fit_transform(features, labels)

            features_test=features_new[0:pivot]
            labels_test=labels[0:pivot]
            labels_test=labels_test.astype('uint8')

            features_train=features_new[pivot:features.shape[0]]
            labels_train=labels[pivot:features.shape[0]]
            labels_train=labels_train.astype('uint8')

        else:

            features_test=features[0:pivot]
            labels_test=labels[0:pivot]
            labels_test=labels_test.astype('uint8')

            features_train=features[pivot:features.shape[0]]
            labels_train=labels[pivot:features.shape[0]]
            labels_train=labels_train.astype('uint8')



        accuracy=np.zeros((len(C_range),len(gamma_range)))

        best_accuracy=0
        n=0
        fig=plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=.96)
        for i in range(0,len(C_range)):

                if k=='rbf':
                    for m in range (0, len(gamma_range)):


                            n=n+1


                            print(m,i)

                            model=svm_ovr(kernel=k,random_state=0,C=C_range[i],gamma=gamma_range[m])
                            model.fit(features_train, labels_train)
                            predictions=model.predict(features_test)
#                             classifier=svm_ovo(kernel = k, C=5000, random_state = 0)


                            prediction=model.predict(features_test)
                            accuracy= np.sum(predictions==labels_test).astype('float32')/pivot





                            mat,classes=create_and_plot_cm(labels_test, prediction)
                            plt.subplot(3, 3, n )
                            plt.imshow(mat,  cmap=plt.cm.Blues)
                            plt.xlabel('Predicted labels')
                            plt.ylabel('True labels')
                            plt.title('C={},  gamma={}'.format(C_range[i], gamma_range[m]))


                            marks = np.arange(len(classes))

                            plt.xticks(marks, classes)
                            plt.yticks(marks, classes)

                            for l in range(mat.shape[0]):
                                for j in (range(mat.shape[1])):
                                    plt.text(j, l, format(mat[l, j], "d"),
                                             horizontalalignment="center")





#

                else:

                        n=n+1



                        model=svm_ovr(kernel=k,random_state=0,C=C_range[i])
                        model.fit(features_train, labels_train)
                        predictions=model.predict(features_test)
#                             classifier=svm_ovo(kernel = k, C=5000, random_state = 0)


                        prediction=model.predict(features_test)
                        accuracy= np.sum(predictions==labels_test).astype('float32')/pivot




                        mat,classes=create_and_plot_cm(labels_test, prediction)
                        plt.subplot(1, 3, n )
                        plt.imshow(mat,  cmap=plt.cm.Blues)
                        plt.xlabel('Predicted labels')
                        plt.ylabel('True labels')
                        plt.title('C={}, kernel=linear, \n Accuracy={}'.format(C_range[i], accuracy))


                        marks = np.arange(len(classes))

                        plt.xticks(marks, classes)
                        plt.yticks(marks, classes)

                        for l in range(mat.shape[0]):
                            for j in (range(mat.shape[1])):
                                plt.text(j, l, format(mat[l, j], "d"),
                                         horizontalalignment="center")


        if k=='linear':
            fig.savefig('./' + d + '_cm_linear.png')
        else:
             fig.savefig('./' + d + '_cm_rbf.png')

        plt.show()





#


# In[8]:


for d in data:
    print(d)


# In[14]:


k=1
print('k is {}'.format(k))


# In[ ]:
