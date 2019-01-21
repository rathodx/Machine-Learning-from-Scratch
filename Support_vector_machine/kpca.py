
# coding: utf-8

# In[1]:


import numpy as np
from data_utils import*
import matplotlib.pyplot as plt
from numpy import linalg as LA
from utils import*
import time


# In[2]:


size=5
f=[]
l=[]
for k in range(1,size+1):
    fe, la=read_data('../data/data_{}.h5'.format(k))

    f.append(fe)
    l.append(la)

feat, lab=f[0],l[0]
for k in range(1,size):
    feat=np.concatenate((feat, f[k]), axis=0)
    lab=np.concatenate((lab, l[k]), axis=0)

ind_0=list(np.where(lab==0)[0])
ind_1=list(np.where(lab==1)[0])
np.random.shuffle(ind_0)
np.random.shuffle(ind_1)

ind_0=ind_0[0:150]
ind_1=ind_1[0:150]

index=np.concatenate((ind_0, ind_1), axis=0)
np.random.shuffle(index)
features, labels=feat[index], lab[index]




set1=np.array([5])


gamma_range=np.logspace(-1, 10, 12)

neigh_range=[3,5,10]

accuracy=np.zeros((len(neigh_range),len(gamma_range)))
ETA=np.zeros((len(neigh_range),len(gamma_range)))
print(accuracy.shape)
best_accuracy=0
idx=-1
for i in range(0,len(gamma_range)):
    start_time = time.time()






    for n in range (0,len(set1)):

        for m in range (0, len(neigh_range)):

            print(i,m)





            pivot=labels.shape[0]//set1[n]
            features_train=np.zeros((features.shape[0]-pivot,features.shape[1]))
            labels_train=np.zeros((labels.shape[0]-pivot))
            features_test=np.zeros((pivot,features.shape[1]))
            labels_test=np.zeros((pivot))

            for k in range(0,set1[n]):

                features_train=np.delete(features, np.s_[pivot*k:pivot*(k+1)],0)
                features_train=kpca(features_train, gamma=gamma_range[i], n_comp=2)
                labels_train=np.delete(labels, np.s_[pivot*k:pivot*(k+1)],0)

                features_test=features[pivot*k:pivot*(k+1)]
                features_test=kpca(features_test, gamma=gamma_range[i], n_comp=2)
                labels_test=labels[pivot*k:pivot*(k+1)]



                pr=neighbors(test=features_test, train=features_train, neigh=neigh_range[m])

                predictions=np.zeros((features_test.shape[0]))

                for p in range(0, len(predictions)):
                    predictions[p]=max_frequency(labels_train[pr[:,p]], 2)





                accuracy[m,i] += round((np.sum(predictions==labels_test).astype('float32')/(set1[n]*pivot)),2)
                ETA[m,i]+=round(time.time() - start_time,2)







accuracy1=round(accuracy[1,0],2)
print(accuracy1)


# In[5]:


fig=plt.figure(figsize=(12, 10))
plt.imshow(accuracy, cmap=plt.cm.RdBu_r)
marks = np.arange(len(gamma_range))
marks1 = np.arange(len(neigh_range))
plt.xticks(marks, gamma_range,rotation=90)
plt.yticks(marks1, neigh_range)
plt.xlabel('gamma')
plt.ylabel('neighbors')
plt.title('Accuracy')
for i in range(accuracy.shape[0]):
        for j in (range(accuracy.shape[1])):
            plt.text(j, i, (accuracy[i, j]),horizontalalignment="center")



fig.savefig('kpca.png')


plt.show()
# In[7]:


fig=plt.figure(figsize=(12, 10))
plt.imshow(accuracy, cmap=plt.cm.RdBu_r)
marks = np.arange(len(gamma_range))
marks1 = np.arange(len(neigh_range))
plt.xticks(marks, gamma_range,rotation=90)
plt.yticks(marks1, neigh_range)
plt.xlabel('gamma')
plt.ylabel('neighbors')
plt.title('Time')
for i in range(accuracy.shape[0]):
        for j in (range(accuracy.shape[1])):
            plt.text(j, i, (ETA[i, j]),horizontalalignment="center")




fig.savefig('kpca_Time.png')
plt.show()

# In[ ]:
