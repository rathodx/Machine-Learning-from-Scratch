
# coding: utf-8

# In[1]:


import numpy as np
import h5py
from data_utils import*
import matplotlib.pyplot as plt
from nn_utils import*


# In[2]:


data, lab=read_data('./data/dataset_partA.h5')

print(data.shape)
print(lab.shape)


# In[3]:


new_data=np.zeros((data.shape[0],data.shape[1]*data.shape[2]))
for i in range(data.shape[0]):
    new_data[i]=data[i].ravel()
print(new_data.shape)

new_labels=np.zeros((lab.shape[0], 1)).astype('uint8')


for k in range(lab.shape[0]):
    if lab[k]==7:
        new_labels[k]=0
    else:
        new_labels[k]=1



# In[4]:


for i in range(data.shape[0]):
    mean=np.mean(new_data[i])
    index=np.where(new_data[i]==0)
    new_data[i,index]=mean
layers=[new_data.T.shape[0],100,50,1]


# In[5]:


npzfile = np.load('weights_partA_relu.npz')
weights=npzfile['arr_0']


# In[6]:


x=new_data.T
y=new_labels.T
output=np.zeros((new_labels.shape[0],1))
for n in range(x.shape[1]):
    layers_output=[]
    a=x[:,n]
    a=a.reshape((a.shape[0],1))
    layers_output.append(a)

    for l in range(len(weights)):



        layer_z=np.dot(weights[l],layers_output[l])
        layer_z=np.reshape(layer_z, (layers[l+1],1))
        if (l != (len(weights)-1)):
                    layer_a=relu(layer_z)

        else:
                    layer_a=1/(1+np.exp(-layer_z))

        layers_output.append(layer_a)



    if layers_output[-1]>=.5:
        output[n]=1
    else:
        output[n]=0

corrects=np.sum(y.T == output).astype('float')
accuracy=corrects/x.shape[1]
accuracy=np.round(accuracy,5)
print('test accuracy is {}'.format(accuracy))
