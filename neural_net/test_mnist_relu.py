
# coding: utf-8

# In[1]:


import numpy as np
import h5py
from data_utils import*
import matplotlib.pyplot as plt
from nn_utils import*
from mnist import*


# In[2]:


data=list(read(dataset = "testing", path = './data/'))

total=len(data)

train_data=np.zeros((total,784))
train_label=np.zeros((total,10)).astype('uint8')


for i in range(total):
    label, pixel=data[i]
    train_data[i]=pixel.ravel()
    train_label[i,label]=1


print(train_label[0])

data=list(read(dataset = "testing", path = './data/'))

total=len(data)

test_data=np.zeros((total,784))
test_label=np.zeros((total,10)).astype('uint8')
for i in range(total):
    label, pixel=data[i]
    test_data[i]=pixel.ravel()
    test_label[i,label]=1

print(test_label.shape)

train_data = np.array(train_data, dtype='float64')/np.max(train_data)
test_data = np.array(test_data, dtype='float64')/np.max(test_data)


print(test_data[0].shape)



# In[3]:


for i in range(train_data.shape[0]):
    mean=np.mean(train_data[i])
    index=np.where(train_data[i]==0)
    train_data[i,index]=mean
layers=[train_data.T.shape[0],100,50,10]


# In[4]:


npzfile = np.load('weights_MNIST_relu.npz')
weights=npzfile['arr_0']


# In[5]:


x=test_data.T
y=test_label.T
output=np.zeros((train_label.shape[0],1))
train_corrects=0.0
for n in range(x.shape[1]):
        layers_output=[]
        a=x[:,n]
        a=a.reshape((a.shape[0],1))
        b=y[:,n]
        b=b.reshape((b.shape[0],1))
        layers_output.append(a)

        for l in range(len(weights)):



            layer_z=np.dot(weights[l],layers_output[l])
            layer_z=np.reshape(layer_z, (layers[l+1],1))

            if (l != (len(weights)-1)):
                layer_a=relu(layer_z)

            else:
                s=np.sum(np.exp(layer_z))
                layer_a=(np.exp(layer_z))/s

            layers_output.append(layer_a)




        arg=np.argmax(layers_output[-1], axis=0)[0]
        arg1=np.argmax(b, axis=0)[0]
        if (arg==arg1):
            train_corrects=train_corrects+1


accuracy=train_corrects/x.shape[1]
accuracy=np.round(accuracy,5)
print('test accuracy is {}'.format(accuracy))
