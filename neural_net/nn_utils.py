
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def cal_derivatives(layers_output, weights, y,layers, activation='sigmoid'):
        
    
    total_layers=len(layers_output)
    derivatives=[]
    delta=.0000000001
    
    for l in range(total_layers):
        
        if (l==0):
            
            layer_grad=(layers_output[total_layers-1-l]-y)*layers_output[total_layers-2-l].T
            derivatives.append(layer_grad)
            
            
            
        else:

            layer_grad=derivatives[l-1].T*weights[total_layers-1-l].T
            layer_grad=np.sum(layer_grad, axis=1)
            layer_grad=np.reshape(layer_grad,(layers[total_layers-1-l],1))
            if activation=='sigmoid':
                layer_grad=layer_grad*(1-layers_output[total_layers-1-l])
            if activation=='relu':
                layer_grad=layer_grad*(1/(delta+layers_output[total_layers-1-l]))
            layer_grad=np.dot(layer_grad, layers_output[total_layers-2-l].T)
            derivatives.append(layer_grad)

    return derivatives


# In[3]:


def update_weights(weights, derivatives, lr):
    total_layers=len(weights)
    for l in range(total_layers):
        weights[total_layers-l-1]=weights[total_layers-l-1]-lr*derivatives[l]
        
    return weights


# In[4]:


def relu(arr):
    for k in range(arr.shape[0]):
        if arr[k]<=0:
            arr[k]=0
            
    return arr


# In[ ]:




