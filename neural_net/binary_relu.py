
# coding: utf-8

# In[1]:


import numpy as np
import h5py
from data_utils import*
import matplotlib.pyplot as plt
from nn_utils import*


# In[2]:


data, lab=read_data('./data/dataset_partA.h5')
plt.figure(figsize=(8,6))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('label: {}'.format(lab[i]))
    plt.imshow(data[i])
plt.show()
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


# In[5]:


x=new_data
y=new_labels
print(x.shape)


# In[6]:


lr=.001
epoch=60

layers=[x.T.shape[0],100,50,1]
total_layers=len(layers)
weights=[]
for l in range(total_layers-1):
    weights.append(np.random.normal(size=(layers[l+1],layers[l])))


# In[7]:


epochs=np.zeros((epoch,1))
folds=5
x_train, y_train, x_test, y_test=make_folds(x,y,folds)
# print(len(x_train[4]))
# print(y_train[4].shape)
# print(x_test[4].shape)
# print(y_test[4].shape)


# In[8]:


for f in range(folds):

    x=x_train[f]
    y=y_train[f]

    x_val=x_test[f]
    y_val=y_test[f]

    x=x.T
    y=y.T


    x_val=x_val.T
    y_val=y_val.T

    output=np.zeros(y.shape)
    val_output=np.zeros(y_val.shape)
#     fig=plt.figure(figsize=(8,10))
    accuracy=np.zeros((epoch,1))
    val_accuracy=np.zeros((epoch,1))
    train_loss=np.zeros((epoch,1))

    weights=[]
    for l in range(total_layers-1):
        weights.append(np.random.normal(size=(layers[l+1],layers[l])))


    for e in range(epoch):

        x,y=shuffle_data(x,y)
        train_running_loss=0.0

        epochs[e]=e

        for n in range(x.shape[1]):

            '''training'''

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
                    output[0,n]=1
            else:
                    output[0,n]=0

#             train_running_loss=train_running_loss+(-(y[0,n]*np.log(layers_output[l])+(1-y[0,n])*np.log(1-layers_output[l])))



            derivatives=cal_derivatives(layers_output, weights, y[0,n],layers, 'relu')

            weights=update_weights(weights,derivatives, lr)


        corrects=np.sum(y == output).astype('float')
        accuracy[e]=corrects/x.shape[1]
        accuracy[e]=np.round(accuracy[e],5)
#         train_loss[e]=train_running_loss[0,0]/x.shape[1]
#         train_loss[e]=np.round(train_loss[e],5)



        '''validation'''

        for n1 in range(x_val.shape[1]):

            val_layers_output=[]
            a1=x_val[:,n1]
            a1=a1.reshape((a.shape[0],1))
            val_layers_output.append(a1)



            for l1 in range(len(weights)):

                val_layer_z=np.dot(weights[l1],val_layers_output[l1])
                val_layer_z=np.reshape(val_layer_z, (layers[l1+1],1))

                if (l1 != (len(weights)-1)):
                    val_layer_a=relu(val_layer_z)

                else:
                    val_layer_a=1/(1+np.exp(-val_layer_z))



                val_layers_output.append(val_layer_a)





            if val_layers_output[-1]>=.5:
                    val_output[0,n1]=1
            else:
                    val_output[0,n1]=0



        val_corrects=np.sum(y_val == val_output).astype('float')
        val_accuracy[e]=val_corrects/x_val.shape[1]
        val_accuracy[e]=np.round(val_accuracy[e],5)






        print('fold {}: epoch {}/{}: train_accuracy {}: val_accuracy {}'.format(f+1, e,epoch-1,accuracy[e,0], val_accuracy[e,0]))

    fig=plt.figure(figsize=(8,6))
    plt.style.use('bmh')
    plt.plot(epochs,accuracy,'r', epochs, val_accuracy, 'g')
    plt.title('model accuracy for partA with relu fold {}'.format(f+1))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    fig.savefig('./fold_' + str(f+1) + '_partA_relu_accuracy.png')
    plt.show()








# In[9]:


np.savez('weights_partA_relu', weights)
