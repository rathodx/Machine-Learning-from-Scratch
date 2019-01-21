
# coding: utf-8

# In[1]:


import numpy as np
import h5py
from data_utils import*
import matplotlib.pyplot as plt
from mnist import*
from nn_utils import*


# In[2]:


data=list(read(dataset = "training", path = './data/'))

total=len(data)

train_data=np.zeros((total,784))
train_label=np.zeros((total,10)).astype('uint8')


for i in range(total):
    label, pixel=data[i]
    train_data[i]=pixel.ravel()
    train_label[i,label]=1

train_data = np.array(train_data, dtype='float64')/np.max(train_data)



plt.figure(figsize=(8,6))

for i in range(12):
    plt.subplot(3,4,i+1)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('label: {}'.format(np.argmax(train_label[i], axis=0)))
    plt.imshow(np.reshape(train_data[i], (28,28)))
plt.show()


# In[3]:


for i in range(train_data.shape[0]):
    mean=np.mean(train_data[i])
    index=np.where(train_data[i]==0)
    train_data[i,index]=mean


# In[4]:


x=train_data
y=train_label
print(y.shape)


# In[5]:


lr=.001
epoch=60

layers=[x.T.shape[0],100,50,10]
total_layers=len(layers)
weights=[]
for l in range(total_layers-1):
    weights.append(np.random.normal(size=(layers[l+1],layers[l])))


# In[6]:


epochs=np.zeros((epoch,1))
folds=5
x_train, y_train, x_test, y_test=make_folds(x,y,folds)
print((x_train[0].shape))
print(y_train[4].shape)
print(x_test[4].shape)
print(y_test[4].shape)


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

        train_corrects=0.0
        val_corrects=0.0

        for n in range(x.shape[1]):

            '''training'''

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



#             train_running_loss=train_running_loss+(-(y[0,n]*np.log(layers_output[-1])+(1-y[0,n])*np.log(1-layers_output[-1])))



            derivatives=cal_derivatives(layers_output, weights, b,layers, 'relu')

            weights=update_weights(weights,derivatives, lr)


#         corrects=np.sum(y_1 == output).astype('float')
        accuracy[e]=train_corrects/x.shape[1]
        accuracy[e]=np.round(accuracy[e],5)
#         train_loss[e]=train_running_loss[0,0]/x.shape[1]
#         train_loss[e]=np.round(train_loss[e],5)



        '''validation'''

        for n1 in range(x_val.shape[1]):

            val_layers_output=[]
            a1=x_val[:,n1]
            a1=a1.reshape((a.shape[0],1))

            b1=y_val[:,n1]
            b1=b1.reshape((b1.shape[0],1))
            val_layers_output.append(a1)



            for l1 in range(len(weights)):

                val_layer_z=np.dot(weights[l1],val_layers_output[l1])
                val_layer_z=np.reshape(val_layer_z, (layers[l1+1],1))

                if (l1 != (len(weights)-1)):
                    val_layer_a=relu(val_layer_z)

                else:
                    s=np.sum(np.exp(val_layer_z))
                    val_layer_a=(np.exp(val_layer_z))/s



                val_layers_output.append(val_layer_a)





            arg=np.argmax(val_layers_output[-1], axis=0)[0]
            arg1=np.argmax(b1, axis=0)[0]
            if (arg==arg1):
                val_corrects=val_corrects+1



#         val_corrects=np.sum(y_val == val_output).astype('float')
        val_accuracy[e]=val_corrects/x_val.shape[1]
        val_accuracy[e]=np.round(val_accuracy[e],5)






        print('fold {}: epoch {}/{}: train_accuracy {}: val_accuracy {}'.format(f+1, e,epoch-1,accuracy[e,0], val_accuracy[e,0]))

    fig=plt.figure(figsize=(8,6))
    plt.style.use('bmh')
    plt.plot(epochs,accuracy,'r', epochs, val_accuracy, 'g')
    plt.title('model accuracy for MNIST with ReLu and softmax fold {}'.format(f+1))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    fig.savefig('./fold_' + str(f+1) + '_relu_softamx_accuracy.png')
    plt.show()





# In[9]:


np.savez('weights_MNIST_relu', weights)
