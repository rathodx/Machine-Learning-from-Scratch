import numpy as np
import h5py
import json
from numpy import linalg as LA


def read_data(path):

    f=h5py.File(path)
    key = (f.keys())
    X=f[key[0]]
    Y=f[key[1]]
    X = np.array(X, dtype='float64')/np.max(X)
    Y = np.array(Y)
    if Y.ndim>1:

        new_labels=np.zeros((Y.shape[0]))

        for k in range(0,Y.shape[0]):
            new_labels[k]=Y[k].argmax(axis=0)

        return X, new_labels


    else:
        return X,Y


def remove_out(data,la):

    mean=np.mean(data, axis=0)
    S=np.linalg.inv(np.cov(data, rowvar=False))

    feat_len=data.shape[1]
    feat_hei=data.shape[0]
    distance=np.zeros(feat_hei)
    for k in range(feat_hei):

            distance[k]=np.dot(np.dot((data[k]-mean),S),(data[k]-mean).T)

    distance=np.sqrt(distance)
    med_dis=(np.median(distance, axis=0))
    Q1_ind=np.where(distance<med_dis)
    Q2_ind=np.where(distance>med_dis)

    Q1=(np.median(distance[Q1_ind], axis=0))
    Q2=(np.median(distance[Q2_ind], axis=0))



    IQR=Q2-Q1

    lower_end=Q1-1.5*IQR
    high_end=Q2+1.5*IQR

    data_new=data
    la_new=la
    indices=[]
    for i in range(len(distance)):
        if distance[i]>high_end or distance[i]<lower_end:
            indices.append(i)

    data_new=np.delete(data_new, indices, 0)
    la_new=np.delete(la_new, indices, 0)

    return data_new, la_new, indices


def feat_max(filename):
    data = json.loads(open(filename).read())
    data_size=len(data)

    feat_max=0
    for i in range(data_size):
        features=data[i]['X']
        feat_size=len(features)
        if(feat_size>feat_max):
            feat_max=feat_size
    return feat_max


def kpca(features, gamma=.1, n_comp=2):
    row=features.shape[0]
    col=features.shape[1]


    dist=np.zeros((row,row))
    for i in range(row):
        for j in range(row):
            for k in range(col):
                dist[i,j]=dist[i,j]+np.square(features[i,k]-features[j,k])

    k=np.exp(-gamma*dist)
    I=np.ones(k.shape)/row
    k=k-np.dot(I,k)-np.dot(k,I)+np.dot(np.dot(I,k),I)
    eigva, eigvec=LA.eigh(k)
    e_row=eigvec.shape[0]
    e_col=eigvec.shape[1]
    pca=np.zeros((e_row,n_comp))
    for c in range(0, n_comp):
        pca[:,c]=eigvec[:,e_col-1-c]

    return pca


def neighbors(test, train, neigh=1):
    row=train.shape[0]
    row_test=test.shape[0]
    col=train.shape[1]
    dist=np.zeros((row,row_test))
    for i in range(row_test):
        for j in range(row):
            for k in range(col):
                dist[j,i]=dist[j,i]+np.square(test[i,k]-train[j,k])

    index =np.argsort(dist,axis=0)

    return index[0:neigh]
