''' Code for plotting the decision boundary returned by support vector machine. Applicable only for binary class datasets'''





from plots import*
from utils import*
from data_utils import*
from sklearn.manifold import TSNE




'''Kernels to be used '''

ker=[ 'linear','rbf']

'''datasets: replace with your own binary classes dataset'''

data=['data_1.h5','data_2.h5']
C_range=[.01,1,100]
gamma_range=[.1,1,10]


# In[3]:


for k in ker:
    for d in data:

        print('Processing {} with {} kernel'.format(d, k))


        features, labels=read_data('../data/' + d)
        classes=np.max(labels).astype('uint8')+1
        pivot=features.shape[0]//3

        '''Use TSNE to reduce the no. of features to two'''
        if (features.shape[1])>2:

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
        fig=plt.figure(figsize=(12, 8))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)

        '''Iterate over range of C and gamma'''

        for i in range(0,len(C_range)):

                if k=='rbf':

                    for m in range (0, len(gamma_range)):

                            print(i,m)



                            '''svm_ovr is self implemented version of SVM. It uses one-vs-rest approach to handle multiclass data. Another version is svm_ovo
                             which uses one-vs-one approach for multiclass data. It uses sklearn library to fit the model. Rest operations are self implemented'''

                            n=n+1
                            model = svm_ovr(kernel = k, C=C_range[i],gamma=gamma_range[m],random_state = 0)
                            model.fit(features_train, labels_train)

                            predictions=model.predict(features_test)
#


                            prediction=model.predict(features_test)
                            accuracy= np.sum(predictions==labels_test).astype('float32')/pivot

                            X_set,y_set=features_train,labels_train
                            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

                            plt.subplot(3,3,n)
                            plt.title('C={},  gamma={}'.format(C_range[i], gamma_range[m]))

                            Z = (model.dec_func(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
                            plt.contour(X1, X2,Z,  colors='black', levels=[-1, 0, 1], alpha=1.0,
                                       linestyles=['--', '-', '--'])
                            plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], s=50,
                                       facecolors='none', zorder=50, edgecolors='white')
                            plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, zorder=10,
                                            edgecolors='black')

                            Z1=model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
                            plt.pcolormesh(X1, X2, Z1,cmap= ListedColormap(('gray', 'red')))
                            plt.xticks(())
                            plt.yticks(())
                            plt.axis('tight')




                else:
                            n=n+1
                            plt.subplot(1,3,n)

                            model = svm_ovr(kernel = k, C=C_range[i],random_state = 0)
                            model.fit(features_train, labels_train)

                            predictions=model.predict(features_test)



                            prediction=model.predict(features_test)
                            accuracy= np.sum(predictions==labels_test).astype('float32')/pivot

                            X_set,y_set=features_train,labels_train
                            X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                                                 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))


                            plt.title('C={}, accuracy={}'.format(C_range[i], accuracy))

                            Z = (model.dec_func(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
                            plt.contour(X1, X2,Z,  colors='black', levels=[-1, 0, 1], alpha=1.0,
                                       linestyles=['--', '-', '--'])
                            plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], s=50,
                                       facecolors='none', zorder=50, edgecolors='white')
                            plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, zorder=10,
                                            edgecolors='black')

                            Z1=model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
                            plt.pcolormesh(X1, X2, Z1,cmap= ListedColormap(('gray', 'red')))
                            plt.xticks(())
                            plt.yticks(())


        if k=='linear':
            fig.savefig('./' + d + '_hp_linear.png')
        else:
             fig.savefig('./' + d + '_hp_rbf.png')

        plt.show()
