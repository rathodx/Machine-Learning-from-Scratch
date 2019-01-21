


from utils import*
from plots import*
from data_utils import*
import time





features,labels=read_data('../data/data_3.h5')

start_time = time.time()
set=np.array([5])


split=np.logspace(-1, 10, 12)

depth=np.logspace(-8, 3, 12)

accuracy=np.zeros((len(split),len(split)))
print(accuracy.shape)
best_accuracy=0
idx=-1
for i in range(0,len(split)):






    for n in range (0,len(set)):

        for m in range (0, len(depth)):





            pivot=labels.shape[0]//set[n]
            model=svm_ovo(kernel='rbf',random_state=0,C=split[i],gamma=depth[m])
            features_train=np.zeros((features.shape[0]-pivot,features.shape[1]))
            labels_train=np.zeros((labels.shape[0]-pivot))
            features_test=np.zeros((pivot,features.shape[1]))
            labels_test=np.zeros((pivot))

            for k in range(0,set[n]):

                features_train=np.delete(features, np.s_[pivot*k:pivot*(k+1)],0)
                labels_train=np.delete(labels, np.s_[pivot*k:pivot*(k+1)],0)

                features_test=features[pivot*k:pivot*(k+1)]
                labels_test=labels[pivot*k:pivot*(k+1)]
                model.fit(features_train, labels_train)

                predictions=model.predict(features_test)
                predictions=predictions

                accuracy[i,m] += (np.sum(predictions==labels_test).astype('float32')/(set[n]*pivot))




ETA=time.time() - start_time
print(ETA)




fig=plt.figure(figsize=(12, 10))
plt.imshow(accuracy, cmap=plt.cm.RdBu_r)
marks = np.arange(len(depth))
plt.xticks(marks, split,rotation=90)
plt.yticks(marks, depth)
plt.xlabel('C')
plt.ylabel('gamma')
plt.title('Accuracy')
for i in range(accuracy.shape[0]):
        for j in (range(accuracy.shape[1])):
            plt.text(j, i, (accuracy[i, j]),horizontalalignment="center")


fig.savefig( './data_3_rbf_ovo.png')


plt.show()
