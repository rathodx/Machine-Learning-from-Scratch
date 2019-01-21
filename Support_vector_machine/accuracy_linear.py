

from utils import*
from plots import*
import time
from data_utils import*


features,labels=read_data('../data/data_3.h5')




start_time = time.time()
set=np.array([5])


split=[.001, .1, 10, 100, 1000, 10000]



accuracy=np.zeros((len(split)))
print(accuracy.shape)
best_accuracy=0
idx=-1
for i in range(0,len(split)):
    print(i)






    for n in range (0,len(set)):



            pivot=labels.shape[0]//set[n]
            model=svm_ovr(kernel='linear',random_state=0,C=split[i])
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


                accuracy[i] += (np.sum(predictions==labels_test).astype('float32')/(set[n]*pivot))




print(accuracy)
ETA=time.time() - start_time
print(ETA)



fig=plt.figure()
width=.50
y_pos = np.arange(len(split))
plt.bar(y_pos, accuracy, width, color="r")
plt.xticks(y_pos, split,rotation=90)
plt.xlabel('C')
plt.ylabel('Accuracy')


fig.savefig ('./data_1_ovr_linear.png')
plt.show()
