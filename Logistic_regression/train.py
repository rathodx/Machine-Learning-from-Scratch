
from read_data import*
from sklearn.externals import joblib
from LogRegScr import *

''' Use gnbScratch class as a classifier'''

features, Y= read_data('../Data/part_C_train.h5')
labels=np.zeros((Y.shape[0]))

for k in range(0,Y.shape[0]):
    labels[k]=Y[k].argmax(axis=0)



''' No of k-folds are defined in 'set'. Experiments can be performed using different number of folds'''

set=np.array([5,10])

accuracy=np.zeros((len(set)))

for n in range (0,len(set)):


    print('Working with {} folds\n'.format(set[n]))



    pivot=labels.shape[0]//set[n]

    features_train=np.zeros((features.shape[0]-pivot,features.shape[1]))
    labels_train=np.zeros((labels.shape[0]-pivot))

    best_accuracy=0




    features_test=np.zeros((pivot,features.shape[1]))
    labels_test=np.zeros((pivot))




    for k in range(0,set[n]):


        features_train=np.delete(features, np.s_[pivot*k:pivot*(k+1)],0)
        labels_train=np.delete(labels, np.s_[pivot*k:pivot*(k+1)],0)

        features_test=features[pivot*k:pivot*(k+1)]
        labels_test=labels[pivot*k:pivot*(k+1)]

        model=LogRegScr()

        model.fit(features_train, labels_train, 1000, 10)

        predictions=model.predict(features_test)



        accuracy[n] += np.sum(predictions==labels_test)/(set[n]*pivot)


        '''model with best accuracy is saved'''

    if (accuracy[n]>best_accuracy):

        print('saving the best model with {} accuracy'.format(accuracy[n]))
        best_accuracy=accuracy[n]
        filename = 'finalized_model.sav'
        joblib.dump(model, filename)

#load the saved model

'''
loaded_model = joblib.load(filename)
result = loaded_model.score(features, labels)
print(result)
'''


print(accuracy)
