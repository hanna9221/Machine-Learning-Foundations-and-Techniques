import numpy as np
import pandas as pd

def kNN_predict(trainSet, k, x):
    d = trainSet.shape[1]-1
    X = trainSet.loc[:, :d-1]
    dist_sq = ((X-x)**2).sum(axis=1).sort_values()
    min_k_ones = dist_sq[:k].index.values
    vote = 0
    for i in min_k_ones:
        vote += trainSet.loc[i, d]
    return 1 if vote > 0 else -1

def errorRate(trainSet, k, testSet):
    d = testSet.shape[1]-1
    y = testSet[d]
    y_tilda = testSet.apply(lambda row: kNN_predict(trainSet, k, row[:d]), axis=1)
    return np.sum(y != y_tilda) / testSet.shape[0]
        
trainSet = pd.read_csv('hw4_knn_train.txt', sep=" ", header=None)
testSet = pd.read_csv('hw4_knn_test.txt', sep=" ", header=None)

# 15-16========================================================================
#E_in = errorRate(trainSet, 1, trainSet)
#print('E_in =', E_in)
#E_out = errorRate(trainSet, 1, testSet)
#print('E_out =', E_out)

#E_in = 0.0
#E_out = 0.344

# 17-18========================================================================
E_in = errorRate(trainSet, 5, trainSet)
print('E_in =', E_in)
E_out = errorRate(trainSet, 5, testSet)
print('E_out =', E_out)

#E_in = 0.16
#E_out = 0.316

