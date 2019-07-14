import numpy as np
import os
from sklearn import svm

path_train = os.getcwd() + "/features.train.txt"
path_test = os.getcwd() + "/features.test.txt"

def readData(path):
    f = open(path, 'r')
    x, y = [], []
    for line in f:
        temp = line.split()
        x.append([float(t) for t in temp[1:]])
        y.append(float(temp[0]))
    f.close()
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y

def errorRate(y_tilda, y):
    return np.sum(y_tilda != y) / len(y)


x, y = readData(path_train)

E_in = np.zeros(10)
sum_alpha = np.zeros(10)

for i in range(0, 9, 2):
    y_i = y.copy()
    for j in range(len(y)):
        if y[j]==i: y_i[j] = 1
        else: y_i[j] = -1
    
    clf = svm.SVC(kernel='poly', degree=2, coef0=1, gamma=1, C=0.01)
    clf.fit(x, y_i)
    y_tilda = clf.predict(x)
    E_in[i] = errorRate(y_tilda, y_i)
    sum_alpha[i] = np.sum(np.abs(clf.dual_coef_[0]))

for i in range(10):
    if E_in[i] != 0:
        print(i, E_in[i])
        print(i, sum_alpha[i])
        print('-------------------')

