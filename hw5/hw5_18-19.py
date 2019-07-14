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
test_x, test_y = readData(path_test)

for j in range(len(y)):
    if y[j]==0: y[j] = 1
    else: y[j] = -1
for j in range(len(test_y)):
    if test_y[j]==0: test_y[j] = 1
    else: test_y[j] = -1

# 18===========================================================================
#C = [10**i for i in range(-3,2)]
#for c in C:
#    clf = svm.SVC(gamma=100, C=c)
#    clf.fit(x, y)
#    test_y_tilda = clf.predict(test_x)
#    E_out = errorRate(test_y_tilda, test_y)
#    print('C=', c, 'E_out=', E_out)
#    print('number of support vectors:', clf.n_support_)

# 19===========================================================================
gamma = [10**i for i in range(5)]
for g in gamma:
    clf = svm.SVC(gamma=g, C=0.1)
    clf.fit(x, y)
    test_y_tilda = clf.predict(test_x)
    E_out = errorRate(test_y_tilda, test_y)
    print('gamma=', g, 'E_out=', E_out)

