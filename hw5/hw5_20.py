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
for j in range(len(y)):
    if y[j]==0: y[j] = 1
    else: y[j] = -1

shuffle = [i for i in range(len(y))]
gamma = [i for i in range(5)]
res = [0]*5

for i in range(100):
    np.random.shuffle(shuffle)
    x_val, x_train = [], []
    y_val, y_train = [], []
    for j in shuffle[:1000]:
        x_val.append(x[j])
        y_val.append(y[j])
    for j in shuffle[1000:]:
        x_train.append(x[j])
        y_train.append(y[j])
    
    min_E_val = 1
    res_g = 0
    for g in gamma:
        clf = svm.SVC(gamma=10**g, C=0.1)
        clf.fit(x_train, y_train)
        y_tilda = clf.predict(x_val)
        E_val = errorRate(y_tilda, y_val)
        if E_val < min_E_val:
            min_E_val = E_val
            res_g = g
    res[res_g] += 1

print(res)
# res = [6, 76, 18, 0, 0], thus the answer is gamma=10.
