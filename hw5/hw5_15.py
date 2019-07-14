import numpy as np
import os
from sklearn import svm
import matplotlib.pyplot as plt

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

x, y = readData(path_train)

for i in range(len(y)):
    if y[i]==0: y[i] = 1
    else: y[i] = -1

clf = svm.SVC(kernel='linear', C=0.01)
clf.fit(x, y)
w = clf.coef_[0]
b = clf.intercept_[0]
print('w = ', w)
print('|w| = ', np.linalg.norm(w))
print('b = ', b)

xx = np.linspace(0, 0.7)
yy = (b + w[0]*xx) / -w[1]
h0 = plt.plot(xx, yy, 'k-')
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
