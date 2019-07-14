import pandas as pd
import numpy as np
from math import exp

path_train = "C:/Users/hannah/.spyder-py3/CheChe's notes/ML by Lin/hw3/hw3_train.dat"
path_test = "C:/Users/hannah/.spyder-py3/CheChe's notes/ML by Lin/hw3/hw3_test.dat"
with open(path_train,'r') as f:
    train = pd.DataFrame(l.split() for l in f)
with open(path_test,'r') as f:
    test = pd.DataFrame(l.split() for l in f)

eta, T = 0.001, 2000


def readData(data):
    x, y = [], []
    for index, row in data.iterrows():
        temp = row.tolist()
        x.append([1]+[float(t) for t in temp[:-1]])
        y.append(float(temp[-1]))
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y

def GD(x, y, bound):
    N, d = len(x), len(x[0])
    w = np.array([0]*d)
    for i in range(bound):
        v = np.array([0]*d)
        for n in range(N):
            v = v + y[n]/(1 + exp(y[n]*w.dot(x[n])))*x[n]
        v = v/N
        w = w + eta*v
    return w

def SGD(x, y, bound):
    N, d = len(x), len(x[0])
    w = np.array([0]*d)
    for i in range(bound):
        for n in range(N):
            w = w + eta * y[n]/(1 + exp(y[n]*w.dot(x[n])))*x[n]
    return w

def sign(t): return 1 if t>0 else -1

def errorRate(x, y, w):
    errorCount, l = 0, len(x)
    for i in range(l):
        if sign(np.inner(x[i], w)) != y[i]:
            errorCount += 1
    return errorCount/l

def main(train, test, eta, bound):
    train_x, train_y = readData(train)
    test_x, test_y = readData(test)
    w = SGD(train_x, train_y, bound)
    ER_train = errorRate(train_x, train_y, w)
    ER_test = errorRate(test_x, test_y, w)
    print('w = ' , w)
    print('ER_train = ', ER_train)
    print('ER_test = ', ER_test)

#18, 19
#main(train, test, eta, T) 

#20
main(train, test, eta, 2)
