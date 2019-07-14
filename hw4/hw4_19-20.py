import numpy as np
import os

path_train = os.getcwd() + "/hw4_train.dat"
path_test = os.getcwd() + "/hw4_test.dat"

def readData(path):
    f = open(path, 'r')
    x, y = [], []
    for line in f:
        temp = line.split()
        x.append([1]+[float(t) for t in temp[:-1]])
        y.append(float(temp[-1]))
    f.close()
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y

def computeW(x, y, Lambda):
    x_t = x.transpose()
    A = x_t.dot(x) + Lambda*np.identity(len(x[0]))
    A_inv = np.linalg.inv(A)
    w = A_inv.dot(x_t).dot(y)
    return w

def sign(t): return 1 if t>0 else -1

def errorRate(x, y, w):
    errorCount, l = 0, len(x)
    for i in range(l):
        if sign(np.inner(x[i], w)) != y[i]:
            errorCount += 1
    return errorCount/l

def computeE_cv(D_x, D_y, Lambda):
    total_E_val = 0
    for i in range(5):
        start, end = 40*i, 40*i+40
        val_x, val_y = D_x[start:end], D_y[start:end]
        train_x = np.concatenate((D_x[:start],D_x[end:]))
        train_y = np.concatenate((D_y[:start],D_y[end:]))
        w = computeW(train_x, train_y, Lambda)
        E_val = errorRate(val_x, val_y, w)
        total_E_val += E_val
    E_cv = total_E_val/5
    return E_cv

def number19_20():
    D_x, D_y = readData(path_train)
    test_x, test_y = readData(path_test)
    log_Lambda = [i for i in range(-10,3)]
    res_log, res_E_cv = 0, 1
    for log in log_Lambda:
        Lambda = 10**log
        E_cv = computeE_cv(D_x, D_y, Lambda)
        print(log, E_cv)
        if E_cv <= res_E_cv:
            res_log = log
            res_E_cv = E_cv

# 19===========================================================================
    return res_log, res_E_cv
    
# 20===========================================================================
#    Lambda = 10**res_log
#    w = computeW(D_x, D_y, Lambda)
#    E_in = errorRate(D_x, D_y, w)
#    E_out = errorRate(test_x, test_y, w)
#    return E_in, E_out


print(number19_20())
