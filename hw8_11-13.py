import numpy as np
import pandas as pd


def initiateWeights(d, M, r):
    W1 = np.random.uniform(-r, r, (d+1, M))
    W2 = np.random.uniform(-r, r, (M+1, 1))
    return W1, W2
    
def nnet(data, M, r, T, eta):
    N = data.shape[0]
    d = data.shape[1]-1
    W1, W2 = initiateWeights(d, M, r)
    
    for i in range(T):
        n = np.random.randint(0, N)
        y = data.loc[n, d]
        x0 = data.loc[n, :d-1].to_numpy()
        x0 = np.insert(x0, 0, 1)
        s1 = x0.dot(W1)
        x1 = np.tanh(s1)
        x1 = np.insert(x1, 0, 1)
        s2 = x1.dot(W2)
        x2 = np.tanh(s2)
        delta2 = 2 * (x2-y) * (1-x2**2)
        delta1 = delta2[0] * W2[1:, 0] * (1-np.tanh(s1)**2)
        
        x0 = np.asmatrix(x0)
        x1 = np.asmatrix(x1)
        delta1 = np.asmatrix(delta1)
        delta2 = np.asmatrix(delta2)
        W1 = np.asarray(W1 - eta * x0.transpose().dot(delta1))
        W2 = np.asarray(W2 - eta * x1.transpose().dot(delta2))
    
    return W1, W2

def nnet_predict(W, x):
    x = x.to_numpy()
    for i in range(len(W)):
        x = np.insert(x, 0, 1)
        s = x.dot(W[i])
        x = np.tanh(s)
    return np.sign(x[0])

def errorRate(W, data):
    d = data.shape[1]-1
    y = data[d]
    y_tilda = data.apply(lambda row: nnet_predict(W, row[:d]), axis=1)
    return np.sum(y != y_tilda) / data.shape[0]
        
trainSet = pd.read_csv('hw4_nnet_train.txt', sep=" ", header=None)
testSet = pd.read_csv('hw4_nnet_test.txt', sep=" ", header=None)
T = 50000

# I did each experiment for only 1 time instead of 500 times.
# 11===========================================================================
#eta, r = 0.1, 0.1
#M_set = [1, 6, 11, 16, 21]
#for M in M_set:
#    W = nnet(trainSet, M, r, T, eta)
#    E_out = errorRate(W, testSet)
#    print('M =', M)
#    print('E_out =', E_out)

# M = 1, E_out = 0.472
# M = 6, E_out = 0.036
# M = 11, E_out = 0.036
# M = 16, E_out = 0.04
# M = 21, E_out = 0.04
# Since more neurons may cause overfitting, M = 6 is a better choice.


# 12===========================================================================
#eta, M = 0.1, 3
#r_set = [0, 0.1, 10, 100, 1000]
#for r in r_set:
#    W = nnet(trainSet, M, r, T, eta)
#    E_out = errorRate(W, testSet)
#    print('r =', r, ', ', 'E_out =', E_out)

#r = 0 ,  E_out = 0.472
#r = 0.1 ,  E_out = 0.036
#r = 10 ,  E_out = 0.472
#r = 100 ,  E_out = 0.464
#r = 1000 ,  E_out = 0.364


# 13===========================================================================
r, M = 0.1, 3
eta_set = [0.001, 0.01, 0.1, 1, 10]
for eta in eta_set:
    W = nnet(trainSet, M, r, T, eta)
    E_out = errorRate(W, testSet)
    print('eta =', eta, ', ', 'E_out =', E_out)

#eta = 0.001 ,  E_out = 0.068
#eta = 0.01 ,  E_out = 0.036
#eta = 0.1 ,  E_out = 0.036
#eta = 1 ,  E_out = 0.472
#eta = 10 ,  E_out = 0.472

