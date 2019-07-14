import numpy as np
import pandas as pd


def initiateWeights(d, M1, M2, r):
    W1 = np.random.uniform(-r, r, (d+1, M1))
    W2 = np.random.uniform(-r, r, (M1+1, M2))
    W3 = np.random.uniform(-r, r, (M2+1, 1))
    return W1, W2, W3
    
def nnet(data, M1, M2, r, T, eta):
    N = data.shape[0]
    d = data.shape[1]-1
    W1, W2, W3 = initiateWeights(d, M1, M2, r)
    
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
        x2 = np.insert(x2, 0, 1)
        s3 = x2.dot(W3)
        x3 = np.tanh(s3)
        delta3 = 2 * (x3-y) * (1-x3**2)
        delta2 = delta3[0] * W3[1:, 0] * (1-x2[1:]**2)
        delta1 = (1-x1[1:]**2) * W2[1:, :].dot(delta2)
        
        x0 = np.asmatrix(x0)
        x1 = np.asmatrix(x1)
        x2 = np.asmatrix(x2)
        W1 = np.asarray(W1 - eta * x0.transpose().dot([delta1]))
        W2 = np.asarray(W2 - eta * x1.transpose().dot([delta2]))
        W3 = np.asarray(W3 - eta * x2.transpose().dot([delta3]))
        
    return W1, W2, W3

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

eta, r = 0.01, 0.1
t = 5
total = 0
for i in range(t):
    W = nnet(trainSet, 3, 8, r, T, eta)
    total += errorRate(W, testSet)
print('avg_E_out =', total/t)

