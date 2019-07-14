import numpy as np
from math import exp

def kernel(x_i, x_j, ga):
    return exp(-ga * np.sum((x_i-x_j)**2))

def KRR(X, Y, ga, la): 
    N = len(X)
    A = np.zeros((N, N)) # A = lambda*I + K
    for i in range(N):
        x_i = X[i]
        for j in range(i, N):
            x_j = X[j]
            temp = kernel(x_i, x_j, ga)
            A[i, j] = temp
            A[j, i] = temp
        A[i, i] += la
    beta = np.linalg.inv(A).dot(Y)
    return beta, X

def predict(beta, ga, X, x):
    N, m = len(beta), len(x)
    y_temp = np.zeros((m, N))
    for i in range(m):
        x_i = x[i]
        for j in range(N):
            x_j = X[j]
            y_temp[i, j] = kernel(x_i, x_j, ga)
    y = np.sign(y_temp.dot(beta))
    return y
    
def errorRate(y_tilda, y):
    return np.sum(y_tilda != y) / len(y)


data = np.loadtxt('hw2_lssvm_all.txt')
X, Y = data[:, :-1], data[:, -1]
X_train, X_test = X[:400], X[400:]
Y_train, Y_test = Y[:400], Y[400:]

Gamma = [32, 2, 1/8]
Lambda = [0.001, 1, 1000]

for ga in Gamma:
    for la in Lambda:
        beta, X = KRR(X_train, Y_train, ga, la)
        E_in = errorRate(predict(beta, ga, X, X_train), Y_train)
        E_out = errorRate(predict(beta, ga, X, X_test), Y_test)
        print('gamma=', ga, ', lambda=', la)
        print('E_in=', E_in, ', E_out=', E_out)
        print('-------------')

