import numpy as np
from cvxopt import matrix, solvers

def SVM(x, y):
    N = len(x)
    d = len(x[0])
    Q = np.identity(d+1)
    Q[0][0] = 0
    Q = matrix(Q, tc='d')
    p = matrix(np.array([0]*(d+1)), tc='d')
    A = []
    for i in range(N):
        temp = [-y[i]]
        temp.extend(-y[i]*x[i])
        A.append(temp)
    A = matrix(np.asarray(A), tc='d')
    c = matrix(np.array([-1]*N), tc='d')
    sol = solvers.qp(Q, p, A, c)
    w = []
    for i in range(d+1):
        w.append(sol['x'][i])
    w = np.asarray(w)
    return w

x = np.array([[1,-2],
              [4,-5],
              [4,-1],
              [5,-2],
              [7,-7],
              [7,1]])
y = np.array([-1,-1,-1,1,1,1])

print(SVM(x,y))