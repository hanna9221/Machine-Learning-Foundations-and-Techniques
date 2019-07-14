import numpy as np
from cvxopt import matrix, solvers

def kernel(x1, x2):
    return (1+np.dot(x1, x2))**2

def dualSVM(x, y):
    N = len(x)
    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Q[i][j] = y[i]*y[j]*kernel(x[i], x[j])
    Q = matrix(Q, tc='d')
    p = matrix(np.array([-1]*N), tc='d')
    A = (np.zeros((N+2, N)))
    for j in range(N):
        A[0][j] = -y[j]
        A[1][j] = y[j]
        A[j+2][j] = -1
    A = matrix(A, tc='d')
    c = matrix(np.array([0]*(N+2)), tc='d')
    
    sol = solvers.qp(Q, p, A, c)
    alpha = []
    for i in range(N):
        alpha.append(sol['x'][i])
    alpha = np.asarray(alpha)
    return alpha

x = np.array([[1,0],
              [0,1],
              [0,-1],
              [-1,0],
              [0,2],
              [0,-2],
              [-2,0]])
y = np.array([-1,-1,-1,1,1,1,1])
alpha = dualSVM(x, y)
print(alpha)

b = y[1]
for i in range(7):
    b -= y[i]*alpha[i]*kernel(x[i], x[1])


