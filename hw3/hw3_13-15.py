import random
import numpy as np

def sign(x):
    return 1 if x>=0 else -1

def target_f(x1, x2):
    t = x1*x1 + x2*x2 -0.6
    return sign(t)

def noise(x):
    return 1 if x<=0.9 else -1

def generateData(N):
    output_x = []
    output_y = []
    for i in range(N):
        x1, x2 = random.uniform(-1,1), random.uniform(-1,1)
        y = target_f(x1, x2)
        y_noise = y*noise(random.random())
        output_x.append((1, x1, x2))
        # 14: output_x.append((1, x1, x2, x1*x2, x1*x1, x2*x2))
        output_y.append(y_noise)
    output_x = np.array(output_x)
    output_y = np.array(output_y)
    return output_x, output_y

def computeW(X, Y):
    X_t = X.transpose()
    A_inv = np.linalg.inv(X_t.dot(X))
    w = A_inv.dot(X_t).dot(Y)
    return w
    
def errorRate(w, X, Y):
    N = len(X)
    temp = X.dot(w)
    for i in range(N):
        temp[i] = sign(temp[i])
    diff = temp - Y
    E_in = np.inner(diff, diff)/(4*N)
    return E_in

def main(N, times):
    total_E_in = 0
    for i in range(times):
        X, Y = generateData(N)
        w = computeW(X,Y)
        #15: (new data) X, Y = generateData(N)
        total_E_in += errorRate(w,X,Y)
    return total_E_in/times

print(main(1000,1000))
