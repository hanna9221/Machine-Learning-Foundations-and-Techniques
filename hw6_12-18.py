import numpy as np
import pandas as pd


def decisionStump(data, u):
    d = data.shape[1]-1
    N = data.shape[0]
    y = data[d].to_numpy()
    sum_u = np.sum(u)
    err_all_p = np.sum((y==-1)*u)
    res_E_in = 1
    # res_func = [s, i, theta] => s*sign(x_i-theta)
    res_func = [0, 0, 0]
    
    for i in range(d):
        data = data.sort_values(by=i)
        err = err_all_p
        pre = -float('inf')
        for j in range(N):
            if data.iloc[j, i] != pre:
                if err/sum_u < res_E_in and err/sum_u <= 0.5:
                    res_E_in = err/sum_u
                    res_func = [1, i, (data.iloc[j, i]+pre)/2]
                elif 1-err/sum_u < res_E_in and err/sum_u >= 0.5:
                    res_E_in = 1-err/sum_u
                    res_func = [-1, i, (data.iloc[j, i]+pre)/2]
                pre = data.iloc[j, i]
            if data.iloc[j, d] == -1:
                err -= u[data.index[j]]
            else: err += u[data.index[j]]
    return res_E_in, res_func

def adaBoost(data, T):
    g_t = []
    alpha_t = []
    u = np.array([1/data.shape[0]]*data.shape[0])
    y = data[data.shape[1]-1].to_numpy()
    for j in range(T):
        E_t, func = decisionStump(data, u)
#        print('i=', i, 'err=', E_t, 'func=', func)
        g_t.append(func)
        temp = np.sqrt((1-E_t)/E_t)
        alpha_t.append(np.log(temp))
        s, i, theta = func
        y_tilda = s*np.sign(data[i]-theta).to_numpy()
        u = np.where(y_tilda!=y, u*temp, u/temp)

# 14 & 15======================================================================
#        print('U_',j+2, '=', np.sum(u))
    
    return np.array(alpha_t), np.array(g_t)

def predict(data, alpha_t, g_t):
    N = data.shape[0]
    T = len(alpha_t)
    y_t = pd.DataFrame(np.tile([0]*T, (N, 1)))
    for j in range(T):
        s, i, theta = g_t[j]
        y_t[j] = s*np.sign(data[i]-theta).to_numpy()
    y = np.sign(y_t.dot(alpha_t))
    
    return y
        
def errorRate(y_tilda, y):
    return np.sum(y_tilda != y) / len(y)


data_train = pd.read_csv('hw2_adaboost_train.txt', 
                         sep=" ", header=None)
data_test = pd.read_csv('hw2_adaboost_test.txt', 
                         sep=" ", header=None)

# 12 & 13======================================================================
# 12: set T=1, 13: set T=300
#alpha_t, g_t = adaBoost(data_train, 300)
#y = predict(data_train, alpha_t, g_t)
#E_in = errorRate(y, data_train[2])
#print(E_in)

# 16===========================================================================
#alpha_t, g_t = adaBoost(data_train, 300)
#alpha = max(alpha_t)
#min_err = 1 / (1+np.exp(2*alpha))
#print(min_err)

# 17 & 18======================================================================
# 17: set T=1, 18: set T=300
alpha_t, g_t = adaBoost(data_train, 300)
y = predict(data_test, alpha_t, g_t)
E_out = errorRate(y, data_test[2])
print(E_out)



