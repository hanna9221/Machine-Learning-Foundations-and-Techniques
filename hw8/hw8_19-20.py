import numpy as np
import pandas as pd

def k_means(data, k):
    N = data.shape[0]
    d = data.shape[1]-1  # d=9
    centers = data.loc[np.random.choice(N, k, replace=False), :d-1]
    centers = centers.reset_index(drop=True)
    converge = False
    while not converge:
        data[d] = data.apply(
                lambda row: k_means_predict(centers, row[:d]), axis=1)
        pre_centers = centers.copy()
        for i in range(k):
            centers.loc[i, :] = np.mean(
                    trainSet.loc[data[d]==i, :d-1], axis=0).to_numpy()
        converge = ((pre_centers != centers).values.sum() == 0)
    
    error = 0
    for i in range(k):
        center = centers.loc[i, :].to_numpy()
        error += ((data.loc[data[d]==i, :d-1]-center)**2).values.sum()
    
    return centers, error/N, data

def k_means_predict(centers, x):
    return np.argmin(((centers-x)**2).sum(axis=1))

        
trainSet = pd.read_csv('hw4_kmeans_train.txt', sep=" ", header=None)

t = 3
total = 0
for i in range(t):
    centers, E_in, data = k_means(trainSet, 10)
    total += E_in
print('avg_E_in =', total/t)
