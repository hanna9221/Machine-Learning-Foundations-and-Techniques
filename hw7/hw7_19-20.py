import numpy as np
import pandas as pd

class TreeNode(object):
    def __init__(self, index, theta, leafValue=None):
        self.index = index
        self.theta = theta
        self.leafValue = leafValue
        self.left = None
        self.right = None

def GiniIndex(data):
    N = data.shape[0]
    d = data.shape[1]-1
    mu_1 = (data[d]==1).sum() / N
    mu_2 = 1 - mu_1
    return 1 - mu_1**2 - mu_2**2

# If cannot branch anymore, the function won't be called.
def decisionStump(data):
    N = data.shape[0]
    d = data.shape[1]-1
    
    impurity = float('inf')
    for i in range(d):
        data = data.sort_values(by=i)
        pre = data.iloc[0, i]
        for j in range(1, N):
            if data.iloc[j, i] != pre:
                data_1 = data.iloc[:j]
                data_2 = data.iloc[j:]
                imp = j*GiniIndex(data_1) + (N-j)*GiniIndex(data_2)
                if imp < impurity:
                    impurity = imp
                    theta = (pre + data.iloc[j, i]) / 2
                    index = i
                    leftSet = data_1
                    rightSet = data_2
                pre = data.iloc[j, i]
    return index, theta, leftSet, rightSet

def decisionTree_pruned(data):
    d = data.shape[1]-1
    # stop condition
    if GiniIndex(data) == 0:
        return TreeNode(None, None, data.iloc[0, d])
    
    index, theta, leftSet, rightSet = decisionStump(data)
    leftValue = leftSet[d].mode()[0]
    rightValue = rightSet[d].mode()[0]
    node = TreeNode(index, theta)
    node.left = TreeNode(None, None, leftValue)
    node.right = TreeNode(None, None, rightValue)
    return node

def countNodes(root):
    if root.leafValue: return 0
    return countNodes(root.left) + countNodes(root.right) + 1

# predict single row of data
def dt_predict(root, x):
    if root.leafValue: return root.leafValue
    index, theta = root.index, root.theta
    if x[index] <= theta:
        return dt_predict(root.left, x)
    else:
        return dt_predict(root.right, x)

def dt_errorRate(root, data):
    N = data.shape[0]
    d = data.shape[1]-1
    count = 0
    for i in range(N):
        if dt_predict(root, data.iloc[i]) != data.iloc[i, d]:
            count += 1
    return count / N

def bootstrap(data):
    N = data.shape[0]
    indices = np.random.randint(0, N, N)
    return data.loc[indices]

def randomForest(data, T):
    ret = []
    for t in range(T):
        data_temp = bootstrap(data)
        root = decisionTree_pruned(data_temp)
        ret.append(root)
    return ret

def rf_predict(roots, x):
    T = len(roots)
    count = 0
    for root in roots:
        if dt_predict(root, x) == 1:
            count += 1
    return 1 if count > T/2 else -1

def rf_errorRate(roots, data):
    N = data.shape[0]
    d = data.shape[1]-1
    count = 0
    for i in range(N):
        if rf_predict(roots, data.iloc[i]) != data.iloc[i, d]:
            count += 1
    return count / N


#A = [[0,0,-1],
#     [1,1,-1],
#     [2,0,-1],
#     [4,-1,-1],
#     [5,-1,-1],
#     [4,0,1],
#     [5,0,1],
#     [4,1,1],
#     [5,1,1]]
#A = pd.DataFrame(A)
#root = decisionTree_pruned(A)
#print(countNodes(root))
#print(root.index, root.theta, root.leafValue)
#print(root.left.index, root.left.theta, root.left.leafValue)
#print(root.right.index, root.right.theta, root.right.leafValue)


trainSet = pd.read_csv('hw3_train.txt', sep=" ", header=None)
testSet = pd.read_csv('hw3_test.txt', sep=" ", header=None)

# The experiment was expected to repeat for 100 times, 
# but that will take too much time.
times = 3
E_in, E_out = 0, 0
for i in range(times):
    roots = randomForest(trainSet, 300)
    E_in += rf_errorRate(roots, trainSet)
    E_out += rf_errorRate(roots, testSet)
print('avg_E_in =', E_in/times, ', avg_E_out =', E_out/times)

#avg_E_in = 0.11333333333333333 , avg_E_out = 0.151
