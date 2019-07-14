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

def decisionTree(data):
    # stop condition
    if GiniIndex(data) == 0:
        return TreeNode(None, None, data.iloc[0, data.shape[1]-1])
    
    index, theta, leftSet, rightSet = decisionStump(data)
    node = TreeNode(index, theta)
    node.left = decisionTree(leftSet)
    node.right = decisionTree(rightSet)
    return node

def countNodes(root):
    if root.leafValue: return 0
    return countNodes(root.left) + countNodes(root.right) + 1

# predict single data
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
       

trainSet = pd.read_csv('hw3_train.txt', sep=" ", header=None)
testSet = pd.read_csv('hw3_test.txt', sep=" ", header=None)

root = decisionTree(trainSet)
print('number of nodes =', countNodes(root))
E_in = dt_errorRate(root, trainSet)
E_out = dt_errorRate(root, testSet)
print('E_in =', E_in, ', E_out =', E_out)
