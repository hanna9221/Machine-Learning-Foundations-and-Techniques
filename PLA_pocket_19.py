import random
import time

def readData(location):
    f = open(location, "r")
    data = []
    for line in f:
        temp = line.split()
        for i in range(5):
            temp[i] = float(temp[i])
        temp.insert(0,1)
        data.append(temp)
    f.close()
    return data

def sign(n): return 1 if n>0 else -1

def randomizeData(data):
    output = list.copy(data)
    random.shuffle(output)
    return output

def errorRate(data, w):
    errorCount = 0
    for x in data:
        if sign(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]+w[3]*x[3]+w[4]*x[4]) != x[5]:
            errorCount += 1
    return errorCount/len(data)

def pocket(trainingData, bound):
    allCorrect = False
    count = 0
    w = [0,0,0,0,0]
    while not allCorrect and count < bound:
        allCorrect = True
        for x in trainingData:
            if sign(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]+w[3]*x[3]+w[4]*x[4]) != x[5]:
                allCorrect = False
                count += 1
                for i in range(5):
                    w[i] += x[5]*x[i]
                if count > bound:
                    return w
    return w

start = time.time()
trainingData = readData("your_path/PLA_train.txt")
testData = readData("your_path/PLA_test.txt")
bound = 50
total = 0
for i in range(2000):
    total += errorRate(testData, pocket(randomizeData(trainingData), bound))
print(total/2000)
elapsed = time.time() - start
print('elapsed:', elapsed)
