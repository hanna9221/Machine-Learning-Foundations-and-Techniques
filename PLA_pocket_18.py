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

#呼叫函數太多次會變慢,如果函數裡面有迴圈,會變超級慢= =
#def innerProduct(w, x):
#    return w[0]*x[0]+w[1]*x[1]+w[2]*x[2]+w[3]*x[3]+w[4]*x[4]

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
    w_pocket = [0,0,0,0,0]
    w_pocketER = errorRate(trainingData, w_pocket)
    while not allCorrect and count < bound:
        allCorrect = True
        for x in trainingData:
            if sign(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]+w[3]*x[3]+w[4]*x[4]) != x[5]:
                allCorrect = False
                count += 1
                for i in range(5):
                    w[i] += x[5]*x[i]
                w_ER = errorRate(trainingData, w)
                if w_ER < w_pocketER:
                    w_pocket = list.copy(w)
                    w_pocketER = w_ER
                if count > bound:
                    return w_pocket
    return w_pocket

start = time.time()
trainingData = readData("C:/Users/hannah/.spyder-py3/CheChe's notes/ML by Lin/PLA_train.txt")
testData = readData("C:/Users/hannah/.spyder-py3/CheChe's notes/ML by Lin/PLA_test.txt")
bound = 50
total = 0
for i in range(2000):
    temp = randomizeData(trainingData)
    total += errorRate(testData, pocket(temp, bound))
print(total/2000)
elapsed = time.time() - start
print('elapsed', elapsed)
