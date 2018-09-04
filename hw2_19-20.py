
def readData(location):
    f = open(location, "r")
    data = []
    for line in f:
        temp = line.split()
        for i in range(10):
            temp[i] = float(temp[i])
        data.append(temp)
    f.close()
    return data

def sign(x): return 1 if x > 0 else -1

def singleDimension(data, i):
    output = []
    for j in range(len(data)):
        output.append([data[j][i], data[j][-1]])
    return sorted(output, key=lambda d:d[0])

def generateTheta(data):
    output = []
    for i in range(len(data)-1):
        theta = (data[i][0] + data[i+1][0])/2
        output.append(theta)
    output.append(data[-1][0]+1)
    return output

def train(data, arrayOfTheta, k):
    l = len(arrayOfTheta)
    error = float('inf')
    for i in range(l):
        theta = arrayOfTheta[i]
        count = 0 #E_in of this round
        for d in data: #s=1
            if sign(d[0]-theta) != d[1]: 
                count += 1
        if count < error:
            error = count
            h = [1, k, theta]
        count = 0
        for d in data: #s=-1
            if sign(theta-d[0]) != d[1]:
                count += 1
        if count < error:
            error = count
            h = [-1, k, theta]
    return error/l, h

def main(data):
    E_in = float('inf')
    output = []
    for i in range(9):
        tempData = singleDimension(data, i)
        tempTheta = generateTheta(tempData)
        r = train(tempData, tempTheta, i)
        if r[0] < E_in: #find the lowest E_in
            E_in = r[0]
            output = r
    return output

def test_E_out(data, h):
    count = 0
    s, i, theta = h[0], h[1], h[2]
    for d in data:
        if s*sign(d[i]-theta) != d[-1]:
            count += 1
    return count/len(data)

trainLoc = "C:/Users/hannah/.spyder-py3/CheChe's notes/ML by Lin/hw2_19_trainingData.txt"
trainingData = readData(trainLoc)
result = main(trainingData)
h = result[1]
testLoc = "C:/Users/hannah/.spyder-py3/CheChe's notes/ML by Lin/hw2_19_testingData.txt"
testingData = readData(testLoc)
print(result[0])
print(test_E_out(testingData, h))

