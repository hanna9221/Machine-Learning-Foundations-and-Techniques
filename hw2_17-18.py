import random

def sign(x): return 1 if x > 0 else -1

def flip(x): return -1 if x > 0.8 else 1

def generateData(dataNum):
    data = []
    for i in range(dataNum):
        x = random.random()*2 - 1
        r = random.random()
        temp = [x, sign(x)*flip(r)]
        data.append(temp)
    return sorted(data, key=lambda d:d[0])

def generateTheta(data):
    output = []
    for i in range(len(data)-1):
        theta = (data[i][0] + data[i+1][0])/2
        output.append(theta)
    output.append(1.1)
    return output

def train(data, arrayOfTheta):
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
            h = [1, theta]
        count = 0
        for d in data: #s=-1
            if sign(theta-d[0]) != d[1]:
                count += 1
        if count < error:
            error = count
            h = [-1, theta]
    return error/l, h

def main(times, dataNum):
    t_E_in, t_E_out = 0, 0
    for i in range(times):
        trainingData = generateData(dataNum)
        thetaData = generateTheta(trainingData)
        r = train(trainingData, thetaData)
        t_E_in += r[0]
        t_E_out += 0.5 + 0.3*r[1][0]*(abs(r[1][1])-1)
    return t_E_in/times, t_E_out/times

print(main(5000,20))

