import random

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

def PLA(data):
    allCorrect = False
    count = 0
    w = [0,0,0,0,0]
    while not allCorrect:
        allCorrect = True
        for x in data:
            if sign(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]+w[3]*x[3]+w[4]*x[4]) != x[5]:
                allCorrect = False
                count += 1
                for i in range(5):
                    w[i] = w[i] + x[5]*x[i]
    return count

location = "C:/Users/hannah/.spyder-py3/CheChe's notes/ML by Lin/PLAdata.txt"
data = readData(location)
total = 0

for i in range(2000):
    total += PLA(randomizeData(data))
print(total/2000)
