import math
d = 50
delta = 0.5
N = 5
def m(n): return n**d


oVC = math.sqrt(8/N * math.log(4*m(2*N) / delta))
RPB = math.sqrt(2*math.log(2*N*m(N)) / N) \
    + math.sqrt(2/N * math.log(1/delta)) + 1/N
PVB = (2 + math.sqrt(4 + 4*N*math.log(6*m(2*N)/delta))) / (2*N)
Dev = (1 + math.sqrt(1 + (N-2)/2 * (math.log(4/delta) + 2*d*math.log(N)))) \
      / (N-2)
vVC = math.sqrt(16/N * math.log(2*m(N) / math.sqrt(delta)))

print(oVC)
print(RPB)
print(PVB)
print(Dev)
print(vVC)