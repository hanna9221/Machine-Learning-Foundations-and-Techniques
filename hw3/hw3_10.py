from math import exp

def gradient_E(u,v):
    return (exp(u)+v*exp(u*v)+2*u-2*v-3, 
            2*exp(2*v)+u*exp(u*v)-2*u+4*v-2)

def Hessian_E(u,v):
    return (exp(u)+(v**2)*exp(u*v)+2,
            exp(u*v)+u*v*exp(u*v)-2,
            4*exp(2*v)+(u**2)*exp(u*v)+4)

u, v = 0, 0
for i in range(5):
    g1, g2 = gradient_E(u,v)
    h1, h2, h3 = Hessian_E(u,v)
    u -= (g1*h3-g2*h2)/(h1*h3-h2**2)
    v -= (g2*h1-g1*h2)/(h1*h3-h2**2)

print(u,v)

def E(u,v):
    return exp(u)+exp(2*v)+exp(u*v)+u**2-2*u*v+2*v**2-3*u-2*v

print(E(u,v))
