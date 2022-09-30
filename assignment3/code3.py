import numpy as np
import matplotlib.pyplot as plt

#Problem 1
'''
c0 = 1/np.exp(np.arctan(-20))

def fun(x, y):
    dydx= y/(1+x**2)
    return dydx

def rk4_step(fun, x, y, h):
    k1=h*fun(x,y)
    k2=h*fun(x+h/2, y+k1/2)
    k3=h*fun(x+h/2, y+k2/2)
    k4=h*fun(x+h, y+k3)
    y_next = y+(k1+2*k2+2*k3+k4)/6
    return y_next

def rk4_stepd(fun, x, y, h):
    y1 = rk4_step(fun, x, y, h)
    y2 = rk4_step(fun, x, y, h/2)
    y2 = rk4_step(fun, x+h/2, y2, h/2)
    delta = y2-y1
    y_next = y2 + delta/15
    return y_next

n_steps = 200
xx = np.linspace(-20, 20, n_steps+1)
yy = np.empty(n_steps+1)
yy[0] = 1.0
h = xx[1]-xx[0]
for i in range(n_steps):
    x = xx[i]
    y = yy[i]
    yy[i+1]= rk4_step(fun, x, y, h)
plt.plot(xx, yy)

yy_real =c0*np.exp(np.arctan(xx))
plt.plot(xx, yy_real)

n_stepsd = n_steps//3
xxd = np.linspace(-20, 20, n_stepsd+1)
yyd = np.empty(n_stepsd+1)
yyd[0] = 1.0
hd = xxd[1]-xxd[0]
for i in range(n_stepsd):
    x = xxd[i]
    y = yyd[i]
    yyd[i+1]= rk4_stepd(fun, x, y, hd)

plt.plot(xxd, yyd)

plt.show()
'''
#Problem 2
hour = 3600
day = 24*hour #in seconds
year = 365.25*day 
half_lives = np.asarray([4.468e9*year, 24.1*day, 6.7*hour, 245500*year, 75380*year,1600*year,3.8235*day,3.1*60,26.8*60,19.9*60])
half_lives = np.append(half_lives, np.array([164.3e-6, 22.3*year, 5.015*year, 138.376*day]))

print(half_lives)
print(half_lives.size)



#Problem 3
