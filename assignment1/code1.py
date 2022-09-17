import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

plt.ion()


ef = 1e-16

#Problem 1

dx = np.logspace(-10, 2, 1001) #Get a range of possible dx
def fun(x): #The example function
    return np.exp(0.01*x)

#Code inspired by the code in num_derivs_clean.py available in the course repo
x0=1 #Point of interest
y1=fun(x0+dx) #We evaluate the function at the 4 points proposed
ym=fun(x0-dx)
y2l=fun(x0+2*dx)
y2m=fun(x0-2*dx)
d=(8*y1-8*ym-y2l+y2m)/(12*dx) #Estimate the derivative how we assumed
plt.clf()
plt.loglog(dx,np.abs(d-0.01*np.exp(0.01*x0))) #Plot the error for each dx to make the optimal dx apparent
plt.show()
plt.savefig('fig1b2.png')

#Problem 2

plt.clf()
fun = np.sin
x = 12.0

def near0(x): 
    if(-1.0<x<1.0): #For x near 0, we can't use that as curvature scale, so we instead set it to 1.0 for that region
        return 1.0
    else:
        return np.abs(x) #Curvature scale can't be a negative number, since it is a scale

def ndiff(fun,x,full=False):
    
    dx = ef**(1/3)*near0(x) #Determine optimal dx following eq. 5.7.8 in numerical recipes
    d = (fun(x+dx)-fun(x-dx))/(2*dx) #Get estimate of derivative using that dx
    err = np.abs(d)*ef**(2/3) #Estimate error following eq. 5.7.9 in Numerical recipes
    if(full==False):
        return d
    else:
        return d, dx, err

d, dx, err = ndiff(fun, x, full=True)

print('Estimated derivative is', d,', while true derivative is', np.cos(x),', error is estimated to be', err, 'while it really is', np.abs(d-np.cos(x)))

#Problem 3

data = np.transpose(np.loadtxt('lakeshore.txt')) #Transpose data so we have all T in one array, all V in another array
def lakeshore(V, data):
    y = data[0] #y-axis is temperature
    x = data[1] #x-axis is voltage
    V = np.array([V]).flatten() #flatten the input V in case it is an array, make it into a single-value array if input is a number
    dydx = 1/(data[2]*0.001) #For each voltage, get the slope recorded as dT/dV (with the appropriate unit conversion)
    T = np.array([]) #Make arrays to return
    Err = np.array([])
    for v in V: 
        i = np.argmin(np.abs(x-v)) #Find the recorded point that is closest to input voltage
        t = dydx[i]*(v-x[i])+y[i] #Determine the temperature by a linear interpolation using the recorded dT/dV at that point
        T = np.append(T, t) #Append result
        err = (dydx[i]*(x[i+1]-x[i])+y[i] - y[i+1]) #Estimate error as the difference between the interpolation and the nearest measurement
        Err = np.append(Err, err) #Append result
    return T, Err #Return array of temperatures and array of errors

V = np.linspace(data[1][2], data[1][-2], 2001) #Take V values in the measurement range
T, Err = lakeshore(V, data) #Interpolate T for each of the V and estimate the error on them
#Plot the data and format
plt.clf()
plt.plot(data[1],data[0],'*',label='Data')
plt.plot(V, T, label='Interpolation')
plt.xlabel('Voltage')
plt.ylabel('Temperature')
plt.legend()
plt.show()
plt.savefig('Problem3T.png')
plt.clf()
plt.plot(V, Err, label='Residuals')
plt.legend()
plt.show()
plt.savefig('Problem3err.png')

#Problem 4

plt.clf()
def lorentz(x):
    return 1.0/(1.0+x**2)

#Define the function to generate the points to interpolate from
fun = lorentz
x = np.linspace(-1, 1, 7) #We take 7 points
y = fun(x) 
xx=np.linspace(x[0],x[-1],2001)
yy=fun(xx)

#Spline interpolation
#Code taken from the class slides 'interpolate_integrate'
spln=interpolate.splrep(x,y) #Find the spline by looking at each point and its neighbors 
yspline=interpolate.splev(xx,spln) #Evaluate the spline found above
plt.plot(xx,yspline-yy, label='Spline interpolation')

#Rational function interpolation
#Code taken from the class slides 'interpolate_integrate'
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q
n=4
m=4
p,q=rat_fit(x,y,n,m)
pred=rat_eval(p,q,x)

yrat=rat_eval(p,q,xx)
plt.plot(xx,yrat-yy, label='Rational function interpolation')



#Polynomial interpolation
#Code taken from the file ratclass.py available on the course repo
poly_coeffs=np.polyfit(x,y,len(y)-1) #Determine the coeffs that make a polynomial of degree # of points -1, which we can construct so
                                     #it passes through all the given points
y_poly=np.polyval(poly_coeffs,xx) #Evaluate the polynomial
plt.plot(xx,y_poly-yy, label='Polynomial interpolation')


plt.legend()
plt.show()
plt.savefig('Problem4lorentz2.png')









