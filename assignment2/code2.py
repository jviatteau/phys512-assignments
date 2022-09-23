import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#Problem 2
j=0 #To count the number of evaluations of f(x) that we save

#The below routine was adapted from the in-class coded routine 'integrate' from integrate_adaptive_class in the class repo
def integrate_adaptative(fun,a,b,tol,extra=None):
    x=np.linspace(a,b,5) #Get the points over the interval to estimate the 5-point integral
    dx=x[1]-x[0]
    if extra is None: #If it's the initial call, evaluate every point
        y=fun(x)
    else:
        #If it's a sub-call, evaluate only the points that weren't passed down and re-create the y array
        y = np.array([extra[0], fun(x[1]), extra[1], fun(x[3]), extra[2]])  
        global j
        j+=3 #3 evaluations saved when doing it this way!
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx) #3-point integral
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx #5-point integral
    myerr=np.abs(i1-i2) #Get the discrepancy between the 2 estimates
    if myerr<tol:
        return i2 #If the discrepancy is lower than the tolerance, the estimate is good enough
    else:
        #If the discrepancy is higher, we pass on info to the sub-calls
        mid=x[2] #Midpoint is always the 3rd point of the linspace above
        left_points=np.array([y[0], y[1], y[2]]) #Pass the already calculated values of y to the left estimate
        right_points=np.array([y[2], y[3], y[4]]) #and to the right estimate
        int1=integrate_adaptative(fun,a,mid,tol/2,extra=left_points) #Do the sub-calls
        int2=integrate_adaptative(fun,mid,b,tol/2,extra=right_points)
        return int1+int2 #Return the sum of the left and right sub-calls

funs = [np.sin, np.exp, np.log] #For some examples, compute the number of evaluations saved
for fun in funs:
    ans=integrate_adaptative(fun, 0.5, 5, 1e-6) #Integrate the result
    print('The integral of ', str(fun), 'is', ans, 'computed with ', j,' evaluations saved') #Print the result
    j=0 #Reset the saved evaluations counter

#Problem 1

R=1 #Work with R=1
zz=np.linspace(0, 5, 1001) #Define range of z to include R=1

def integrand(x):
    #Define the function to  integrate using what we determined and disregard the constant coefficient
    num = z-R*np.cos(x)
    denum = (R**2+z**2-2*R*z*np.cos(x))**(3/2)
    return num*np.sin(x)/denum

#Create arrays to store the integrals for each z for each of the routines
ansq=np.array([])
ans=np.array([])
for z in zz:
    Eq=integrate.quad(integrand,0.0, np.pi)[0] #For each z, integrate from 0 to pi using scipy.integrate.quad
    ansq=np.append(ansq, Eq) #Store the result
    if z!=R: #If z is at R, just skip that point and assume the integral is the same as for the point to the left
        E=integrate_adaptative(integrand, 0, np.pi, 1e-6) #Integrate using our routine from problem 2
    ans=np.append(ans, E)#Store the result
#Plot the results
fig1, ax1 = plt.subplots(2, 1, sharex=True)
ax1[0].plot(zz, ans, label='Homemade Integrator')
ax1[0].plot(zz, ansq, label='Quad integrated')
ax1[0].legend()
ax1[0].set(ylabel='Field')
ax1[1].plot(zz, ans-ansq)
ax1[1].set(xlabel='z')
fig1.show()
fig1.savefig('Problem1.png')

#Problem 3

x = np.linspace(0.5, 1, 100) #Get a bunch of x/y points to make our modelling possible
y = np.log2(x)
x_rescale = 4*x-3 #Rescale the 0.5,1 x range to fit into -1,1
chebs = np.polynomial.chebyshev.chebfit(x_rescale, y, 50) #Evaluate the coefficients of the Chebyshev poly model

tol=1e-6 #Max error that we want
deg=0
for cheb in chebs: #For each order, check if the coefficient is lower in absolute value than our max error of 1e-6
    if np.abs(cheb)>tol:
        deg+=1
chebs=chebs[:deg] #Truncate the coefficients that are too low to bring the error over 1e-6

def mylog2(x, chebs):
    m, e = np.frexp(x) #Break the number into mantissa and exponent
    log2m = np.polynomial.chebyshev.chebval(4*m-3, chebs)
    #Use the coefficients above to get the log2 of the mantissa without forgetting to rescale it
    return log2m + e #Add the exponent to the computed log2 of the mantissa

xx=np.linspace(0.1, 10, 1001) #For a range of x, compare the numpy and our natural logarithms

#Plot and format
fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(xx, mylog2(xx, chebs)*np.log(2), label='Homemade ln')
ax[0].plot(xx, np.log(xx), label='Numpy ln')
ax[0].legend()
ax[1].plot(xx, mylog2(xx, chebs)*np.log(2)-np.log(xx))

fig.show()
fig.savefig('Problem3.png')
