import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#Problem 1

c0 = 1/np.exp(np.arctan(-20)) #The constant required for the analytical result

def fun(x, y):
    dydx= y/(1+x**2) #The ODE to solve
    return dydx

def rk4_step(fun, x, y, h): #The rk4 solver as given in class, in rk4.py in the repo
    k1=h*fun(x,y) #Get each of the coefficients as given by the Taylor expansion of y(x+h)
    k2=h*fun(x+h/2, y+k1/2)
    k3=h*fun(x+h/2, y+k2/2)
    k4=h*fun(x+h, y+k3)
    y_next = y+(k1+2*k2+2*k3+k4)/6 #Add the y from the step
    return y_next

def rk4_stepd(fun, x, y, h):
    y1 = rk4_step(fun, x, y, h) #Get the y from making a step of h
    y2 = rk4_step(fun, x, y, h/2) 
    y2 = rk4_step(fun, x+h/2, y2, h/2) #Get the y from making two steps of h/2
    delta = y2-y1 #Get the difference between the two methods
    y_next = y2 + delta/15 #Eliminate the 5th order error
    return y_next 

n_steps = 200 #For the 4th order rk4, take 200 steps
xx = np.linspace(-20, 20, n_steps+1) #Array of x at each step
yy = np.empty(n_steps+1)
yy[0] = 1.0 #Initialize the y's with the value at x=-20
h = xx[1]-xx[0] #Step size
for i in range(n_steps):
    x = xx[i]
    y = yy[i]
    yy[i+1]= rk4_step(fun, x, y, h) #For each step, record the next y value

n_stepsd = n_steps//3 #Take less steps for the fifth order method
xxd = np.linspace(-20, 20, n_stepsd+1) #Initialize in the same way
yyd = np.empty(n_stepsd+1)
yyd[0] = 1.0
hd = xxd[1]-xxd[0]
for i in range(n_stepsd):
    x = xxd[i]
    y = yyd[i]
    yyd[i+1]= rk4_stepd(fun, x, y, hd) #For each step, record the next y value

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(xx, yy, label='4th order') #Fourth order
ax[0].plot(xxd, yyd, label='5th order') #Fifth order
yy_real =c0*np.exp(np.arctan(xx)) #Analytic result for the 200 steps case
yyd_real = c0*np.exp(np.arctan(xxd)) #Analytic result for the 200//3 steps case
ax[0].plot(xx, yy_real, label='Analytic result')
ax[0].legend()
ax[1].plot(xx, yy-yy_real) #Residuals
ax[1].plot(xxd, yyd-yyd_real)

plt.savefig('Problem1.png')
plt.clf()

#Problem 2
#Time conversion factors to make writing the half-lives easier
hour = 3600
day = 24*hour 
year = 365.25*day
#Half-lives array, in order of the decay chain
half_lives = np.asarray([4.468e9*year, 24.1*day, 6.7*hour, 245500*year, 75380*year])
                        #U238           Th234     Pa234     U234        Th230
half_lives = np.append(half_lives, np.array([1600*year,3.8235*day,3.1*60,26.8*60,19.9*60]))
                                             #Ra226     Rn222     Po218   Pb214  Bi214  
half_lives = np.append(half_lives, np.array([164.3e-6, 22.3*year, 5.015*year, 138.376*day]))
                                            #Po214     Pb210      Bi210       Po210
x0 = 0 #Initial time
x1 = 1e20 #Final time
y0=np.asarray([1]+[0]*14) #Initial composition (only U238)

def decay(x, y):
    dydx=np.zeros(half_lives.size+1)
    dydx[0]=-y[0]/half_lives[0] #Only way for U238 to disappear is to decay, none can be created
    for i in range(1, half_lives.size):
        #For intermediate products, the earlier product decaying gives a positive term, otherwise it decays same as U238
        dydx[i] = y[i-1]/half_lives[i-1] - y[i]/half_lives[i]
    dydx[half_lives.size]=y[half_lives.size-1]/half_lives[-1] #Finally, Pb206 can only be created since it is stable
    return dydx #Return the array of quantities of each product

#Solve the initial value problem by using scipy.integrate, same as in lecture.
#Use the Radau method to minimize the number of evaluations needed, since the typical RK4 would be too demanding.
ans_stiff = integrate.solve_ivp(decay, [x0, x1], y0,method='Radau')
y = ans_stiff.y #Array of complete curves for each product
#Below, avoid first point in both cases to avoid having to divide by 0 (we constrain those values anyway)
Pb206_to_U238_ratio = y[-1,1:]/y[0,1:] #Pb206/U238 ratio over time
Th230_to_U234_ratio = y[4,1:]/y[3,1:] #Th230/U234 ratio over time
t = ans_stiff.t #The times for each evaluations

#Plot the ratios and format
plt.loglog(t[1:], Pb206_to_U238_ratio)
plt.savefig('Problem2_Pb_to_U238.png')
plt.clf()
plt.loglog(t[1:], Th230_to_U234_ratio)
plt.savefig('Problem2_Th230_to_U234.png')
plt.clf()


#Problem 3

data = np.loadtxt('dish_zenith.txt').T #Load and transpose
x = data[0] #Separate by axes
y = data[1]
z = data[2]
A = np.empty([x.size,4]) #Initialize A matrix
for i in range(x.size):
    #For each point, fill the approriate row of A with the info at that data point
    A[i, 0] = 1
    A[i, 1] = x[i]
    A[i, 2] = y[i]
    A[i, 3] = x[i]**2+y[i]**2

#Fitting code taken from 'polyfit_class.py' in the class repo
lhs = A.T@A
rhs = A.T@z
fits = np.linalg.inv(lhs)@rhs
#Get the original parameters using the fit parameters
a = fits[3]
x_0 = -fits[1]/(2*a)
y_0 = -fits[2]/(2*a)
z_0 = fits[0]-a*(x_0**2+y_0**2)
print('a is ', a, ', x_0 is ', x_0, ', y_0 is ', y_0, ', z_0 is ', z_0) #Answer to 3.b)
#Make a 2d surface to plot z over it
x_fit = np.linspace(np.min(x), np.max(x), 1000)
y_fit = np.linspace(np.min(y), np.max(y), 1000)
xx, yy = np.meshgrid(x_fit, y_fit)
z_fit = a*((xx-x_0)**2 + (yy-y_0)**2) + z_0 #Get the fit values for the whole surface

#Show data points and continuous fit on the same plot
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z)
ax.plot_surface(xx, yy, z_fit, cmap='autumn', alpha=0.4)
plt.savefig('Problem3.png')

error = np.linalg.inv(A.T@A) #The covariance matrix, estimate errors from it
print('a is ', a,' with an uncertainty of ', np.sqrt(error[3,3])) #Answers to 3.c)
print('Focal length is just', 1/(4*a),' with error ', np.sqrt(error[3,3])/(4*a**2))


