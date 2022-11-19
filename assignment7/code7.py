import numpy as np
import matplotlib.pyplot as plt
import random

plt.ion()

#Problem 1

#Load the generated points
points = np.transpose(np.loadtxt('rand_points2.txt'))

#The code below is to generate a 3D scatter plot of the points
#Commented out because it's easier to see the planes on a 2D plot
#points = points[:,:10000]
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(points[0], points[1], points[2],marker='o', s=(72./fig.dpi)**2)
#fig.show()

#a and b values derived from looking at the 3d scatter plot
a = -0.153188
b = 0.073168

#Evaluate the x-coordinate for the 2D plot
x = a*points[0] + b*points[1]
z = points[2] #y-coordinate is the z in 3D

#Plot the figure and save it
fig = plt.figure()
ax=fig.add_subplot()
ax.plot(x, z, 'o', ms=72./fig.dpi) #Make it so each marker is a pixel on the fig
fig.show()
fig.savefig('1a.png')

#Generate the same number of random points using the standard python library
points = np.empty((np.size(points)//3, 3))
for i in range(len(points)):
    for j in range(3):
        points[i][j] = random.uniform(0, 2**31) #Generate 3 random numbers for each coord
points = np.transpose(points)


#Plot the data on a 3D scatter plot and move around to see a pattern
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[0], points[1], points[2],marker='o', s=(72./fig.dpi)**2)
fig.show()
fig.savefig('1b.png')

#Problem 2

def lorentz(x):
    #Get a standard Lorentzian
    return 1/(1+x**2) 

def rand_lorentz(n):
    #Generate a set of Lorentzian-distributed random numbers
    #Taken from rng.ipynb, code shown at a tutorial
    q = np.pi*(np.random.rand(n)-0.5)
    return np.abs(np.tan(q)) #Take absolute value since we're only after numbers>0

#Check if the used Lorentzian is always above the exponential of interest
N=10000
M = 1.01 #Coefficient of the Lorentzian
x = np.linspace(0, 10, N) #Range of points
plt.figure()
plt.plot(x, M*lorentz(x), label='Lorentzian') #Coefficient times Lorentzian
plt.plot(x, np.exp(-x), label='Exponential')#Exponential of interest
plt.legend()
plt.show()
plt.savefig('2a.png')


n = 1000000
ys = rand_lorentz(n) #Generate Lorentzian-distributed array

fs = np.exp(-ys) #For each value, compare the exponential value
gs = lorentz(ys) #with the Lorentzian value

#For each value, generate a random number. If the random number is above the ratio
#of exponential to Lorentzian (which is always <1 as we make sure using M),
#accept the deviate as an exponential deviate. If not, reject it.
accept=(np.random.rand(n))< fs/(M*gs)
#Accept fraction is just the mean of the accept array
#(accept array has 1 for accepted, 0 for rejected)
print(f'Accept fraction is {np.mean(accept)}')  

y_use = ys[accept] #Keep only the accepted deviates

#Plotting code adapted from rand_arctan.py on the course repo
aa,bb = np.histogram(y_use, np.linspace(0, 10, 101)) #Make a histogram
b_cent=0.5*(bb[1:]+bb[:-1]) #Get the bin centers
pred=np.exp(-b_cent) #Evaluate the prediction at the beam centers
pred = pred/pred.sum() #Normalize
aa = aa/aa.sum() #Normalize
#Plot the prediction and histogram
plt.figure()
plt.plot(b_cent, pred,'r')
plt.bar(b_cent, aa, 0.08)
plt.show()
plt.savefig('2b.png')


#Problem 3

#First, plot the box in u,v where we want to sample from
u = np.linspace(0.00001, 1, 100000) 
v = -2*u*np.log(u) #Comes from inverting the u = sqrt(exp(-v/u)) relation
#Plot the box
plt.figure()
plt.plot(u, v, 'b')
plt.plot(u, [0]*len(u), 'b') #Close off at 0, since we don't want to sample negative values
plt.show()
plt.savefig('3a.png')

u = np.random.rand(n) #Get random u between 0 and 1
v = 0.75*np.random.rand(n) #Get random v for the appropriate box
r = v/u #Get the ratio
umax = np.exp(-r/2) #Max u depends on the ratio we got

accept = u<umax #Accept if u falls within accepted range
r_use = r[accept] #Use the accepted ratios
print(f'Accept fraction is {np.mean(accept)}') #Print accepted fraction

#Same plotting code as above
aa,bb = np.histogram(r_use, np.linspace(0, 10, 101))
b_cent=0.5*(bb[1:]+bb[:-1])
pred=np.exp(-b_cent)
pred = pred/pred.sum()
aa = aa/aa.sum()
plt.figure()
plt.plot(b_cent, pred,'r')
plt.bar(b_cent, aa, 0.08)
plt.show()
plt.savefig('3b.png')





