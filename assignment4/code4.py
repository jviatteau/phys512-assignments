import numpy as np
import matplotlib.pyplot as plt
plt.ion()

data = np.load('sidebands.npz')
t=data['time']
d=data['signal']

#Single Lorentzian fit

def calc_lorentz(m, t):
    #Method to calculate the gradient in parameter space
    a = m[0] #The amplitude
    w = m[1] #The width
    t_0 = m[2] #The center
    y = a*w**2 / (w**2+(t-t_0)**2) #The function for the given parameters
    #The analytical derivatives of the Lorentzian with respect to each of the parameters
    dyda = w**2/(w**2+(t-t_0)**2)
    dydw = 2*a*w*(t-t_0)**2/(w**2+(t-t_0)**2)**2
    dydt_0 = 2*a*w**2*(t-t_0)/(w**2+(t-t_0)**2)**2
    #Load the derivatives into an array
    grad = np.zeros([t.size, m.size])
    grad[:, 0] = dyda
    grad[:, 1] = dydw
    grad[:, 2] = dydt_0
    return y, grad #Return the function and gradient at the given parameters

m = np.asarray([1.4, 0.00001, 0.0002]) #Initial guess

#For each step taken, use the gradient to determine the next step
for j in range(10):
    pred, grad = calc_lorentz(m, t)
    r = d - pred #Difference between data and model at current step
    lhs = grad.T@grad
    rhs = grad.T@r
    dm = np.linalg.inv(lhs)@rhs #Solve for the next best step
    m += dm #Add step to parameters

#Plot the figure and residuals

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t*1000, d) #Time in ms to make it more readable
ax[0].plot(t*1000, pred)
ax[1].plot(t*1000, d-pred)
fig.show()
fig.savefig('Analytic_Lorentz.png')

avg_err = np.mean(np.abs(d-pred)) #An estimate of the noise in the measurements
Ninv = 1/(avg_err**2) #The value on the diagonal of the N inverse matrix
cov = np.linalg.inv((grad.T*Ninv)@grad) #Covariance matrix
m_errs = np.diag(cov)**0.5 #Extract errors for each parameter from covariance matrix and print them
print('Single Lorentzian analytic fit :')
print('a = ', m[0], ', w = ', m[1], ', t_0 = ', m[2])
print('Errors are : ', m_errs[0],' for a, ', m_errs[1],' for w, ', m_errs[2], ' for t_0.')


#Single Lorentzian numerical fit
#params = [a, w, t_0]
def lorentz(t, params):
    #Define the function that we will take numerical derivatives of (here a simple Lorentzian)
    return params[0] / (1 + (t-params[2])**2/params[1]**2)

def num_derivs(fun, t, params):
    #Estimate the numerical derivative w.r.t. each parameter
    #Basically ndiff from problem set 1, but for each parameter
    derivs = np.zeros([t.size, params.size])
    for i in range(params.size):
        x = params[i] #i-th parameter becomes x to determine d/dx of the function 
        dx = np.abs((1e-16)**(1/3)*x) #Good enough estimate of optimal dx
        params_up = params.copy() 
        params_up[i] = x+dx #x replaced by x+dx, where x is the i-th parameter
        params_down = params.copy()
        params_down[i] = x-dx #x replaced by x-dx
        derivs[:,i] = (fun(t, params_up)-fun(t, params_down) )/(2*dx) #2-sided derivative stored in array
    return derivs

def calc_lorentz_num(fun, m, t):
    #Same as calc_lorentz above, but we use numerical derivatives given by num_derivs
    y = fun(t, m)
    grad = num_derivs(fun, t, m)
    return y, grad

m = np.asarray([1.4, 0.00001, 0.0002]) #Initial guess

#Exactly the same process as above
for j in range(10):
    pred, grad = calc_lorentz_num(lorentz, m, t)
    r = d - pred
    lhs = grad.T@grad
    rhs = grad.T@r
    dm = np.linalg.inv(lhs)@rhs
    m += dm

#Print best-fit parameters and plot the model and data
print('Single Lorentzian numerical fit :')
print('a = ', m[0], ', w = ', m[1], ', t$_0$ = ', m[2])

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t*1000, d)
ax[0].plot(t*1000, pred)
ax[1].plot(t*1000, d-pred)
fig.show()
fig.savefig('Numerical_lorentzian.png')

#Triple Lorentzian numerical fit
p = ['a', 'w', 't_0', 'b', 'c', 'dt'] #List of parameters in the order of the array, useful for printing later
def lorentz_triple(t, params):
    #Here, define the sum of 3 Lorentzians using the appropriate parameters
    A = params[0] / (1 + (t-params[2])**2/params[1]**2)
    B = params[3] / (1 + (t-params[2]+params[5])**2/params[1]**2)
    C = params[4] / (1 + (t-params[2]-params[5])**2/params[1]**2)
    return A+B+C


#Take best-fit parameters from above as initial guesses for a, w and t_0 and append guesses for b, c and dt
m = np.append(m, [0.2, 0.2, 5e-5]) 

#From here it's the same as above, only the function fed into calc_lorentz_num is different
for j in range(10):
    pred, grad = calc_lorentz_num(lorentz_triple, m, t)
    r = d - pred
    lhs = grad.T@grad
    rhs = grad.T@r
    dm = np.linalg.inv(lhs)@rhs
    m += dm

avg_err = np.mean(np.abs(d-pred)) #An estimate of the noise in the measurements
Ninv = 1/(avg_err**2) #The value on the diagonal of the N inverse matrix
cov = np.linalg.inv((grad.T*Ninv)@grad) #Covariance matrix
m_errs = np.diag(cov)**0.5 #Extract errors for each parameter from covariance matrix and print them

#Print the best-fit parameters and plot the model and data
print('Triple Lorentzian numerical fit :')
for i in range(len(p)):
    print(p[i], ' = ', m[i], '+/-', m_errs[i])

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t*1000, d)
ax[0].plot(t*1000, pred)
ax[1].plot(t*1000, d-pred)
fig.show()
fig.savefig('Triple_lorentian.png')

#(f)
def get_chisq(m, t, d): #Function to get chi squared as a function of parameters
    pred, grad = calc_lorentz_num(lorentz_triple, m, t) #Get the info for the fit
    r = d - pred #Difference between data and model prediction
    avg_err = np.mean(np.abs(r)) #Noise estimate
    Ninv = 1/(avg_err**2) #Value of inverted N matrix
    chisq = Ninv*r.T@r #Chi squared
    return chisq

print('Best-fit chi squared is : ', get_chisq(m, t, d))

m_new = np.random.multivariate_normal(m, cov) #Get a set of parameters slightly offset from the best-fit ones
chisq_new = get_chisq(m_new, t, d) #Determine new chi squared
print("Random offset parameters chi squared is : ",chisq_new)
pred_new = lorentz_triple(t, m_new)
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t*1000, d)
ax[0].plot(t*1000, pred_new)
ax[1].plot(t*1000, d-pred_new)
fig.show()
fig.savefig('f.png')


#MCMC

nstep = 10000 #Number of steps we want to take
chain = np.zeros([nstep, np.size(m)]) #Initialize chain
chain[0] = m_new #Start at the position after a random step from the position we had above
chisq=get_chisq(chain[0], t, d) #Get chi squared at starting position
for i in range(1,nstep):
    #For each step, get a random step using the covariance matrix from (d) and the previous step
    step = np.random.multivariate_normal(chain[i-1,:], cov)
    chisq_new=get_chisq(step, t, d) #Get new chi squared
    accept=np.exp(-0.5*(chisq_new-chisq)) 
    if chisq_new<chisq: #If new chi^2 is lower than old chi^2, accept the step
        chain[i,:] = step #Register the new position
        chisq = chisq_new #Update chi^2
    elif accept>np.random.rand(1): #else, sometimes randomly accept the step
        chain[i,:] = step
        chisq = chisq_new
    else: #else, reject the step
        chain[i,:] = chain[i-1,:] #Next position is the same as current
    if(i%100==0):
        #Print the progress to give me an update
        #and keep me from being anxious about a possible crash
        print('Chain has done ', i, ' steps.')

#Plot the full chain
plt.clf()
plt.plot(chain[:,0])
plt.savefig('MCMC.png')

chain = chain[100:] #Discard the first 100 non-converged points to get the parameter estimates

#Print the results for each parameter
for i in range(len(p)):
    print(p[i], ' = ', np.mean(chain[:,i]), '+/-', np.std(chain[:,i]))










