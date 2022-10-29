import numpy as np
import matplotlib.pyplot as plt
import camb

plt.ion()
pars_list = ['H0','ombh2','omch2','tau','As','ns']


#get_spectrum method taken from planck_likelihood.py
def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0] 
    return tt[2:]


#2
pars=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95]) #Initial guesses
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1) #Load data
ell=planck[:,0]
spec=planck[:,1]

def get_model(pars, end=len(spec)):
    #Define a method to get the model without having to resize the array each time
    model = get_spectrum(pars)
    return model[:end]

    
#Method for taking numerical derivatives w.r.t. each parameter from the code for PS4
def num_derivs(fun, pars, size, ds):
    derivs = np.zeros([size, pars.size])
    for i in range(pars.size):
        x = pars[i] #i-th parameter becomes x to determine d/dx of the function 
        dx = ds[i] #Passed down derivative step
        pars_up = pars.copy() 
        pars_up[i] = x+dx #x replaced by x+dx, where x is the i-th parameter
        pars_down = pars.copy()
        pars_down[i] = x-dx #x replaced by x-dx
        derivs[:,i] = (fun(pars_up)-fun(pars_down) )/(2*dx) #2-sided derivative stored in array
    return derivs


def calc_newton(fun, pars, ds):
    #Get the model and gradient arrays for a given set of parameters
    y = get_model(pars)
    grad = num_derivs(get_model, pars, np.size(y), ds)
    return y, grad

def get_chisq(pars, data, err):
    #Get chi-squared from the error and parameters
    resid = data - get_model(pars)
    chisq = np.sum((resid/err)**2)
    return chisq

ds = np.abs((1e-16)**(1/3)*pars) #Estimate of the optimal dx for each parameter
ds[3] *= 1e-5 #Tau seems to be especially picky

err = 0.5*(planck[:,2]+planck[:,3]) #Errors array from the data
for j in range(10):
    #Perform a 10-step newton method
    pred, grad = calc_newton(get_model, pars, ds) #Get model and gradient
    r = spec - pred #Determine residuals
    Ninv = (np.diag(1/err**2)) #Noise matrix
    lhs = grad.T@Ninv@grad #Left hand side
    cov = np.linalg.inv(lhs) #Covariance matrix
    rhs = grad.T@Ninv@r #Right hand side
    dm = cov@rhs #Get next step
    pars += dm #Update parameters
    print(f'Newton has done {j+1} steps') #Give update

#Write results to a file
f = open('planck_fit_params.txt', 'w')
f.write('Best fit parameters are :\n')
for i in range(len(pars)):
    f.write(f'{pars_list[i]} = {pars[i]} +/- {np.sqrt(cov[i,i])}\n')
f.close()

#Load the points, taken from planck_likelihood.py
planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3]);
plt.clf()
plt.plot(ell, spec)
plt.plot(ell,pred)
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
plt.show()
#plt.savefig('10-step-newton.png')

#Generate MCMC step using the covariance matrix from Newton's
def generate_step(npars, cov, scale):
    return scale * np.random.multivariate_normal(np.zeros(npars), cov)

def mcmc_runner(filename, init, cov, nstep, data, scale, fun_chisq):
    chain = np.zeros([nstep, np.size(init)]) #Initialize chain
    chain[0] = init
    chisq=fun_chisq(chain[0], data, err) #Get chi squared at starting position
    for i in range(1,nstep):
        step = chain[i-1,:] + generate_step(len(init), cov, scale)
        chisq_new=fun_chisq(step, data, err) #Get new chi squared
        accept=np.exp(-0.5*(chisq_new-chisq)) 
        if chisq_new<chisq: #If new chi^2 is lower than old chi^2, accept the step
            chain[i,:] = step #Register the new position
            chisq = chisq_new #Update chi^2
        elif accept>np.random.rand(1): #else, sometimes randomly accept the step
            chain[i,:] = step
            chisq = chisq_new
        else: #else, reject the step
            chain[i,:] = chain[i-1,:]
        print("Chain has done ", i, " steps") #Give update
    np.savetxt(filename, chain, header=str(pars_list)) #Save data
nstep = 15001

#Run the MCMC without the tau prior
#mcmc_runner('planck_chain.txt', pars, cov, nstep, spec, 0.9, get_chisq)

#Include the polarization tau prior when determining chi squared
def get_chisq_tauprior(pars, data, err):
    resid = data - get_model(pars)
    chisq = np.sum((resid/err)**2)
    return chisq + (pars[3]-0.054)**2/(2*0.0074**2)

#Run the MCMC with the tau prior
mcmc_runner('planck_chain_tauprior.txt', pars, cov, nstep, spec, 0.9, get_chisq_tauprior)





