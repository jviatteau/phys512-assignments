import numpy as np
import matplotlib.pyplot as plt
from corner import corner

plt.ion()
pars = ['H0', 'ombh2', 'omch2','tau','As','ns'] #Parameters list
chain = np.loadtxt('planck_chain_tauprior.txt') #Load whichever chain we want to analyze
#Make corner plot of the results
figure = corner(chain, labels=pars,show_titles=True)
figure.show()
figure.savefig('corner.png')
plt.figure()

#For each parameter, print the value and error
for i in range(len(pars)):     
    psd = np.abs(np.fft.rfft(chain[:,i]))**2 #Fourier transform to check convergence
    plt.loglog(np.linspace(0,1,len(psd)), psd) #Plot transform
    value = np.mean(chain[:,i]) #Get value
    uncert = np.std(chain[:,i]) #Get error
    print(f'{pars[i]} is {value} +/- {uncert}') #Print value +/- error

#Importance sampling
print('With importance sampling :')
#Get the array of weights for the whole chain
weights = np.exp(-(chain[:,3]-0.054)**2/(2*0.0074**2))
for i in range(len(pars)):
    value = np.sum(chain[:,i]*weights)/np.sum(weights) #Weighted mean
    weights_norm = weights/np.sum(weights) #To determine error
    uncert = np.std(chain[:,i])*np.sqrt(np.sum(weights_norm**2))
    print(f'{pars[i]} is {value} +/- {uncert}') #Print value +/- error
   
plt.show()
#plt.savefig('converged_tauprior.png')
