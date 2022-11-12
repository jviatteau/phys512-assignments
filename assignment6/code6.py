import numpy as np
import matplotlib.pyplot as plt

plt.ion()

#Problem 1
def dirac_delta(i, size):
    #Get a Dirac delta array, which is actually a Kronecker delta since it's discrete
    arr = np.zeros(size)
    arr[i] = 1 #Put a 1 where you want it
    return arr

def conv(f, g):
    #Convolution method, taken from the lecture slides
    ft1=np.fft.fft(f) #Get Fourier transforms and multiply in Fourier space
    ft2=np.fft.fft(g)
    return np.real(np.fft.ifft(ft1*ft2)) #Return inverse FT


def shift_conv(arr, amount):
    #Method to shift an array by some amount
    delta = dirac_delta(amount, len(arr)) #Get a delta shifted from 0 by the amount we want to shift by
    shifted = conv(delta, arr) #Convolve input with delta to shift
    return shifted/shifted.sum() #Return the normalized shifted array

def gauss(x, mu, sig):
    #Get a Gaussian array
    gauss = np.exp(-(x-mu)**2/(2*sig**2))
    return gauss/gauss.sum() #Normalize it

N=10001 #Size of the array
x = np.linspace(-10, 10, N) #x-range
gaussian = gauss(x, 0, 1) #Gaussian with 0 mean, so peak is at center of the array
delta = dirac_delta(len(gaussian)//2, len(gaussian)) #The delta array that will be used by shift_conv 
shifted = shift_conv(gaussian, len(gaussian)//2) #Shift input by half the length

#Plot it all
plt.plot(x, gaussian, label='original array')
plt.plot(x, shifted, label='shifted array')
plt.plot(x, delta, label='delta array')
plt.legend()
plt.savefig('1.png')
plt.clf()

#Problem 2a

def corr(arr1, arr2):
    #Correlation as given 
    cor = np.fft.ifft(np/fft.fft(arr1)*np.conj(np.fft.fft(arr2)))
    return cor/cor.sum() #Normalize

corr_gauss = corr(gaussian, gaussian) #Get the correlation of the Gaussian with itself
corr_gauss = np.fft.fftshift(corr_gauss) #Shift it to put the peak at the center

#Plot input and output
plt.plot(x, gaussian, label='Original Gaussian')
plt.plot(x, np.real(corr_gauss), label='Correlation Gaussian')
plt.legend()
plt.savefig('2a.png')
plt.clf()

#Problem 2b

def shift_cor(arr, amount):
    #Combine our shift method with the correlation
    shifted = shift_conv(arr, amount)
    cor = corr(shifted, shifted)
    return cor #Already normalized by corr

fig, ax = plt.subplots(2, 1)
shifts = [N-1, N//2, N//3, N//4, N//5, N//10] #Get a variety of shifts
for shift in shifts:
    to_plot = shift_cor(gaussian, shift) #Get correlation for each shift
    ax[0].plot(np.abs(shift_conv(gaussian, shift))) #Plot input/output
    ax[1].plot(np.abs(to_plot), label = str(shift)+' shift')
    
fig.legend()
fig.show()
fig.savefig('2b.png')

#Problem 3

def add_zeroes(arr):
    #Method for adding zeros at the end of the input
    zeros = np.zeros(arr.size)
    arr = np.append(arr, zeros)
    return arr

def conv_nowrap(arr1, arr2):
    n = len(arr1) #Keep track of the length of the original arrays
    arr1 = add_zeroes(arr1) #Add zeros to both
    arr2 = add_zeroes(arr2)
    to_return = conv(arr1, arr2) #Convolute the extended arrays
    return to_return[n//2:3*n//2] #Discard low (+/-) frequencies

wrap = conv(gaussian, gaussian) #Try it with a Gaussian, with ordinary conv
no_wrap = conv_nowrap(gaussian, gaussian) #and our no-wrap conv

#Plot results
plt.plot(wrap, label='Ordinary convolution')
plt.plot(no_wrap, label='No-wrap convolution')
plt.legend()
plt.show()
plt.savefig('3.png')

#Problem 4

def sine_transform(k1, k2, N):
    #Method for getting analytic transform as determined in the pdf
    k = k1-k2
    transform = (1 - np.exp(2*np.pi*1j*k))/(1 - np.exp(2*np.pi*1j*k/N))
    k = k1+k2
    transform -= (1 - np.exp(-2*np.pi*1j*k))/(1 - np.exp(-2*np.pi*1j*k/N))
    transform = np.abs(transform)
    return transform/transform.sum()

N = 101 
k1 = 21 #Set k of sine such that a non-integer amount of periods is contained in the array
k2 = np.linspace(0, N, N) #k range for analytic transform
x = np.linspace(0, N, N) #x range for numerical numpy transform
sine = 2j*np.sin(2*np.pi*k1*x/N) #sine for numerical transform
analytic_transform = sine_transform(k1, k2, N) #Get analytic transform
numeric_transform = np.fft.fft(sine) #Get numeric transform and normalize
numeric_transform = np.abs(numeric_transform)
numeric_transform /= numeric_transform.sum()

#Plot the results
plt.plot(analytic_transform,label='Analytic transform')
plt.plot(numeric_transform, label='Numerical Transform')
plt.legend()
plt.savefig('4c.png')
plt.clf()

window = 0.5-0.5*np.cos(2*np.pi*x/N) #Define the window function
win_trans = np.fft.fft(window) #Transform the window
sine_win = sine*window #Get windowed array
window_transform = np.abs(np.fft.fft(sine_win)) #Transform the windowed array
window_transform /= window_transform.sum() #Normalize

#Plot the result
plt.plot(numeric_transform, label='Without the window')
plt.plot(np.abs(window_transform), label='With the window')
plt.legend()
plt.savefig('4d.png')
plt.clf()









