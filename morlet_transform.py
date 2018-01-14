import numpy as np
import pylab as plt
%matplotlib


# Creation of the wave (filter)
f = 2
time = np.linspace(-5,5,500)
wave = (1+np.exp(-f**2) - 2*np.exp(-0.75*f**2)) * np.pi**(-1/4) * np.exp(-0.5*time**2) * (np.exp(1j*f*time)-np.exp(-0.5*f**2))
# Randomly generated signal
signal = np.random.random(50000)

# METHOD 1 : Convolution
from scipy.signal import convolve
output1 = convolve(signal,wave, mode="valid")
output1.shape


# METHOD 2 : Linear Algebra
    # creation of the matrix of signal N * T where T is the time windows of 500 points (determine by the size of wave)
signal2 = signal.copy()
if signal2.shape[0]%wave.shape[0] != 0:
    signal2 = signal2[:-(signal2.shape[0]%wave.shape[0])]

signal2 = signal2.reshape(-1,500)
    # Matrix multiplication
output2 = np.dot(wave,signal2.T)


# METHOD 2' : Linear Algebra with steps
    # creation of the matrix of signal N * T with steps
step = 1
signal3 = signal.copy()
if signal3.shape[0]%wave.shape[0] != 0: # TODO only works for certain numbers (check the size of wave, signal and step)
    signal3 = signal3[:-(signal3.shape[0]%wave.shape[0])]

signal3 = np.array([signal3[(step*i):(step*i+500)] for i in np.arange((signal3.shape[0]-500)//step)])
    # Matrix multiplication
output3 = np.dot(wave,signal3.T)
